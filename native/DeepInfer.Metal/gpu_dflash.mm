// Native DFlash2 drafter (Issue80). Opt-in: nothing here allocates or compiles until di_dflash_create runs.
// All weights stay BF16 (read as float), activations/accumulation/KV are FP32, RMSNorm scales are DIRECT BF16->float.
static NSString *dflash_source = @R"metal(
#include <metal_stdlib>
using namespace metal;
struct GemmP {uint n,k,m,xs,ys;};
// y[m][n] = sum_k bf16(w[n][k]) * x[m][k]; one simdgroup per output row, 8 rows per grid-y tile.
kernel void df_gemm(device const ushort *w [[buffer(0)]],device const float *x [[buffer(1)]],device float *y [[buffer(2)]],
 constant GemmP &p [[buffer(3)]],uint2 tg [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 uint n=tg.x*4+sg;if(n>=p.n)return;uint m0=tg.y*8,rows=min(8u,p.m-m0);
 float acc[8]={0,0,0,0,0,0,0,0};device const ushort *wr=w+ulong(n)*p.k;
 for(uint k=lane*4;k<p.k;k+=128){
  ushort4 b=*(device const ushort4*)(wr+k);
  float4 wf=float4(as_type<float>(uint(b.x)<<16),as_type<float>(uint(b.y)<<16),as_type<float>(uint(b.z)<<16),as_type<float>(uint(b.w)<<16));
  for(uint r=0;r<8;r++){if(r<rows){float4 xv=*(device const float4*)(x+ulong(m0+r)*p.xs+k);acc[r]+=wf.x*xv.x+wf.y*xv.y+wf.z*xv.z+wf.w*xv.w;}}
 }
 for(uint r=0;r<8;r++){float s=simd_sum(acc[r]);if(lane==0&&r<rows)y[ulong(m0+r)*p.ys+n]=s;}
}
struct RmsP {uint dim,xs,ys;float eps;};
kernel void df_rms(device const float *x [[buffer(0)]],device const float *w [[buffer(1)]],device float *y [[buffer(2)]],constant RmsP &p [[buffer(3)]],
 uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float part[8];float s=0;
 for(uint d=t;d<p.dim;d+=256){float v=x[ulong(row)*p.xs+d];s+=v*v;}
 s=simd_sum(s);if(lane==0)part[sg]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
 float tot=0;for(uint i=0;i<8;i++)tot+=part[i];float sc=rsqrt(tot/float(p.dim)+p.eps);
 for(uint d=t;d<p.dim;d+=256)y[ulong(row)*p.ys+d]=x[ulong(row)*p.xs+d]*sc*w[d];
}
struct NrP {uint heads,srcStride,dstStride,firstPos,ringMod;float eps;};
// Per-head RMSNorm (direct scale) then non-traditional full-width RoPE (pairs i,i+64) with a host double-precision table.
// ringMod!=0 writes row r to slot (firstPos+r)%ringMod (committed-context ring); otherwise to row r.
kernel void df_norm_rope(device const float *src [[buffer(0)]],device const float *w [[buffer(1)]],device const float2 *rope [[buffer(2)]],
 device float *dst [[buffer(3)]],constant NrP &p [[buffer(4)]],uint2 tg [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float part[4];threadgroup float buf[128];
 uint h=tg.x,r=tg.y;float v=src[ulong(r)*p.srcStride+h*128+t];float s=simd_sum(v*v);if(lane==0)part[sg]=s;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float sc=rsqrt((part[0]+part[1]+part[2]+part[3])/128.0f+p.eps);buf[t]=v*sc*w[t];threadgroup_barrier(mem_flags::mem_threadgroup);
 uint pos=p.firstPos+r;float2 cs=rope[ulong(pos)*64+(t&63)];
 float o=t<64?buf[t]*cs.x-buf[t+64]*cs.y:buf[t]*cs.x+buf[t-64]*cs.y;
 uint drow=p.ringMod?pos%p.ringMod:r;dst[ulong(drow)*p.dstStride+h*128+t]=o;
}
struct RcP {uint srcStride,firstPos,ringMod;};
kernel void df_ring_copy(device const float *src [[buffer(0)]],device float *dst [[buffer(1)]],constant RcP &p [[buffer(2)]],uint2 at [[thread_position_in_grid]]) {
 dst[ulong((p.firstPos+at.y)%p.ringMod)*1024+at.x]=src[ulong(at.y)*p.srcStride+at.x];
}
struct ConvP {uint rows,dynWhich,baseWhich,accumulate;};
// out[m][c] (+)= sum_off (base[which][off][c]*x[m-off][c]) + (dyn[m][which*256+off*128+c/16]*x[m-off][c]), zero padded for m<off.
kernel void df_dynconv(device const float *x [[buffer(0)]],device const float *dyn [[buffer(1)]],device const float *base [[buffer(2)]],
 device float *out [[buffer(3)]],constant ConvP &p [[buffer(4)]],uint2 at [[thread_position_in_grid]]) {
 uint c=at.x,m=at.y;float s=0;
 for(uint off=0;off<2;off++){
  if(m<off)continue;float xv=x[ulong(m-off)*2048+c];
  s+=base[p.baseWhich*4096+off*2048+c]*xv;s+=dyn[ulong(m)*512+p.dynWhich*256+off*128+c/16]*xv;
 }
 if(p.accumulate)out[ulong(m)*2048+c]+=s;else out[ulong(m)*2048+c]=s;
}
kernel void df_silu_mul(device float *gate [[buffer(0)]],device const float *up [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=n)return;float g=gate[i];gate[i]=(g/(1.0f+exp(-g)))*up[i];
}
kernel void df_copy(device const float *src [[buffer(0)]],device float *dst [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<n)dst[i]=src[i];}
kernel void df_tap(device const float *x [[buffer(0)]],device float *tap [[buffer(1)]],constant uint2 &p [[buffer(2)]],uint2 at [[thread_position_in_grid]]) {
 tap[ulong(at.y)*16384+p.y*2048+at.x]=x[ulong(at.y)*2048+at.x];
}
struct AttnP {uint nctx,ctxBegin,nblk,qbase,rows,nparts;float scale;};
// Partition = 128 logical keys: committed ring context (visibility per query sliding window) then the proposal block (always visible).
kernel void df_attn_part(device const float *q [[buffer(0)]],device const float *ck [[buffer(1)]],device const float *cv [[buffer(2)]],
 device const float *bk [[buffer(3)]],device const float *bv [[buffer(4)]],device float *pmax [[buffer(5)]],device float *psum [[buffer(6)]],
 device float *pacc [[buffer(7)]],constant AttnP &p [[buffer(8)]],uint2 tg [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
 threadgroup float sc[4][8][128];
 uint part=tg.x,kvh=tg.y,head=kvh*4+sg,total=p.nctx+p.nblk;
 float4 qv[8];
 for(uint r=0;r<8;r++)qv[r]=r<p.rows?*(device const float4*)(q+ulong(r)*4096+head*128+lane*4):float4(0);
 for(uint j=0;j<128;j++){
  uint J=part*128+j;float4 kv=float4(0);
  if(J<total){
   if(J<p.nctx)kv=*(device const float4*)(ck+ulong((p.ctxBegin+J)&2047)*1024+kvh*128+lane*4);
   else kv=*(device const float4*)(bk+ulong(J-p.nctx)*1024+kvh*128+lane*4);
  }
  for(uint r=0;r<8;r++){
   if(r>=p.rows)continue;
   float d=simd_sum(dot(qv[r],kv));
   bool visible=J<total&&(J>=p.nctx||int(p.ctxBegin+J)>=int(p.qbase)+int(r)-2047);
   if(lane==0)sc[sg][r][j]=visible?d*p.scale:-INFINITY;
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float4 acc[8];for(uint r=0;r<8;r++)acc[r]=float4(0);
 for(uint r=0;r<p.rows;r++){
  float m=-INFINITY;for(uint j=lane;j<128;j+=32)m=max(m,sc[sg][r][j]);m=simd_max(m);
  float s=0;
  for(uint j=lane;j<128;j+=32){float e=m==-INFINITY?0.0f:exp(sc[sg][r][j]-m);sc[sg][r][j]=e;s+=e;}
  s=simd_sum(s);
  if(lane==0){ulong at=(ulong(part)*8+r)*32+head;pmax[at]=m;psum[at]=s;}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=0;j<128;j++){
  uint J=part*128+j;if(J>=total)break;
  float4 vv=J<p.nctx?*(device const float4*)(cv+ulong((p.ctxBegin+J)&2047)*1024+kvh*128+lane*4):*(device const float4*)(bv+ulong(J-p.nctx)*1024+kvh*128+lane*4);
  for(uint r=0;r<8;r++)if(r<p.rows)acc[r]+=sc[sg][r][j]*vv;
 }
 for(uint r=0;r<p.rows;r++)*(device float4*)(pacc+((ulong(part)*8+r)*32+head)*128+lane*4)=acc[r];
}
kernel void df_attn_combine(device const float *pmax [[buffer(0)]],device const float *psum [[buffer(1)]],device const float *pacc [[buffer(2)]],
 device float *out [[buffer(3)]],constant AttnP &p [[buffer(4)]],uint2 tg [[threadgroup_position_in_grid]],uint d [[thread_index_in_threadgroup]]) {
 uint head=tg.x,r=tg.y;float gm=-INFINITY;
 for(uint pt=0;pt<p.nparts;pt++)gm=max(gm,pmax[(ulong(pt)*8+r)*32+head]);
 float s=0,a=0;
 for(uint pt=0;pt<p.nparts;pt++){
  ulong at=(ulong(pt)*8+r)*32+head;float m=pmax[at];if(m==-INFINITY)continue;
  float f=exp(m-gm);s+=psum[at]*f;a+=pacc[at*128+d]*f;
 }
 out[ulong(r)*4096+head*128+d]=a/s;
}
inline void df_topk16(thread float *v,thread uint *id,device float *ov,device uint *oi,uint t,uint lane,uint sg,threadgroup float *pv,threadgroup uint *pi) {
 uint removed=0;
 for(uint k=0;k<16;k++){
  float bv=-INFINITY;uint bi=0xFFFFFFFFu;
  for(uint j=0;j<8;j++)if(!((removed>>j)&1u)){if(v[j]>bv||(v[j]==bv&&id[j]<bi)){bv=v[j];bi=id[j];}}
  for(uint off=16;off>=1;off>>=1){
   float ov2=simd_shuffle_xor(bv,off);uint oi2=simd_shuffle_xor(bi,off);
   if(ov2>bv||(ov2==bv&&oi2<bi)){bv=ov2;bi=oi2;}
  }
  if(lane==0){pv[sg]=bv;pi[sg]=bi;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float gv=pv[0];uint gi=pi[0];
  for(uint i=1;i<8;i++){if(pv[i]>gv||(pv[i]==gv&&pi[i]<gi)){gv=pv[i];gi=pi[i];}}
  for(uint j=0;j<8;j++)if(!((removed>>j)&1u)&&id[j]==gi&&v[j]==gv&&gi!=0xFFFFFFFFu)removed|=1u<<j;
  if(t==0){ov[k]=gv;oi[k]=gi;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
struct TkP {uint vocab,chunks;};
kernel void df_topk1(device const float *logits [[buffer(0)]],device float *s1v [[buffer(1)]],device uint *s1i [[buffer(2)]],constant TkP &p [[buffer(3)]],
 uint2 tg [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float pv[8];threadgroup uint pi[8];float v[8];uint id[8];
 for(uint j=0;j<8;j++){uint e=tg.x*2048+t+256*j;if(e<p.vocab){v[j]=logits[ulong(tg.y)*p.vocab+e];id[j]=e;}else{v[j]=-INFINITY;id[j]=0xFFFFFFFFu;}}
 ulong o=(ulong(tg.y)*p.chunks+tg.x)*16;df_topk16(v,id,s1v+o,s1i+o,t,lane,sg,pv,pi);
}
kernel void df_topk2(device const float *s1v [[buffer(0)]],device const uint *s1i [[buffer(1)]],device float *cv [[buffer(2)]],device uint *ci [[buffer(3)]],
 constant TkP &p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float pv[8];threadgroup uint pi[8];float v[8];uint id[8];uint n=p.chunks*16;
 for(uint j=0;j<8;j++){uint e=t+256*j;if(e<n){v[j]=s1v[ulong(row)*n+e];id[j]=s1i[ulong(row)*n+e];}else{v[j]=-INFINITY;id[j]=0xFFFFFFFFu;}}
 df_topk16(v,id,cv+row*16,ci+row*16,t,lane,sg,pv,pi);
}
struct SelP {uint rows;uint anchor;};
// Sequential trained edge selector: score[c]=unary+sum_r predCB[pred][r]*hproj[p][r]*succCB[cand][r]; first maximum wins.
kernel void df_selector(device const float *hproj [[buffer(0)]],device const uint *ci [[buffer(1)]],device const float *cvv [[buffer(2)]],
 device const ushort *pcb [[buffer(3)]],device const ushort *scb [[buffer(4)]],device int *path [[buffer(5)]],device float *scores [[buffer(6)]],
 constant SelP &p [[buffer(7)]],uint t [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup uint pred;threadgroup float sc[16];
 if(t==0)pred=p.anchor;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint r=0;r<p.rows;r++){
  uint cand=ci[r*16+sg];device const ushort *pr=pcb+ulong(pred)*256;device const ushort *su=scb+ulong(cand)*256;
  float s=0;
  for(uint k=lane;k<256;k+=32)s+=(as_type<float>(uint(pr[k])<<16)*hproj[r*256+k])*as_type<float>(uint(su[k])<<16);
  s=simd_sum(s);
  if(lane==0){float v=cvv[r*16+sg]+s;sc[sg]=v;scores[r*16+sg]=v;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(t==0){uint best=0;for(uint c=1;c<16;c++)if(sc[c]>sc[best])best=c;pred=ci[r*16+best];path[r]=int(pred);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
)metal";

namespace {
constexpr int kDfMatrices=58,kDfVectors=38,kDfLayers=6,kDfMaxBlock=8,kDfRing=2048,kDfKeep=2047,kDfMaxIngest=512,kDfVocab=248320,kDfChunks=122;
constexpr int kDfTapLayers[8]={1,6,11,16,22,27,32,37};
enum DfMat {DFc,DHProj,DPredCB,DSuccCB,DLayerBase};
enum DfLayerMat {LQ,LK,LV,LO,LGate,LUp,LDown,LAttnKp,LMlpKp};
enum DfLayerVec {VInLn,VPostLn,VQNorm,VKNorm,VAttnBase,VMlpBase};
uint64_t df_matrix_elements(int i){
    if(i==DFc)return uint64_t(2048)*16384;if(i==DHProj)return uint64_t(256)*2048;
    if(i==DPredCB||i==DSuccCB)return uint64_t(kDfVocab)*256;
    switch((i-DLayerBase)%9){case LQ:return uint64_t(4096)*2048;case LK:case LV:return uint64_t(1024)*2048;case LO:return uint64_t(2048)*4096;
        case LGate:case LUp:return uint64_t(6144)*2048;case LDown:return uint64_t(2048)*6144;default:return uint64_t(512)*2048;}
}
uint32_t df_vector_elements(int i){
    if(i<2)return 2048;switch((i-2)%6){case VInLn:case VPostLn:return 2048;case VQNorm:case VKNorm:return 128;default:return 2*2*2048;}
}
struct DfGemmP {uint32_t n,k,m,xs,ys;};
struct DfRmsP {uint32_t dim,xs,ys;float eps;};
struct DfNrP {uint32_t heads,srcStride,dstStride,firstPos,ringMod;float eps;};
struct DfRcP {uint32_t srcStride,firstPos,ringMod;};
struct DfConvP {uint32_t rows,dynWhich,baseWhich,accumulate;};
struct DfAttnP {uint32_t nctx,ctxBegin,nblk,qbase,rows,nparts;float scale;};
struct DfTkP {uint32_t vocab,chunks;};
struct DfSelP {uint32_t rows,anchor;};
enum DfPipe {PGemm,PRms,PNormRope,PRingCopy,PConv,PSilu,PCopy,PTap,PAttnPart,PAttnCombine,PTopk1,PTopk2,PSelector,PCount};
enum DfDebug {XCtxRaw,XCtxNorm,XAttnDyn,XAttnPrep,XAttnOut,XHAttn,XMlpDyn,XMlpPrep,XMlpAct,XHOut,XFinal,XHProj,XCandIds,XCandVals,XScores,XKinds};
}
static int dflash_tap_slot(int layer){for(int i=0;i<8;i++)if(kDfTapLayers[i]==layer)return i;return -1;}

struct GDflash {
    GModel *graph=nullptr;uint64_t graphId=0;bool sealed=false,debug=false;
    id<MTLBuffer> w[kDfMatrices],v[kDfVectors],rope,ringK[kDfLayers],ringV[kDfLayers];
    id<MTLBuffer> ids,h,norm,dyn,cx,q,kb,vb,attn,o,gate,up,pmax,psum,pacc,logits,hn,hproj,s1v,s1i,candV,candI,path,scores;
    id<MTLBuffer> ctxRaw,ctxNorm,kraw,vraw;
    std::vector<id<MTLBuffer>> dbg;
    id<MTLComputePipelineState> ps[PCount];
    std::shared_ptr<QResidencyLease> lease;
    uint32_t ctxBegin=0,ctxEnd=0,positions=0;uint64_t ingests=0,proposals=0,ingestedRows=0;
    uint64_t submissions=0,residentSubmissions=0,unregisteredSubmissions=0;
};
static id<MTLBuffer> df_floats(GModel *g,uint64_t floats){auto b=[g->compute.device newBufferWithLength:NSUInteger(floats*4) options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);return b;}
static bool df_live(const GDflash *d){return d&&d->graph&&d->graph->instanceId==d->graphId;}
static uint64_t df_allocated(const GDflash *d){
    uint64_t total=0;
    for(auto b:d->w)if(b)total+=b.length;for(auto b:d->v)if(b)total+=b.length;
    for(int i=0;i<kDfLayers;i++){total+=d->ringK[i].length+d->ringV[i].length;}
    id<MTLBuffer> all[]={d->rope,d->ids,d->h,d->norm,d->dyn,d->cx,d->q,d->kb,d->vb,d->attn,d->o,d->gate,d->up,d->pmax,d->psum,d->pacc,d->logits,d->hn,d->hproj,
        d->s1v,d->s1i,d->candV,d->candI,d->path,d->scores,d->ctxRaw,d->ctxNorm,d->kraw,d->vraw};
    for(auto b:all)if(b)total+=b.length;for(auto b:d->dbg)if(b)total+=b.length;
    return total;
}
static void df_submission(GDflash *d){
    d->submissions++;
    if(d->lease&&d->graph->compute.residency&&d->lease->buffers.size()==size_t(kDfMatrices+kDfVectors)&&d->graph->compute.residency->allocated()>0)d->residentSubmissions++;
    else d->unregisteredSubmissions++;
}
static void df_bind(id<MTLComputeCommandEncoder> e,id<MTLBuffer> b,int i,NSUInteger off=0){[e setBuffer:b offset:off atIndex:i];}
static void df_dispatch(id<MTLComputeCommandEncoder> e,GDflash *d,int pipe,MTLSize groups,MTLSize threads){
    [e setComputePipelineState:d->ps[pipe]];[e dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}
static void df_gemm(id<MTLComputeCommandEncoder> e,GDflash *d,int matrix,id<MTLBuffer> x,NSUInteger xOffset,uint32_t xs,id<MTLBuffer> y,uint32_t ys,uint32_t n,uint32_t k,uint32_t m){
    DfGemmP p{n,k,m,xs,ys};df_bind(e,d->w[matrix],0);df_bind(e,x,1,xOffset);df_bind(e,y,2);[e setBytes:&p length:sizeof(p) atIndex:3];
    df_dispatch(e,d,PGemm,MTLSizeMake((n+3)/4,(m+7)/8,1),MTLSizeMake(128,1,1));
}
static void df_rms(id<MTLComputeCommandEncoder> e,GDflash *d,id<MTLBuffer> x,NSUInteger xOffset,id<MTLBuffer> w,id<MTLBuffer> y,NSUInteger yOffset,uint32_t rows){
    DfRmsP p{2048,2048,2048,1e-6f};df_bind(e,x,0,xOffset);df_bind(e,w,1);df_bind(e,y,2,yOffset);[e setBytes:&p length:sizeof(p) atIndex:3];
    df_dispatch(e,d,PRms,MTLSizeMake(rows,1,1),MTLSizeMake(256,1,1));
}
static void df_copy(id<MTLComputeCommandEncoder> e,GDflash *d,id<MTLBuffer> src,NSUInteger srcOffset,id<MTLBuffer> dst,uint32_t n){
    df_bind(e,src,0,srcOffset);df_bind(e,dst,1);[e setBytes:&n length:4 atIndex:2];
    [e setComputePipelineState:d->ps[PCopy]];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static id<MTLBuffer> df_dbg(GDflash *d,int kind,int layer){return d->debug?d->dbg[size_t(kind)*kDfLayers+size_t(layer)]:nil;}
static void df_checkpoint(id<MTLComputeCommandEncoder> e,GDflash *d,int kind,int layer,id<MTLBuffer> src,NSUInteger srcOffset,uint32_t floats){
    if(auto b=df_dbg(d,kind,layer))df_copy(e,d,src,srcOffset,b,floats);
}
static void df_conv(id<MTLComputeCommandEncoder> e,GDflash *d,id<MTLBuffer> x,id<MTLBuffer> dyn,id<MTLBuffer> base,id<MTLBuffer> out,uint32_t rows,uint32_t which,uint32_t accumulate){
    DfConvP p{rows,which,which,accumulate};df_bind(e,x,0);df_bind(e,dyn,1);df_bind(e,base,2);df_bind(e,out,3);[e setBytes:&p length:sizeof(p) atIndex:4];
    [e setComputePipelineState:d->ps[PConv]];[e dispatchThreads:MTLSizeMake(2048,rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}
static void df_norm_rope(id<MTLComputeCommandEncoder> e,GDflash *d,id<MTLBuffer> src,id<MTLBuffer> w,id<MTLBuffer> dst,uint32_t heads,uint32_t srcStride,uint32_t dstStride,uint32_t rows,uint32_t firstPos,uint32_t ringMod){
    DfNrP p{heads,srcStride,dstStride,firstPos,ringMod,1e-6f};df_bind(e,src,0);df_bind(e,w,1);df_bind(e,d->rope,2);df_bind(e,dst,3);[e setBytes:&p length:sizeof(p) atIndex:4];
    df_dispatch(e,d,PNormRope,MTLSizeMake(heads,rows,1),MTLSizeMake(128,1,1));
}

extern "C" {
void di_dflash_destroy(void *dflash){
    auto *d=static_cast<GDflash*>(dflash);
    if(d&&d->graph&&d->graph->instanceId==d->graphId){d->graph->tapsEnabled=false;d->graph->tapBuffer=nil;d->graph->tapPipeline=nil;d->graph->tapRows=0;d->graph->tapValidRows=0;}
    delete d;
}
// Requires a prepared full-attention target graph (fused-fp32 decode attention, FP32 KV). Compiles its own library; nothing is
// allocated unless this is called. debug!=0 additionally allocates per-stage checkpoint buffers for the CPU oracle.
void *di_dflash_create(void *graph,int debug,char *error,int capacity){
    @autoreleasepool {
        if(!graph){error_text(error,capacity,@"Invalid graph");return nullptr;}
        auto *g=static_cast<GModel*>(graph);
        if(!g->prefill||!g->decodeFusedFp32||g->kvBf16||g->capacity<64){error_text(error,capacity,@"DFlash2 requires a prepared graph with fused-fp32 decode attention and FP32 KV");return nullptr;}
        if(g->prefill->maxTokens>kDfMaxIngest){error_text(error,capacity,@"DFlash2 requires --metal-scratch-tokens <= 512");return nullptr;}
        if(g->tapsEnabled){error_text(error,capacity,@"DFlash2 already attached to this graph");return nullptr;}
        auto d=std::make_unique<GDflash>();d->graph=g;d->graphId=g->instanceId;d->debug=debug!=0;d->positions=std::max<uint32_t>(g->capacity,32768+4096)+64;
        for(int i=0;i<kDfMatrices;i++){d->w[i]=[g->compute.device newBufferWithLength:NSUInteger(df_matrix_elements(i)*2) options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);if(!d->w[i]){error_text(error,capacity,@"DFlash2 weight allocation failed");return nullptr;}}
        for(int i=0;i<kDfVectors;i++){d->v[i]=df_floats(g,df_vector_elements(i));if(!d->v[i]){error_text(error,capacity,@"DFlash2 norm allocation failed");return nullptr;}}
        const uint64_t B=kDfMaxBlock;
        for(int l=0;l<kDfLayers;l++){d->ringK[l]=df_floats(g,uint64_t(kDfRing)*1024);d->ringV[l]=df_floats(g,uint64_t(kDfRing)*1024);
            if(!d->ringK[l]||!d->ringV[l]){error_text(error,capacity,@"DFlash2 context allocation failed");return nullptr;}
            memset(d->ringK[l].contents,0,d->ringK[l].length);memset(d->ringV[l].contents,0,d->ringV[l].length);}
        d->rope=df_floats(g,uint64_t(d->positions)*64*2);
        {auto *t=static_cast<float*>(d->rope.contents);
         for(uint32_t pos=0;pos<d->positions;pos++)for(int i=0;i<64;i++){double a=double(pos)*std::pow(1.0e7,-double(i)/64.0);t[(uint64_t(pos)*64+i)*2]=float(std::cos(a));t[(uint64_t(pos)*64+i)*2+1]=float(std::sin(a));}}
        const uint64_t parts=(kDfRing+kDfMaxBlock+127)/128;
        d->ids=df_floats(g,B);d->h=df_floats(g,B*2048);d->norm=df_floats(g,B*2048);d->dyn=df_floats(g,B*512);d->cx=df_floats(g,B*2048);
        d->q=df_floats(g,B*4096);d->kb=df_floats(g,B*1024);d->vb=df_floats(g,B*1024);d->attn=df_floats(g,B*4096);d->o=df_floats(g,B*2048);
        d->gate=df_floats(g,B*6144);d->up=df_floats(g,B*6144);d->pmax=df_floats(g,parts*B*32);d->psum=df_floats(g,parts*B*32);d->pacc=df_floats(g,parts*B*32*128);
        d->logits=df_floats(g,uint64_t(kDfMaxBlock-1)*kDfVocab);d->hn=df_floats(g,B*2048);d->hproj=df_floats(g,B*256);
        d->s1v=df_floats(g,uint64_t(7)*kDfChunks*16);d->s1i=df_floats(g,uint64_t(7)*kDfChunks*16);d->candV=df_floats(g,7*16);d->candI=df_floats(g,7*16);
        d->path=df_floats(g,8);d->scores=df_floats(g,7*16);
        d->ctxRaw=df_floats(g,uint64_t(kDfMaxIngest)*2048);d->ctxNorm=df_floats(g,uint64_t(kDfMaxIngest)*2048);
        d->kraw=df_floats(g,uint64_t(kDfMaxIngest)*1024);d->vraw=df_floats(g,uint64_t(kDfMaxIngest)*1024);
        id<MTLBuffer> required[]={d->ids,d->h,d->norm,d->dyn,d->cx,d->q,d->kb,d->vb,d->attn,d->o,d->gate,d->up,d->pmax,d->psum,d->pacc,d->logits,d->hn,d->hproj,d->s1v,d->s1i,d->candV,d->candI,d->path,d->scores,d->ctxRaw,d->ctxNorm,d->kraw,d->vraw,d->rope};
        for(auto b:required)if(!b){error_text(error,capacity,@"DFlash2 scratch allocation failed");return nullptr;}
        if(d->debug){
            const uint32_t sizes[XKinds]={kDfMaxIngest*2048,kDfMaxIngest*2048,8*512,8*2048,8*4096,8*2048,8*512,8*2048,8*6144,8*2048,8*2048,8*256,7*16,7*16,7*16};
            for(int k=0;k<XKinds;k++)for(int l=0;l<kDfLayers;l++){d->dbg.push_back(df_floats(g,sizes[k]));if(!d->dbg.back()){error_text(error,capacity,@"DFlash2 debug allocation failed");return nullptr;}}
        }
        {auto *t=static_cast<float*>(d->ids.contents);for(int i=0;i<8;i++)t[i]=0;}
        NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        auto lib=[g->compute.device newLibraryWithSource:dflash_source options:options error:&failure];
        if(!lib){error_text(error,capacity,failure.localizedDescription);return nullptr;}
        NSString *names[PCount]={@"df_gemm",@"df_rms",@"df_norm_rope",@"df_ring_copy",@"df_dynconv",@"df_silu_mul",@"df_copy",@"df_tap",@"df_attn_part",@"df_attn_combine",@"df_topk1",@"df_topk2",@"df_selector"};
        for(int i=0;i<PCount;i++){
            d->ps[i]=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&failure];
            if(!d->ps[i]||d->ps[i].threadExecutionWidth!=32){error_text(error,capacity,failure.localizedDescription?:@"Unsupported DFlash2 pipeline");return nullptr;}
        }
        if(d->ps[PGemm].maxTotalThreadsPerThreadgroup<128||d->ps[PAttnPart].maxTotalThreadsPerThreadgroup<128||d->ps[PSelector].maxTotalThreadsPerThreadgroup<512||d->ps[PTopk1].maxTotalThreadsPerThreadgroup<256){
            error_text(error,capacity,@"DFlash2 pipelines need larger threadgroups than this device supports");return nullptr;}
        g->tapBuffer=df_floats(g,uint64_t(g->prefill->maxTokens)*16384);
        if(!g->tapBuffer){error_text(error,capacity,@"DFlash2 tap buffer allocation failed");return nullptr;}
        g->tapRows=g->prefill->maxTokens;g->tapPipeline=d->ps[PTap];g->tapsEnabled=true;
        if(g->compute.residency){
            std::vector<id<MTLBuffer>> buffers;for(auto b:d->w)buffers.push_back(b);for(auto b:d->v)buffers.push_back(b);
            if(!g->compute.residency->add(buffers)){g->tapsEnabled=false;error_text(error,capacity,@"DFlash2 weights exceed safe Metal residency budget (no fallback)");return nullptr;}
            d->lease=std::make_shared<QResidencyLease>(g->compute.residency,std::move(buffers));
        }
        return d.release();
    }
}
// kind 0 = BF16 matrix (index order: fc,hidden_projection,pred_cb,succ_cb, then per layer q,k,v,o,gate,up,down,attn_kp,mlp_kp),
// kind 1 = FP32 vector (hidden_norm,norm, then per layer input_ln,post_ln,q_norm,k_norm,attn_base,mlp_base).
int di_dflash_tensor(void *dflash,int kind,int index,void **pointer,uint64_t *bytes){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||d->sealed||!pointer||!bytes)return -1;
    if(kind==0&&index>=0&&index<kDfMatrices){*pointer=d->w[index].contents;*bytes=d->w[index].length;return 0;}
    if(kind==1&&index>=0&&index<kDfVectors){*pointer=d->v[index].contents;*bytes=d->v[index].length;return 0;}
    return -1;
}
int di_dflash_seal(void *dflash){auto *d=static_cast<GDflash*>(dflash);if(!df_live(d))return -1;d->sealed=true;return 0;}
// out: [0]=ctxBegin [1]=ctxEnd [2]=allocated bytes [3]=ingests [4]=proposals [5]=ingested rows [6]=sealed [7]=rope positions
int di_dflash_state(void *dflash,uint64_t *out,int count){
    auto *d=static_cast<GDflash*>(dflash);if(!d||!out||count!=8)return -1;
    const uint64_t v[8]={d->ctxBegin,d->ctxEnd,df_allocated(d),d->ingests,d->proposals,d->ingestedRows,uint64_t(d->sealed),d->positions};memcpy(out,v,sizeof(v));return 0;
}
// out: [0]=DFlash2 command-buffer submissions [1]=with all 96 weight buffers residency-registered [2]=unregistered
int di_dflash_evidence(void *dflash,uint64_t *out,int count){
    auto *d=static_cast<GDflash*>(dflash);if(!d||!out||count!=3)return -1;
    const uint64_t v[3]={d->submissions,d->residentSubmissions,d->unregisteredSubmissions};memcpy(out,v,sizeof(v));return 0;
}
int di_dflash_reset(void *dflash){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d))return -1;d->ctxBegin=d->ctxEnd=0;return 0;
}
// Fills ring slots outside the valid window with NaN so any stale-context read poisons the proposal.
int di_dflash_poison_ring(void *dflash){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d))return -1;
    std::vector<bool> valid(kDfRing,false);for(uint32_t a=d->ctxBegin;a<d->ctxEnd;a++)valid[a%kDfRing]=true;
    for(int l=0;l<kDfLayers;l++){auto *k=static_cast<float*>(d->ringK[l].contents);auto *v=static_cast<float*>(d->ringV[l].contents);
        for(int s=0;s<kDfRing;s++)if(!valid[s])for(int i=0;i<1024;i++){k[s*1024+i]=NAN;v[s*1024+i]=NAN;}}
    return 0;
}
// Exports the valid window of one layer in logical order (rows = ctxEnd-ctxBegin; keys carry k_norm+RoPE).
int di_dflash_export_context(void *dflash,int layer,float *keys,float *values){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||layer<0||layer>=kDfLayers||!keys||!values)return -1;
    auto *k=static_cast<const float*>(d->ringK[layer].contents);auto *v=static_cast<const float*>(d->ringV[layer].contents);
    for(uint32_t a=d->ctxBegin;a<d->ctxEnd;a++){memcpy(keys+uint64_t(a-d->ctxBegin)*1024,k+uint64_t(a%kDfRing)*1024,4096);memcpy(values+uint64_t(a-d->ctxBegin)*1024,v+uint64_t(a%kDfRing)*1024,4096);}
    return 0;
}
// Exports taps of the most recent target prefill/verify call: [rows][8*2048].
int di_graph_taps_export(void *graph,int rows,float *out){
    if(!graph||!out||rows<1)return -1;auto *g=static_cast<GModel*>(graph);if(!g->tapsEnabled||uint32_t(rows)>g->tapValidRows)return -1;
    memcpy(out,g->tapBuffer.contents,size_t(rows)*16384*4);return 0;
}
// Debug checkpoint export (debug-created devices only). Sizes in floats are those of the last call that wrote it.
int di_dflash_debug(void *dflash,int kind,int layer,void *out,uint64_t bytes){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||!d->debug||kind<0||kind>=XKinds||layer<0||layer>=kDfLayers||!out)return -1;
    auto b=df_dbg(d,kind,layer);if(bytes>b.length)return -1;memcpy(out,b.contents,bytes);return 0;
}
// Ingests rows of the target tap buffer [tapRow, tapRow+rows) as committed context at absolute positions firstPos... Positions must
// be contiguous with the current window (or start it when empty). The whole ingest is one command buffer.
int di_dflash_ingest(void *dflash,int tapRow,int rows,int firstPos,double *ms,char *error,int capacity){
    @autoreleasepool {
        auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||!d->sealed){error_text(error,capacity,@"DFlash2 not sealed");return -1;}
        auto *g=d->graph;
        if(rows<1||rows>kDfMaxIngest||tapRow<0||uint32_t(tapRow+rows)>g->tapValidRows||firstPos<0||uint32_t(firstPos)!=g->tapBasePosition+uint32_t(tapRow)){error_text(error,capacity,@"Invalid or stale DFlash2 tap ingest range/position");return -1;}
        if(d->ctxEnd==d->ctxBegin)d->ctxBegin=d->ctxEnd=uint32_t(firstPos);
        else if(uint32_t(firstPos)!=d->ctxEnd){error_text(error,capacity,@"DFlash2 ingest must continue at the committed context end");return -1;}
        if(uint32_t(firstPos+rows)+kDfMaxBlock>d->positions){error_text(error,capacity,@"DFlash2 ingest exceeds the RoPE position table");return -1;}
        auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if(!command||!e){error_text(error,capacity,@"Cannot create DFlash2 ingest encoder");return -1;}
        const uint32_t R=uint32_t(rows);
        df_gemm(e,d,DFc,g->tapBuffer,NSUInteger(tapRow)*16384*4,16384,d->ctxRaw,2048,2048,16384,R);
        df_checkpoint(e,d,XCtxRaw,0,d->ctxRaw,0,R*2048);
        df_rms(e,d,d->ctxRaw,0,d->v[0],d->ctxNorm,0,R);
        df_checkpoint(e,d,XCtxNorm,0,d->ctxNorm,0,R*2048);
        for(int l=0;l<kDfLayers;l++){
            int mb=DLayerBase+l*9,vb=2+l*6;
            df_gemm(e,d,mb+LK,d->ctxNorm,0,2048,d->kraw,1024,1024,2048,R);
            df_gemm(e,d,mb+LV,d->ctxNorm,0,2048,d->vraw,1024,1024,2048,R);
            df_norm_rope(e,d,d->kraw,d->v[vb+VKNorm],d->ringK[l],8,1024,1024,R,uint32_t(firstPos),kDfRing);
            DfRcP rc{1024,uint32_t(firstPos),kDfRing};df_bind(e,d->vraw,0);df_bind(e,d->ringV[l],1);[e setBytes:&rc length:sizeof(rc) atIndex:2];
            [e setComputePipelineState:d->ps[PRingCopy]];[e dispatchThreads:MTLSizeMake(1024,R,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }
        [e endEncoding];df_submission(d);
        double interval=0;if(!q_complete(command,&interval,error,capacity))return -1;
        d->ctxEnd=uint32_t(firstPos+rows);if(d->ctxEnd-d->ctxBegin>kDfKeep)d->ctxBegin=d->ctxEnd-kDfKeep;
        d->ingests++;d->ingestedRows+=R;if(ms)*ms=interval;return 0;
    }
}
// Runs the full drafter for one block. tokens[0]=anchor, tokens[1..B-1]=mask id. The context window must be non-empty.
// Outputs proposals[B-1] (trained-selector path). candidates (optional) = [B-1][16] ids; scores (optional) = [B-1][16].
int di_dflash_propose(void *dflash,const int32_t *tokens,int block,int32_t *proposals,int32_t *candidates,float *scores,double *ms,char *error,int capacity){
    @autoreleasepool {
        auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||!d->sealed){error_text(error,capacity,@"DFlash2 not sealed");return -1;}
        auto *g=d->graph;
        if(!tokens||!proposals||block<2||block>kDfMaxBlock){error_text(error,capacity,@"Invalid DFlash2 block");return -1;}
        if(d->ctxEnd<=d->ctxBegin){error_text(error,capacity,@"DFlash2 context is empty");return -1;}
        if(d->ctxEnd+uint32_t(block)>d->positions){error_text(error,capacity,@"DFlash2 block exceeds the RoPE position table");return -1;}
        const uint32_t B=uint32_t(block),R=B-1,nctx=d->ctxEnd-d->ctxBegin,total=nctx+B,nparts=(total+127)/128;
        auto *ids=static_cast<uint32_t*>(d->ids.contents);for(uint32_t i=0;i<kDfMaxBlock;i++)ids[i]=i<B?uint32_t(tokens[i]):0;
        for(uint32_t i=0;i<B;i++)if(tokens[i]<0||tokens[i]>=kDfVocab){error_text(error,capacity,@"DFlash2 token outside vocabulary");return -1;}
        auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if(!command||!e){error_text(error,capacity,@"Cannot create DFlash2 encoder");return -1;}
        df_bind(e,g->embedding.packed,0);df_bind(e,g->embedding.scales,1);df_bind(e,g->embedding.biases,2);df_bind(e,d->ids,3);df_bind(e,d->h,4);[e setBytes:&g->embedding.bits length:4 atIndex:5];
        [e setComputePipelineState:g->prefill->ps[PEmbed]];[e dispatchThreadgroups:MTLSizeMake(16,B,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        DfAttnP ap{nctx,d->ctxBegin,B,d->ctxEnd,B,nparts,1.0f/std::sqrt(128.0f)};
        for(int l=0;l<kDfLayers;l++){
            int mb=DLayerBase+l*9,vb=2+l*6;
            df_rms(e,d,d->h,0,d->v[vb+VInLn],d->norm,0,B);
            df_gemm(e,d,mb+LAttnKp,d->norm,0,2048,d->dyn,512,512,2048,B);
            df_checkpoint(e,d,XAttnDyn,l,d->dyn,0,B*512);
            df_conv(e,d,d->norm,d->dyn,d->v[vb+VAttnBase],d->cx,B,0,0);
            df_checkpoint(e,d,XAttnPrep,l,d->cx,0,B*2048);
            df_gemm(e,d,mb+LQ,d->cx,0,2048,d->q,4096,4096,2048,B);
            df_gemm(e,d,mb+LK,d->cx,0,2048,d->kb,1024,1024,2048,B);
            df_gemm(e,d,mb+LV,d->cx,0,2048,d->vb,1024,1024,2048,B);
            df_norm_rope(e,d,d->q,d->v[vb+VQNorm],d->q,32,4096,4096,B,d->ctxEnd,0);
            df_norm_rope(e,d,d->kb,d->v[vb+VKNorm],d->kb,8,1024,1024,B,d->ctxEnd,0);
            df_bind(e,d->q,0);df_bind(e,d->ringK[l],1);df_bind(e,d->ringV[l],2);df_bind(e,d->kb,3);df_bind(e,d->vb,4);df_bind(e,d->pmax,5);df_bind(e,d->psum,6);df_bind(e,d->pacc,7);[e setBytes:&ap length:sizeof(ap) atIndex:8];
            df_dispatch(e,d,PAttnPart,MTLSizeMake(nparts,8,1),MTLSizeMake(128,1,1));
            df_bind(e,d->pmax,0);df_bind(e,d->psum,1);df_bind(e,d->pacc,2);df_bind(e,d->attn,3);[e setBytes:&ap length:sizeof(ap) atIndex:4];
            df_dispatch(e,d,PAttnCombine,MTLSizeMake(32,B,1),MTLSizeMake(128,1,1));
            df_checkpoint(e,d,XAttnOut,l,d->attn,0,B*4096);
            df_gemm(e,d,mb+LO,d->attn,0,4096,d->o,2048,2048,4096,B);
            df_conv(e,d,d->o,d->dyn,d->v[vb+VAttnBase],d->h,B,1,1);
            df_checkpoint(e,d,XHAttn,l,d->h,0,B*2048);
            df_rms(e,d,d->h,0,d->v[vb+VPostLn],d->norm,0,B);
            df_gemm(e,d,mb+LMlpKp,d->norm,0,2048,d->dyn,512,512,2048,B);
            df_checkpoint(e,d,XMlpDyn,l,d->dyn,0,B*512);
            df_conv(e,d,d->norm,d->dyn,d->v[vb+VMlpBase],d->cx,B,0,0);
            df_checkpoint(e,d,XMlpPrep,l,d->cx,0,B*2048);
            df_gemm(e,d,mb+LGate,d->cx,0,2048,d->gate,6144,6144,2048,B);
            df_gemm(e,d,mb+LUp,d->cx,0,2048,d->up,6144,6144,2048,B);
            {uint32_t n=B*6144;df_bind(e,d->gate,0);df_bind(e,d->up,1);[e setBytes:&n length:4 atIndex:2];[e setComputePipelineState:d->ps[PSilu]];[e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
            df_checkpoint(e,d,XMlpAct,l,d->gate,0,B*6144);
            df_gemm(e,d,mb+LDown,d->gate,0,6144,d->o,2048,2048,6144,B);
            df_conv(e,d,d->o,d->dyn,d->v[vb+VMlpBase],d->h,B,1,1);
            df_checkpoint(e,d,XHOut,l,d->h,0,B*2048);
        }
        df_rms(e,d,d->h,2048*4,d->v[1],d->hn,0,R);
        df_checkpoint(e,d,XFinal,0,d->hn,0,R*2048);
        PShape shape{};shape.tokens=R;shape.epsilon=g->epsilon;shape.theta=g->theta;shape.capacity=g->capacity;
        p_gemv(g,e,g->classifier,d->hn,d->logits,shape);
        DfGemmP hp{256,2048,R,2048,256};df_bind(e,d->w[DHProj],0);df_bind(e,d->hn,1);df_bind(e,d->hproj,2);[e setBytes:&hp length:sizeof(hp) atIndex:3];
        df_dispatch(e,d,PGemm,MTLSizeMake(64,(R+7)/8,1),MTLSizeMake(128,1,1));
        df_checkpoint(e,d,XHProj,0,d->hproj,0,R*256);
        DfTkP tk{kDfVocab,kDfChunks};
        df_bind(e,d->logits,0);df_bind(e,d->s1v,1);df_bind(e,d->s1i,2);[e setBytes:&tk length:sizeof(tk) atIndex:3];
        df_dispatch(e,d,PTopk1,MTLSizeMake(kDfChunks,R,1),MTLSizeMake(256,1,1));
        df_bind(e,d->s1v,0);df_bind(e,d->s1i,1);df_bind(e,d->candV,2);df_bind(e,d->candI,3);[e setBytes:&tk length:sizeof(tk) atIndex:4];
        df_dispatch(e,d,PTopk2,MTLSizeMake(R,1,1),MTLSizeMake(256,1,1));
        DfSelP sp{R,uint32_t(tokens[0])};
        df_bind(e,d->hproj,0);df_bind(e,d->candI,1);df_bind(e,d->candV,2);df_bind(e,d->w[DPredCB],3);df_bind(e,d->w[DSuccCB],4);df_bind(e,d->path,5);df_bind(e,d->scores,6);[e setBytes:&sp length:sizeof(sp) atIndex:7];
        df_dispatch(e,d,PSelector,MTLSizeMake(1,1,1),MTLSizeMake(512,1,1));
        df_checkpoint(e,d,XCandIds,0,d->candI,0,R*16);df_checkpoint(e,d,XCandVals,0,d->candV,0,R*16);df_checkpoint(e,d,XScores,0,d->scores,0,R*16);
        [e endEncoding];df_submission(d);
        double interval=0;if(!q_complete(command,&interval,error,capacity))return -1;
        {   // Active-finiteness gate on the small outputs only (final hidden, projected hidden, top-16 values, selector scores); no logits download.
            const float *hn=static_cast<const float*>(d->hn.contents),*hp2=static_cast<const float*>(d->hproj.contents),*cv=static_cast<const float*>(d->candV.contents),*sc=static_cast<const float*>(d->scores.contents);
            const uint32_t *ci=static_cast<const uint32_t*>(d->candI.contents);const int32_t *pt=static_cast<const int32_t*>(d->path.contents);bool ok=true;
            for(uint32_t i=0;i<R*2048&&ok;i++)ok=std::isfinite(hn[i]);
            for(uint32_t i=0;i<R*256&&ok;i++)ok=std::isfinite(hp2[i]);
            for(uint32_t i=0;i<R*16&&ok;i++)ok=std::isfinite(cv[i])&&std::isfinite(sc[i])&&ci[i]<uint32_t(kDfVocab);
            for(uint32_t i=0;i<R&&ok;i++)ok=pt[i]>=0&&pt[i]<kDfVocab;
            if(!ok){error_text(error,capacity,@"DFlash2 produced non-finite or invalid draft activations");return -1;}
        }
        const int32_t *path=static_cast<const int32_t*>(d->path.contents);for(uint32_t i=0;i<R;i++)proposals[i]=path[i];
        if(candidates)memcpy(candidates,d->candI.contents,size_t(R)*16*4);
        if(scores)memcpy(scores,d->scores.contents,size_t(R)*16*4);
        d->proposals++;if(ms)*ms=interval;return 0;
    }
}
// Copies the last propose() logits rows [B-1][vocab] (diagnostic/oracle use only).
int di_dflash_logits(void *dflash,int rows,float *out){
    auto *d=static_cast<GDflash*>(dflash);if(!df_live(d)||rows<1||rows>kDfMaxBlock-1||!out)return -1;
    memcpy(out,d->logits.contents,size_t(rows)*kDfVocab*4);return 0;
}
}
