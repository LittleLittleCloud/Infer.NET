// Official Qwen3.6 MTP drafter (Issue #76): one full-attention + MoE layer, BF16 matrices,
// FP32 norm scales / activations / accumulation / KV. Shares the target embedding and
// classifier QMatrix buffers; the graph must outlive the MTP object.
//
// Nothing here is created unless di_mtp_create is called, so graphs that never enable MTP
// allocate no MTP weights, KV, scratch or pipelines.
//
// Reuse of target kernels is deliberate: embedding gather, RMSNorm, residual+norm, top-8
// routing, merge+norm and the partitioned decode attention are the same arithmetic as the
// target layer, with MTP buffers bound. Only the BF16 matrix operators are new.

static NSString *mtp_source = @R"metal(
#include <metal_stdlib>
using namespace metal;
struct MShape {uint rows,cols,tokens,split,slots,useIds;};
inline float mtp_bf(ushort v){return as_type<float>(uint(v)<<16);}
inline float4 mtp_bf4(device const ushort *p){ushort4 r=*reinterpret_cast<device const ushort4*>(p);return float4(mtp_bf(r.x),mtp_bf(r.y),mtp_bf(r.z),mtp_bf(r.w));}
// Row-major BF16 [rows, cols] times up to 8 tokens per simdgroup. Columns below `split` read x1, the rest x2.
kernel void mtp_gemv(device const ushort *w [[buffer(0)]],device const float *x1 [[buffer(1)]],device const float *x2 [[buffer(2)]],
 device float *out [[buffer(3)]],constant MShape &s [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.x*4+sg,t0=group.y*8;if(row>=s.rows)return;
 float acc[8]={0,0,0,0,0,0,0,0};device const ushort *wr=w+ulong(row)*s.cols;
 for(uint c=lane*4;c<s.cols;c+=128){
  float4 wv=mtp_bf4(wr+c);
  for(uint i=0;i<8;i++){uint t=t0+i;if(t>=s.tokens)break;
   float4 xv=c<s.split?*reinterpret_cast<device const float4*>(x1+ulong(t)*s.split+c):*reinterpret_cast<device const float4*>(x2+ulong(t)*(s.cols-s.split)+(c-s.split));
   acc[i]+=dot(wv,xv);}
 }
 for(uint i=0;i<8;i++){float v=simd_sum(acc[i]);uint t=t0+i;if(lane==0&&t<s.tokens)out[ulong(t)*s.rows+row]=v;}
}
// Routed (useIds) or shared gate/up with decode-order SwiGLU: (gate*(1/(1+exp(-gate))))*up.
kernel void mtp_gateup(device const ushort *gw [[buffer(0)]],device const ushort *uw [[buffer(1)]],device const float *x [[buffer(2)]],
 device float *out [[buffer(3)]],constant MShape &s [[buffer(4)]],device const uint *ids [[buffer(5)]],
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.x*4+sg,slot=group.y,t=group.z;if(row>=s.rows)return;
 ulong e=s.useIds?ids[t*s.slots+slot]:0;ulong base=(e*s.rows+row)*s.cols;device const float *input=x+ulong(t)*s.cols;
 float a=0,b=0;
 for(uint c=lane*4;c<s.cols;c+=128){float4 xv=*reinterpret_cast<device const float4*>(input+c);a+=dot(mtp_bf4(gw+base+c),xv);b+=dot(mtp_bf4(uw+base+c),xv);}
 a=simd_sum(a);b=simd_sum(b);
 if(lane==0)out[(ulong(t)*s.slots+slot)*s.rows+row]=(a*(1.0f/(1.0f+exp(-a))))*b;
}
kernel void mtp_down(device const ushort *w [[buffer(0)]],device const float *act [[buffer(1)]],device float *out [[buffer(2)]],
 constant MShape &s [[buffer(3)]],device const uint *ids [[buffer(4)]],
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.x*4+sg,slot=group.y,t=group.z;if(row>=s.rows)return;
 ulong e=s.useIds?ids[t*s.slots+slot]:0;ulong base=(e*s.rows+row)*s.cols;device const float *input=act+(ulong(t)*s.slots+slot)*s.cols;
 float a=0;
 for(uint c=lane*4;c<s.cols;c+=128)a+=dot(mtp_bf4(w+base+c),*reinterpret_cast<device const float4*>(input+c));
 a=simd_sum(a);
 if(lane==0)out[(ulong(t)*s.slots+slot)*s.rows+row]=a;
}
kernel void mtp_copy(device const float *src [[buffer(0)]],device float *dst [[buffer(1)]],constant uint &n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)dst[i]=src[i];}
)metal";

namespace {
constexpr int kMtpMatrices=13,kMtpNorms=7,kMtpRows=32;
// Element counts per matrix in the managed Qwen36MtpNames.Matrices order:
// fc,q,k,v,o,router,expertGate,expertUp,expertDown,sharedGate,sharedUp,sharedDown,sharedRoute.
const uint64_t kMtpMatrixElements[kMtpMatrices]={
    uint64_t(2048)*4096,uint64_t(8192)*2048,uint64_t(512)*2048,uint64_t(512)*2048,uint64_t(2048)*4096,uint64_t(256)*2048,
    uint64_t(256)*512*2048,uint64_t(256)*512*2048,uint64_t(256)*2048*512,uint64_t(512)*2048,uint64_t(512)*2048,uint64_t(2048)*512,uint64_t(2048)};
// preEmbedding,preHidden,final,input,post,q,k.
const uint32_t kMtpNormElements[kMtpNorms]={2048,2048,2048,2048,2048,256,256};
enum MtpMatrix {MFc,MQ,MK,MV,MO,MRouter,MEGate,MEUp,MEDown,MSGate,MSUp,MSDown,MSRoute};
enum MtpNorm {NPreEmb,NPreHid,NFinal,NInput,NPost,NQ,NK};
struct MtpShape {uint32_t rows,cols,tokens,split,slots,useIds;};
}
struct GMtp {
    GModel *graph=nullptr;uint64_t graphId=0;uint32_t capacity=0,validRows=0,lastCount=0;
    bool sealed=false,debug=false;uint64_t runs=0;
    id<MTLBuffer> w[kMtpMatrices],n[kMtpNorms],keys,values;
    id<MTLBuffer> tokens,emb,hid,embN,hidN,x,nrm,qraw,k,v,q,attn,proj,router,sroute,ids,prob,tiles,sAct,eAct,sOut,eOut,y;
    id<MTLBuffer> logits,argValues,argIds,next,xFc,xAttn,xIn;
    uint64_t submissions=0,residentSubmissions=0,unregisteredSubmissions=0,weightBuffers=0;
    id<MTLComputePipelineState> gemv,gateup,down,copy;
    std::shared_ptr<QResidencyLease> lease;
};
static id<MTLBuffer> mtp_floats(GModel *g,uint64_t floats){auto b=[g->compute.device newBufferWithLength:NSUInteger(floats*4) options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);return b;}
static uint64_t mtp_allocated(const GMtp *m){
    uint64_t total=0;
    id<MTLBuffer> all[]={m->keys,m->values,m->tokens,m->emb,m->hid,m->embN,m->hidN,m->x,m->nrm,m->qraw,m->k,m->v,m->q,m->attn,m->proj,m->router,m->sroute,m->ids,m->prob,m->tiles,
        m->sAct,m->eAct,m->sOut,m->eOut,m->y,m->logits,m->argValues,m->argIds,m->next,m->xFc,m->xAttn,m->xIn};
    for(id<MTLBuffer> b:all)if(b)total+=b.length;
    for(id<MTLBuffer> b:m->w)if(b)total+=b.length;
    for(id<MTLBuffer> b:m->n)if(b)total+=b.length;
    return total;
}
static bool mtp_live(const GMtp *m){return m&&m->graph&&m->graph->instanceId==m->graphId;}
// Evidence recorded immediately before every MTP command-buffer commit: all 7 norm and 12 matrix buffers
// must be members of the live residency lease. KV/scratch are shared-storage and, like the target's, are
// not residency-set members; their footprint is covered by the allocator trace, not by this counter.
static void mtp_submission(GMtp *m){
    m->submissions++;
    if(m->lease&&m->graph->compute.residency&&m->lease->buffers.size()==size_t(kMtpMatrices+kMtpNorms)&&m->graph->compute.residency->allocated()>0)m->residentSubmissions++;
    else m->unregisteredSubmissions++;
}

extern "C" {
void di_mtp_destroy(void *mtp){delete static_cast<GMtp*>(mtp);}
// Capacity is the target graph capacity at creation; weights are filled through di_mtp_tensor then sealed.
void *di_mtp_create(void *graph,int debug,char *error,int capacity){
    @autoreleasepool {
        if(!graph){error_text(error,capacity,@"Invalid graph");return nullptr;}
        auto *g=static_cast<GModel*>(graph);
        if(!g->prefill||!g->decodeFusedFp32||g->kvBf16||g->capacity<kMtpRows){error_text(error,capacity,@"MTP requires a prepared graph with fused-fp32 decode attention and FP32 KV");return nullptr;}
        auto m=std::make_unique<GMtp>();m->graph=g;m->graphId=g->instanceId;m->capacity=g->capacity;m->debug=debug!=0;
        for(int i=0;i<kMtpMatrices;i++){m->w[i]=[g->compute.device newBufferWithLength:NSUInteger(kMtpMatrixElements[i]*2) options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);}
        for(int i=0;i<kMtpNorms;i++)m->n[i]=mtp_floats(g,kMtpNormElements[i]);
        const uint64_t T=kMtpRows,C=m->capacity;
        m->keys=mtp_floats(g,C*512);m->values=mtp_floats(g,C*512);m->tokens=mtp_floats(g,T);
        m->emb=mtp_floats(g,T*2048);m->hid=mtp_floats(g,T*2048);m->embN=mtp_floats(g,T*2048);m->hidN=mtp_floats(g,T*2048);
        m->x=mtp_floats(g,T*2048);m->nrm=mtp_floats(g,T*2048);m->qraw=mtp_floats(g,T*8192);m->k=mtp_floats(g,T*512);m->v=mtp_floats(g,T*512);
        m->q=mtp_floats(g,T*4096);m->attn=mtp_floats(g,T*4096);m->proj=mtp_floats(g,T*2048);m->router=mtp_floats(g,T*256);m->sroute=mtp_floats(g,T);
        m->ids=mtp_floats(g,T*8);m->prob=mtp_floats(g,T*8);m->tiles=mtp_floats(g,4);m->sAct=mtp_floats(g,T*512);m->eAct=mtp_floats(g,T*8*512);
        m->sOut=mtp_floats(g,T*2048);m->eOut=mtp_floats(g,T*8*2048);m->y=mtp_floats(g,T*2048);
        m->logits=mtp_floats(g,248320);m->argValues=mtp_floats(g,970);m->argIds=mtp_floats(g,970);m->next=mtp_floats(g,1);
        if(m->debug){m->xFc=mtp_floats(g,T*2048);m->xAttn=mtp_floats(g,T*2048);m->xIn=mtp_floats(g,T*2048);}
        if(!m->keys||!m->values||!m->eOut||!m->logits||!m->y||(m->debug&&(!m->xFc||!m->xAttn||!m->xIn))){error_text(error,capacity,@"MTP buffer allocation failed");return nullptr;}
        for(id<MTLBuffer> b:m->w)if(!b){error_text(error,capacity,@"MTP weight allocation failed");return nullptr;}
        for(id<MTLBuffer> b:m->n)if(!b){error_text(error,capacity,@"MTP norm allocation failed");return nullptr;}
        memset(m->keys.contents,0,m->keys.length);memset(m->values.contents,0,m->values.length);
        NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        auto lib=[g->compute.device newLibraryWithSource:mtp_source options:options error:&failure];
        if(!lib){error_text(error,capacity,failure.localizedDescription);return nullptr;}
        NSString *names[4]={@"mtp_gemv",@"mtp_gateup",@"mtp_down",@"mtp_copy"};id<MTLComputePipelineState> built[4];
        for(int i=0;i<4;i++){built[i]=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&failure];
            if(!built[i]||built[i].threadExecutionWidth!=32||(i<3&&built[i].maxTotalThreadsPerThreadgroup<128)){error_text(error,capacity,failure.localizedDescription?:@"Unsupported MTP pipeline");return nullptr;}}
        m->gemv=built[0];m->gateup=built[1];m->down=built[2];m->copy=built[3];
        if(g->compute.residency){
            std::vector<id<MTLBuffer>> buffers;for(id<MTLBuffer> b:m->w)buffers.push_back(b);for(id<MTLBuffer> b:m->n)buffers.push_back(b);
            if(!g->compute.residency->add(buffers)){error_text(error,capacity,@"MTP weights exceed safe Metal residency budget (no fallback)");return nullptr;}
            m->lease=std::make_shared<QResidencyLease>(g->compute.residency,std::move(buffers));
        }
        return m.release();
    }
}
// kind 0 = BF16 matrix, 1 = FP32 norm scales. Returns shared-memory pointer and exact byte size.
int di_mtp_tensor(void *mtp,int kind,int index,void **pointer,uint64_t *bytes){
    auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||m->sealed||!pointer||!bytes)return -1;
    if(kind==0&&index>=0&&index<kMtpMatrices){*pointer=m->w[index].contents;*bytes=m->w[index].length;return 0;}
    if(kind==1&&index>=0&&index<kMtpNorms){*pointer=m->n[index].contents;*bytes=m->n[index].length;return 0;}
    return -1;
}
int di_mtp_seal(void *mtp){auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m))return -1;m->sealed=true;return 0;}
// out: [0]=valid rows [1]=capacity [2]=allocated bytes [3]=runs [4]=sealed
int di_mtp_state(void *mtp,uint64_t *out,int count){
    auto *m=static_cast<GMtp*>(mtp);if(!m||!out||count!=5)return -1;
    const uint64_t v[5]={m->validRows,m->capacity,mtp_allocated(m),m->runs,uint64_t(m->sealed)};memcpy(out,v,sizeof(v));return 0;
}
// out: [0]=MTP command-buffer submissions [1]=submissions with every weight buffer residency-registered [2]=unregistered
int di_mtp_evidence(void *mtp,uint64_t *out,int count){
    auto *m=static_cast<GMtp*>(mtp);if(!m||!out||count!=3)return -1;
    const uint64_t v[3]={m->submissions,m->residentSubmissions,m->unregisteredSubmissions};memcpy(out,v,sizeof(v));return 0;
}
// Allocator trace (DEEPINFER_METAL_ALLOC_TRACE=1 only). out: [0]=enabled [1]=peak bytes at sampled points
// [2]=last [3]=samples [4]=tracked allocations [5]=new-peak transitions [6]=submissions [7]=submissions above
// recommendedMaxWorkingSetSize [8]=peak at a submission [9]=recommendedMaxWorkingSetSize.
// Sampled, not continuous; not comparable with di_graph_stats (single point-in-time read).
int di_alloc_trace(uint64_t *out,int count){
    if(!out||count!=10)return -1;QAllocTrace &t=q_trace();
    const uint64_t v[10]={t.on?1u:0u,t.peak,t.current,t.samples,t.allocations,t.growthTransitions,t.submissions,t.overBudgetSubmissions,t.submissionPeak,t.budget};memcpy(out,v,sizeof(v));return 0;
}
int di_mtp_reset(void *mtp){auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m))return -1;m->validRows=0;return 0;}
// Lower the target-conditioned cursor; rows above it are never read again until rewritten.
int di_mtp_truncate(void *mtp,int rows){auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||rows<0||uint32_t(rows)>m->validRows)return -1;m->validRows=uint32_t(rows);return 0;}
// Grow KV to the target's new capacity, preserving valid rows.
int di_mtp_reserve(void *mtp,int tokens,char *error,int capacity){
    @autoreleasepool {
        auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||tokens<1){error_text(error,capacity,@"Invalid MTP reserve");return -1;}
        if(uint32_t(tokens)<=m->capacity)return 0;
        auto *g=m->graph;if(uint32_t(tokens)>g->capacity){error_text(error,capacity,@"MTP capacity cannot exceed the target graph capacity");return -1;}
        uint32_t target=g->capacity;
        for(int pass=0;pass<2;pass++){
            id<MTLBuffer> old=pass?m->values:m->keys;id<MTLBuffer> fresh=mtp_floats(g,uint64_t(target)*512);
            if(!fresh){error_text(error,capacity,@"MTP KV growth failed");return -1;}
            memset(fresh.contents,0,fresh.length);memcpy(fresh.contents,old.contents,size_t(m->validRows)*512*4);
            if(pass)m->values=fresh;else m->keys=fresh;
        }
        m->capacity=target;return 0;
    }
}
// Verification hook: NaN-fill KV rows [from, from+rows) (invisible suffixes must never be read).
int di_mtp_poison(void *mtp,int from,int rows){
    auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||from<0||rows<1||uint64_t(from)+rows>m->capacity)return -1;
    auto *k=static_cast<uint32_t*>(m->keys.contents),*v=static_cast<uint32_t*>(m->values.contents);
    for(uint64_t i=uint64_t(from)*512;i<uint64_t(from+rows)*512;i++){k[i]=0x7fc00000u;v[i]=0x7fc00000u;}return 0;
}
int di_mtp_export_kv(void *mtp,int rows,float *keys,float *values){
    auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||rows<0||uint32_t(rows)>m->capacity||(!keys&&!values))return -1;
    if(keys)memcpy(keys,m->keys.contents,size_t(rows)*512*4);if(values)memcpy(values,m->values.contents,size_t(rows)*512*4);return 0;
}
// Debug copies of the last run's scratch tensors (indices 2 and 10 require di_mtp_create(debug=1)).
// 0 embN 1 hidN 2 fc 3 inputNorm 4 qraw 5 q 6 k 7 v 8 attention 9 oProj 10 afterAttention 11 router 12 ids 13 prob 14 sharedAct 15 expertAct 16 sharedOut 17 expertOut 18 y 19 sharedRoute
int di_mtp_debug(void *mtp,int index,void *out,uint64_t bytes){
    auto *m=static_cast<GMtp*>(mtp);if(!mtp_live(m)||!out)return -1;
    id<MTLBuffer> table[20]={m->embN,m->hidN,m->xFc,m->xIn,m->qraw,m->q,m->k,m->v,m->attn,m->proj,m->xAttn,m->router,m->ids,m->prob,m->sAct,m->eAct,m->sOut,m->eOut,m->y,m->sroute};
    if(index<0||index>=20||!table[index]||bytes>table[index].length)return -1;memcpy(out,table[index].contents,bytes);return 0;
}
// One MTP evaluation over `count` rows. Row r pairs tokens[r] with a target hidden row at position base+r.
// base must equal the target-conditioned cursor. commit=1 advances the cursor; commit=0 (provisional draft
// rows) writes KV above the cursor and leaves it unchanged.
// hiddenFromPrevious=1 feeds the previous run's last mtp.norm output as the hidden input (count must be 1).
// The classifier (shared target head) runs on the last row when classify!=0.
int di_mtp_run(void *mtp,const int32_t *tokens,const float *hidden,int count,int base,int commit,int hiddenFromPrevious,int classify,
 int32_t *nextToken,float *outputRow,float *logitsOut,int *dispatches,double *ms,char *error,int capacity){
    @autoreleasepool {
        auto *m=static_cast<GMtp*>(mtp);
        if(!mtp_live(m)||!m->sealed||!tokens||count<1||count>kMtpRows||base<0||(!hidden&&!hiddenFromPrevious)||(hiddenFromPrevious&&(count!=1||hidden))){error_text(error,capacity,@"Invalid MTP run arguments");return -1;}
        auto *g=m->graph;auto *p=g->prefill;
        if(uint32_t(base)!=m->validRows){error_text(error,capacity,[NSString stringWithFormat:@"MTP base %d must equal the target-conditioned cursor %u",base,m->validRows]);return -1;}
        if(uint64_t(base)+count>m->capacity||uint64_t(base)+count>g->capacity){error_text(error,capacity,@"MTP position exceeds capacity");return -1;}
        if(!g->decodeFusedFp32||g->kvBf16||!p){error_text(error,capacity,@"MTP requires fused-fp32 decode attention and FP32 KV");return -1;}
        for(int i=0;i<count;i++)if(tokens[i]<0||tokens[i]>=248320){error_text(error,capacity,@"Invalid MTP token");return -1;}
        if(hiddenFromPrevious&&m->runs==0){error_text(error,capacity,@"No previous MTP output");return -1;}
        memcpy(m->tokens.contents,tokens,size_t(count)*4);
        if(hiddenFromPrevious)memmove(m->hid.contents,static_cast<float*>(m->y.contents)+size_t(m->lastCount-1)*2048,2048*4);
        else{
            for(int i=0;i<count*2048;i++)if(!std::isfinite(hidden[i])){error_text(error,capacity,@"Non-finite target hidden input");return -1;}
            memcpy(m->hid.contents,hidden,size_t(count)*2048*4);
        }
        const int dispatchStart=g->dispatches;
        PShape shape{0,0,(uint32_t)count,(uint32_t)base,0,0,g->epsilon,g->theta,g->capacity};
        auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        if(!command||!e){error_text(error,capacity,@"Cannot create MTP command");return -1;}
        auto norm=[&](id<MTLBuffer> in,id<MTLBuffer> scale,id<MTLBuffer> out){
            g_bind(e,in,0);g_bind(e,scale,1);g_bind(e,out,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];p_run(g,e,PNorm,MTLSizeMake(count,1,1),256);g_barrier(e);};
        auto gemv=[&](int matrix,id<MTLBuffer> x1,id<MTLBuffer> x2,uint32_t rows,uint32_t cols,uint32_t split,id<MTLBuffer> out){
            MtpShape s{rows,cols,(uint32_t)count,split,1,0};
            [e setComputePipelineState:m->gemv];g_bind(e,m->w[matrix],0);g_bind(e,x1,1);g_bind(e,x2,2);g_bind(e,out,3);[e setBytes:&s length:sizeof(s) atIndex:4];
            [e dispatchThreadgroups:MTLSizeMake((rows+3)/4,(count+7)/8,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g_barrier(e);};
        auto gateup=[&](int gate,int up,id<MTLBuffer> out,uint32_t slots,bool routed){
            MtpShape s{512,2048,(uint32_t)count,2048,slots,routed?1u:0u};
            [e setComputePipelineState:m->gateup];g_bind(e,m->w[gate],0);g_bind(e,m->w[up],1);g_bind(e,m->nrm,2);g_bind(e,out,3);[e setBytes:&s length:sizeof(s) atIndex:4];g_bind(e,m->ids,5);
            [e dispatchThreadgroups:MTLSizeMake(128,slots,count) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g_barrier(e);};
        auto down=[&](int matrix,id<MTLBuffer> act,id<MTLBuffer> out,uint32_t slots,bool routed){
            MtpShape s{2048,512,(uint32_t)count,512,slots,routed?1u:0u};
            [e setComputePipelineState:m->down];g_bind(e,m->w[matrix],0);g_bind(e,act,1);g_bind(e,out,2);[e setBytes:&s length:sizeof(s) atIndex:3];g_bind(e,m->ids,4);
            [e dispatchThreadgroups:MTLSizeMake(512,slots,count) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g_barrier(e);};
        auto copy=[&](id<MTLBuffer> src,id<MTLBuffer> dst){uint32_t n=count*2048;[e setComputePipelineState:m->copy];g_bind(e,src,0);g_bind(e,dst,1);[e setBytes:&n length:4 atIndex:2];
            [e dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g_barrier(e);};
        // Normalize the shared-embedding row and the target hidden row separately, then fc over [e,h].
        g_bind(e,g->embedding.packed,0);g_bind(e,g->embedding.scales,1);g_bind(e,g->embedding.biases,2);g_bind(e,m->tokens,3);g_bind(e,m->emb,4);[e setBytes:&g->embedding.bits length:4 atIndex:5];
        p_run(g,e,PEmbed,MTLSizeMake(16,count,1),128);g_barrier(e);
        norm(m->emb,m->n[NPreEmb],m->embN);norm(m->hid,m->n[NPreHid],m->hidN);
        gemv(MFc,m->embN,m->hidN,2048,4096,2048,m->x);if(m->debug)copy(m->x,m->xFc);
        norm(m->x,m->n[NInput],m->nrm);if(m->debug)copy(m->nrm,m->xIn);
        gemv(MQ,m->nrm,m->nrm,8192,2048,2048,m->qraw);gemv(MK,m->nrm,m->nrm,512,2048,2048,m->k);gemv(MV,m->nrm,m->nrm,512,2048,2048,m->v);
        g_bind(e,m->qraw,0);g_bind(e,m->k,1);g_bind(e,m->v,2);g_bind(e,m->n[NQ],3);g_bind(e,m->n[NK],4);g_bind(e,m->q,5);g_bind(e,m->keys,6);g_bind(e,m->values,7);
        [e setBytes:&shape length:sizeof(shape) atIndex:8];p_run(g,e,PHead,MTLSizeMake(18,count,1),256);g_barrier(e);
        {const uint32_t saved=g->position;g->position=uint32_t(base);
         for(int row=0;row<count;row++)p_decode_attention(g,e,m->q,m->keys,m->values,m->qraw,m->attn,uint32_t(row));
         g->position=saved;}
        gemv(MO,m->attn,m->attn,2048,4096,4096,m->proj);
        g_bind(e,m->x,0);g_bind(e,m->proj,1);g_bind(e,m->n[NPost],2);g_bind(e,m->nrm,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];p_run(g,e,PAdd,MTLSizeMake(count,1,1),256);g_barrier(e);
        if(m->debug)copy(m->x,m->xAttn);
        gemv(MRouter,m->nrm,m->nrm,256,2048,2048,m->router);gemv(MSRoute,m->nrm,m->nrm,1,2048,2048,m->sroute);
        g_bind(e,m->router,0);g_bind(e,m->ids,1);g_bind(e,m->prob,2);g_bind(e,m->tiles,3);p_run(g,e,PTop,MTLSizeMake(count,1,1),32);g_barrier(e);
        gateup(MEGate,MEUp,m->eAct,8,true);gateup(MSGate,MSUp,m->sAct,1,false);
        down(MEDown,m->eAct,m->eOut,8,true);down(MSDown,m->sAct,m->sOut,1,false);
        g_bind(e,m->x,0);g_bind(e,m->eOut,1);g_bind(e,m->sOut,2);g_bind(e,m->sroute,3);g_bind(e,m->prob,4);g_bind(e,m->n[NFinal],5);g_bind(e,m->y,6);g_bind(e,m->tiles,7);
        [e setBytes:&shape length:sizeof(shape) atIndex:8];p_run(g,e,PMerge,MTLSizeMake(count,1,1),256);g_barrier(e);
        [e endEncoding];mtp_submission(m);double layerMs=0;if(!q_complete(command,&layerMs,error,capacity))return -1;
        const float *rows=static_cast<const float*>(m->y.contents);
        for(int i=0;i<count*2048;i++)if(!std::isfinite(rows[i])){error_text(error,capacity,@"Non-finite MTP output");return -1;}
        m->lastCount=uint32_t(count);m->runs++;
        double classifyMs=0;
        if(classify){
            command=[g->compute.queue commandBuffer];e=[command computeCommandEncoder];
            if(!command||!e){error_text(error,capacity,@"Cannot create MTP classifier command");return -1;}
            QMatrix &c=g->classifier;struct {uint32_t experts,rows,columns,bits,group,count,batched;} qs{1,c.rows,c.columns,c.bits,64,1,0};
            [e setComputePipelineState:c.bits==8?(g->compute.rowTile==8?g->compute.wide8:g->compute.tiled8):(g->compute.rowTile==8?g->compute.wide5:g->compute.tiled5)];
            g_bind(e,c.packed,0);g_bind(e,c.scales,1);g_bind(e,c.biases,2);g_bind(e,m->y,3,NSUInteger(count-1)*2048*4);g_bind(e,m->logits,4);[e setBytes:&qs length:sizeof(qs) atIndex:5];g_bind(e,g->zero,6);
            [e dispatchThreadgroups:MTLSizeMake((c.rows+g->compute.rowTile-1)/g->compute.rowTile,1,1) threadsPerThreadgroup:MTLSizeMake(g->compute.rowTile*32,1,1)];g_barrier(e);
            g_bind(e,m->logits,0);g_bind(e,m->argValues,1);g_bind(e,m->argIds,2);g_run(g,e,GArgBlocks,970,32);g_barrier(e);
            g_bind(e,m->argValues,0);g_bind(e,m->argIds,1);g_bind(e,m->next,2);g_run(g,e,GArgFinal,1,32);
            [e endEncoding];mtp_submission(m);if(!q_complete(command,&classifyMs,error,capacity))return -1;
            int32_t id=static_cast<const int32_t*>(m->next.contents)[0];
            if(id<0||id>=248320){error_text(error,capacity,@"MTP classifier produced no valid argmax (non-finite logits)");return -1;}
            if(nextToken)*nextToken=id;
            if(logitsOut)memcpy(logitsOut,m->logits.contents,248320*4);
        }else if(nextToken)*nextToken=-1;
        if(outputRow)memcpy(outputRow,rows+size_t(count-1)*2048,2048*4);
        if(commit)m->validRows=uint32_t(base+count);
        if(dispatches)*dispatches=g->dispatches-dispatchStart;
        if(ms)*ms=layerMs+classifyMs;
        return 0;
    }
}
// Exports the post-final-norm hidden rows of the most recent prefill chunk (valid until the next prefill/verify call).
int di_graph_prefill_hidden(void *graph,int rows,float *out){
    if(!graph||!out||rows<1||rows>2048)return -1;auto *g=static_cast<GModel*>(graph);if(!g->prefill)return -1;
    memcpy(out,g->prefill->norm.contents,size_t(rows)*2048*4);return 0;
}
}
