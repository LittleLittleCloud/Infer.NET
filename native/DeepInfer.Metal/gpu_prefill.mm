// Original bounded layer-major prefill. No external matrix/inference library.
// Dense and grouped expert projections reuse packed weights across token tiles.
#include "prefill_moe.mm"
#include "prefill_delta.mm"
#include "prefill_bf16.mm"
#include "prefill_bf16_expanded.mm"
#include "prefill_attention_bf16.mm"
#include "prefill_expert_packed.mm"
#include <string>
static NSString *prefill_source = @R"metal(
struct PShape {uint rows,columns,tokens,position,layer,trace;float epsilon,theta;uint capacity,bits;};
template<uint Bits>
inline float4 prefill_dot(device const uint *w,device const ushort *scales,device const ushort *biases,
 device const float *x,uint columns,uint token,uint tokens,uint lane) {
    float4 sum=0;
    for(uint g=0;g<columns/64;g+=4) {
        uint local=lane*8,bit=local*Bits,word=bit/32,shift=bit%32;
        uint low=w[word],high=w[word+1];low=quant_align_low(low,high,shift);high>>=shift;
        float scale=as_type<float>(uint(scales[g+lane/8])<<16),bias=as_type<float>(uint(biases[g+lane/8])<<16);
        #pragma clang loop unroll(full)
        for(uint j=0;j<8;j++) {
            uint code=quant_octet<Bits>(low,high,j);
            float value=float(code)*scale+bias;
            uint c=g*64+local+j;
            float4 input=0;for(uint i=0;i<4;i++)if(token+i<tokens)input[i]=x[(token+i)*columns+c];
            sum+=value*input;
        }
        w+=Bits*8;
    }
    return float4(simd_sum(sum.x),simd_sum(sum.y),simd_sum(sum.z),simd_sum(sum.w));
}
#define PREFILL_GEMV(NAME,BITS) \
kernel void NAME(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]], \
 device const ushort *biases [[buffer(2)]],device const float *x [[buffer(3)]],device float *out [[buffer(4)]], \
 constant PShape &s [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]], \
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
 uint row=group.x*4+sg,token=group.y*4;if(row>=s.rows)return;ulong p=ulong(row)*(s.columns/64); \
 float4 result=prefill_dot<BITS>(w+p*(BITS*2),scales+p,biases+p,x,s.columns,token,s.tokens,lane); \
 if(lane==0)for(uint i=0;i<4;i++)if(token+i<s.tokens)out[(token+i)*s.rows+row]=result[i]; }
PREFILL_GEMV(prefill_gemv5,5)
PREFILL_GEMV(prefill_gemv8,8)
kernel void prefill_shared(device const uint *gw [[buffer(0)]],device const ushort *gs [[buffer(1)]],device const ushort *gb [[buffer(2)]],
 device const uint *uw [[buffer(3)]],device const ushort *us [[buffer(4)]],device const ushort *ub [[buffer(5)]],
 device const float *x [[buffer(6)]],device float *out [[buffer(7)]],constant PShape &s [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x*4+sg,token=group.y*4;if(row>=s.rows)return;ulong p=ulong(row)*(s.columns/64);
    float4 gate=s.bits==8?prefill_dot<8>(gw+p*16,gs+p,gb+p,x,s.columns,token,s.tokens,lane):prefill_dot<5>(gw+p*10,gs+p,gb+p,x,s.columns,token,s.tokens,lane);
    float4 up=s.bits==8?prefill_dot<8>(uw+p*16,us+p,ub+p,x,s.columns,token,s.tokens,lane):prefill_dot<5>(uw+p*10,us+p,ub+p,x,s.columns,token,s.tokens,lane);
    if(lane==0)for(uint i=0;i<4;i++)if(token+i<s.tokens)out[(token+i)*s.rows+row]=(gate[i]/(1.0f+exp(-gate[i])))*up[i];
}
#define PREFILL_EXPERTS(NAME,BITS) \
kernel void NAME(device const uint *gw [[buffer(0)]],device const ushort *gs [[buffer(1)]],device const ushort *gb [[buffer(2)]], \
 device const uint *uw [[buffer(3)]],device const ushort *us [[buffer(4)]],device const ushort *ub [[buffer(5)]], \
 device const float *x [[buffer(6)]],device float *out [[buffer(7)]],constant PShape &s [[buffer(8)]],device const uint *ids [[buffer(9)]], \
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
    uint row=group.x*4+sg,slot=group.y,token=group.z;if(row>=s.rows)return; \
    ulong p=(ulong(ids[token*8+slot])*s.rows+row)*(s.columns/64);device const float *input=x+token*s.columns; \
    gw+=p*(BITS*2);uw+=p*(BITS*2);gs+=p;gb+=p;us+=p;ub+=p;float gate=0,up=0; \
    _Pragma("clang loop unroll_count(2)") \
    for(uint g=0;g<s.columns/64;g+=4){uint local=lane*8,bit=local*BITS,word=bit/32,shift=bit%32; \
        uint gl=gw[word],gh=gw[word+1],ul=uw[word],uh=uw[word+1];gl=quant_align_low(gl,gh,shift);gh>>=shift;ul=quant_align_low(ul,uh,shift);uh>>=shift;uint at=g+lane/8; \
        float sgain=as_type<float>(uint(gs[at])<<16),gbias=as_type<float>(uint(gb[at])<<16); \
        float ugain=as_type<float>(uint(us[at])<<16),ubias=as_type<float>(uint(ub[at])<<16); \
        _Pragma("clang loop unroll(full)") \
        for(uint j=0;j<8;j++){uint gc=quant_octet<BITS>(gl,gh,j),uc=quant_octet<BITS>(ul,uh,j);float value=input[g*64+local+j]; \
            gate+=(float(gc&((1u<<BITS)-1))*sgain+gbias)*value;up+=(float(uc&((1u<<BITS)-1))*ugain+ubias)*value;} \
        gw+=BITS*8;uw+=BITS*8; \
    } \
    gate=simd_sum(gate);up=simd_sum(up); \
    if(lane==0)out[(token*8+slot)*s.rows+row]=(gate/(1.0f+exp(-gate)))*up; \
}
PREFILL_EXPERTS(prefill_experts5,5)
PREFILL_EXPERTS(prefill_experts8,8)
kernel void prefill_expert_down(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]],
 device const float *x [[buffer(3)]],device float *out [[buffer(4)]],constant PShape &s [[buffer(5)]],device const uint *ids [[buffer(6)]],
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x*4+sg,slot=group.y,token=group.z;if(row>=s.rows)return;
    ulong p=(ulong(ids[token*8+slot])*s.rows+row)*(s.columns/64);
    float result=s.bits==8?quant_dot<8>(w+p*16,scales+p,biases+p,x+(token*8+slot)*s.columns,s.columns/64,lane):quant_dot<5>(w+p*10,scales+p,biases+p,x+(token*8+slot)*s.columns,s.columns/64,lane);
    if(lane==0)out[(token*8+slot)*s.rows+row]=result;
}

#include <metal_simdgroup_matrix>

#define PREFILL_MATRIX(NAME,BITS,TOKENS) \
kernel void NAME(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]], \
 device const float *x [[buffer(3)]],device float *out [[buffer(4)]],constant PShape &s [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]], \
 uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) { \
 threadgroup float weights[PREFILL_MATRIX_ROWS*32],inputs[32*TOKENS],results[PREFILL_MATRIX_ROWS*TOKENS];simdgroup_float8x8 acc(0.0f),acc1(0.0f),acc2(0.0f),acc3(0.0f); \
 for(uint start=0;start<s.columns;start+=32) { \
 for(uint i=tid;i<PREFILL_MATRIX_ROWS*32;i+=PREFILL_MATRIX_ROWS*4){uint row=group.x*PREFILL_MATRIX_ROWS+i/32,col=start+i%32;float value=0; \
 if(row<s.rows){ulong r=row,base=r*(s.columns*BITS/32),p=r*(s.columns/64)+col/64;uint bit=col*BITS; \
 value=float(quant_code<BITS>(w+base,bit/32,bit%32))*as_type<float>(uint(scales[p])<<16)+as_type<float>(uint(biases[p])<<16);}weights[i]=value;} \
 for(uint i=tid;i<32*TOKENS;i+=PREFILL_MATRIX_ROWS*4){uint t=i%TOKENS,token=group.y*TOKENS+t; \
 inputs[i]=token<s.tokens?x[token*s.columns+start+i/TOKENS]:0;} \
 threadgroup_barrier(mem_flags::mem_threadgroup); \
 for(uint k=0;k<4;k++){simdgroup_float8x8 a,b;simdgroup_load(a,weights+sg*8*32+k*8,32);simdgroup_load(b,inputs+k*8*TOKENS,TOKENS);simdgroup_multiply_accumulate(acc,a,b,acc); \
 if(TOKENS>8){simdgroup_load(b,inputs+k*8*TOKENS+8,TOKENS);simdgroup_multiply_accumulate(acc1,a,b,acc1);} \
 if(TOKENS>16){simdgroup_load(b,inputs+k*8*TOKENS+16,TOKENS);simdgroup_multiply_accumulate(acc2,a,b,acc2);}if(TOKENS>24){simdgroup_load(b,inputs+k*8*TOKENS+24,TOKENS);simdgroup_multiply_accumulate(acc3,a,b,acc3);}} \
 threadgroup_barrier(mem_flags::mem_threadgroup); } \
 simdgroup_store(acc,results+sg*8*TOKENS,TOKENS); \
 if(TOKENS>8)simdgroup_store(acc1,results+sg*8*TOKENS+8,TOKENS);if(TOKENS>16)simdgroup_store(acc2,results+sg*8*TOKENS+16,TOKENS);if(TOKENS>24)simdgroup_store(acc3,results+sg*8*TOKENS+24,TOKENS);threadgroup_barrier(mem_flags::mem_threadgroup); \
 for(uint i=tid;i<PREFILL_MATRIX_ROWS*TOKENS;i+=PREFILL_MATRIX_ROWS*4){uint row=group.x*PREFILL_MATRIX_ROWS+i/TOKENS,t=i%TOKENS,token=group.y*TOKENS+t; \
 if(row<s.rows&&token<s.tokens)out[token*s.rows+row]=results[i];} }
PREFILL_MATRIX(prefill_matrix5_8,5,8)
PREFILL_MATRIX(prefill_matrix8_8,8,8)
PREFILL_MATRIX(prefill_matrix5_16,5,16)
PREFILL_MATRIX(prefill_matrix8_16,8,16)
PREFILL_MATRIX(prefill_matrix5_24,5,24)
PREFILL_MATRIX(prefill_matrix8_24,8,24)
PREFILL_MATRIX(prefill_matrix5_32,5,32)
PREFILL_MATRIX(prefill_matrix8_32,8,32)



)metal";
static NSString *prefill_graph_source = @R"metal(
struct PShape {uint rows,columns,tokens,position,layer,trace;float epsilon,theta;uint capacity,bits;};
kernel void prefill_embedding(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]],
 device const uint *ids [[buffer(3)]],device float *out [[buffer(4)]],constant uint &bits [[buffer(5)]],uint2 at [[thread_position_in_grid]]) {
 uint c=at.x,token=ids[at.y],bit=c*bits,word=bit/32,shift=bit%32;ulong base=ulong(token)*(2048*bits/32);
 uint code=w[base+word]>>shift;if(shift+bits>32)code|=w[base+word+1]<<(32-shift);
 uint p=token*32+c/64;out[at.y*2048+c]=float(code&((1u<<bits)-1))*as_type<float>(uint(scales[p])<<16)+as_type<float>(uint(biases[p])<<16);
}
kernel void prefill_norm(device const float *x [[buffer(0)]],device const float *w [[buffer(1)]],device float *out [[buffer(2)]],
 constant PShape &s [[buffer(3)]],uint token [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float partials[8];float total=0;for(uint d=tid;d<2048;d+=256){float v=x[token*2048+d];total+=v*v;}
 float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
 for(uint d=tid;d<2048;d+=256)out[token*2048+d]=x[token*2048+d]*scale*w[d];
}
kernel void prefill_add_norm(device float *x [[buffer(0)]],device const float *add [[buffer(1)]],device const float *w [[buffer(2)]],
 device float *out [[buffer(3)]],constant PShape &s [[buffer(4)]],uint token [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float partials[8];float values[8],total=0;
 for(uint i=0;i<8;i++){uint d=token*2048+tid+i*256;values[i]=x[d]+add[d];total+=values[i]*values[i];}
 float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
 for(uint i=0;i<8;i++){uint c=tid+i*256,d=token*2048+c;x[d]=values[i];out[d]=values[i]*scale*w[c];}
}
kernel void prefill_merge(device float *x [[buffer(0)]],device const float *expert [[buffer(1)]],device const float *shared [[buffer(2)]],
 device const float *route [[buffer(3)]],device const float *prob [[buffer(4)]],device const float *w [[buffer(5)]],
 device float *out [[buffer(6)]],device float *trace [[buffer(7)]],constant PShape &s [[buffer(8)]],uint token [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float partials[8];float values[8],total=0;
 for(uint i=0;i<8;i++){uint c=tid+i*256;float y=shared[token*2048+c]*sigmoid(route[token]);
 for(uint e=0;e<8;e++)y+=expert[(token*8+e)*2048+c]*prob[token*8+e];values[i]=x[token*2048+c]+y;total+=values[i]*values[i];}
 float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
 for(uint i=0;i<8;i++){uint c=tid+i*256,d=token*2048+c;x[d]=values[i];out[d]=values[i]*scale*w[c];
 if(s.trace)trace[(token*40+s.layer)*2048+c]=values[i];}
}
kernel void prefill_top8(device const float *logits [[buffer(0)]],device uint *ids [[buffer(1)]],device float *prob [[buffer(2)]],device atomic_uint *tileCount [[buffer(3)]],
 uint token [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 if(token==0&&lane==0)atomic_store_explicit(tileCount,0u,memory_order_relaxed);
 float candidates[8],chosen[8];for(uint j=0;j<8;j++)candidates[j]=logits[token*256+lane+j*32];
 for(uint n=0;n<8;n++){float best=candidates[0];uint id=lane;
 for(uint j=1;j<8;j++)if(candidates[j]>best){best=candidates[j];id=lane+j*32;}
 float maximum=simd_max(best);uint selected=simd_min(best==maximum?id:0xffffffffu);
 if(lane==0){ids[token*8+n]=selected;chosen[n]=maximum;}if((selected&31)==lane)candidates[selected>>5]=-INFINITY;}
 if(lane==0){float total=0;for(uint i=0;i<8;i++){prob[token*8+i]=exp(chosen[i]-chosen[0]);total+=prob[token*8+i];}
 for(uint i=0;i<8;i++)prob[token*8+i]/=total;}
}
// Convolution reads prior history without modifying it. Recurrent execution
// updates that history only after all token preparations have completed.
kernel void prefill_linear(device const float *raw [[buffer(0)]],device const float *history [[buffer(1)]],
 device const float *conv [[buffer(2)]],device float *out [[buffer(3)]],device float *a [[buffer(4)]],device float *b [[buffer(5)]],
 device const float *alog [[buffer(6)]],device const float *dt [[buffer(7)]],constant PShape &s [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint head=group.x,token=group.y,d=head*128+tid;float value=0;
 for(uint j=0;j<4;j++){int at=int(token)+int(j)-3;float x=at<0?history[(at+3)*8192+d]:raw[at*8192+d];value+=x*conv[d*4+j];}
 value=silu(value);
 if(head<32){threadgroup float partials[4];float sum=simd_sum(value*value);if(lane==0)partials[sg]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);if(tid==0)partials[0]=rsqrt(partials[0]+partials[1]+partials[2]+partials[3]+1e-6f)*(head<16?rsqrt(128.0f):1.0f);
 threadgroup_barrier(mem_flags::mem_threadgroup);value*=partials[0];}
 out[token*8192+d]=value;
 if(head==0&&tid<32){uint at=token*32+tid;float t=a[at]+dt[tid];a[at]=exp(-exp(alog[tid])*(t>20?t:log(1.0f+exp(t))));b[at]=sigmoid(b[at]);}
}
kernel void prefill_delta(device float *state [[buffer(0)]],device const float *conv [[buffer(1)]],device const float *decay [[buffer(2)]],
 device const float *beta [[buffer(3)]],device float *out [[buffer(4)]],device float *history [[buffer(5)]],device const float *raw [[buffer(6)]],
 constant PShape &s [[buffer(7)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group*2+sg,head=row/128,kh=head/2;float values[4];for(uint j=0;j<4;j++)values[j]=state[row*128+lane+32*j];
 for(uint t=0;t<s.tokens;t++){
 float prediction=0;for(uint j=0;j<4;j++){uint d=lane+32*j;values[j]*=decay[t*32+head];prediction+=values[j]*conv[t*8192+2048+kh*128+d];}
 prediction=simd_sum(prediction);float correction=(conv[t*8192+4096+row]-prediction)*beta[t*32+head],result=0;
 for(uint j=0;j<4;j++){uint d=lane+32*j;values[j]+=conv[t*8192+2048+kh*128+d]*correction;result+=values[j]*conv[t*8192+kh*128+d];}
 result=simd_sum(result);if(lane==0)out[t*4096+row]=result;
 }
 for(uint j=0;j<4;j++)state[row*128+lane+32*j]=values[j];
 uint d=group*64+tid;if(d<8192){float next[3];for(uint j=0;j<3;j++){uint at=j+s.tokens;next[j]=at<3?history[at*8192+d]:raw[(at-3)*8192+d];}
 for(uint j=0;j<3;j++)history[j*8192+d]=next[j];}
}
kernel void prefill_delta_norm(device float *x [[buffer(0)]],device const float *z [[buffer(1)]],device const float *w [[buffer(2)]],
 constant PShape &s [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float partials[4];uint at=group.y*4096+group.x*128+tid;float value=x[at],sum=simd_sum(value*value);
 if(lane==0)partials[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0)partials[0]=rsqrt((partials[0]+partials[1]+partials[2]+partials[3])/128.0f+s.epsilon);
 threadgroup_barrier(mem_flags::mem_threadgroup);x[at]=value*partials[0]*w[tid]*silu(z[at]);
}

kernel void prefill_head(device const float *raw [[buffer(0)]],device const float *k [[buffer(1)]],
 device const float *v [[buffer(2)]],device const float *qw [[buffer(3)]],device const float *kw [[buffer(4)]],
 device float *q [[buffer(5)]],device DIKV *keys [[buffer(6)]],device DIKV *values [[buffer(7)]],constant PShape &s [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint head=group.x,token=group.y,position=s.position+token;raw+=token*8192;k+=token*512;v+=token*512;q+=token*4096;
    threadgroup float partials[8];bool query=head<16;uint h=query?head:head-16;
    device const float *input=query?raw+h*512:k+h*256;
    device const float *weight=query?qw:kw;
    float value=input[tid];float scale=rsqrt(reduce_sum(value*value,partials,tid,lane,sg)/256.0f+s.epsilon);
    value=value*scale*weight[tid];
    if(tid<64) {
        uint frequency=tid%32,partner=tid<32?tid+32:tid-32;
        float angle=float(position)/pow(s.theta,float(frequency)/32.0f);
        float other=input[partner]*scale*weight[partner];
        value=value*cos(angle)+(tid<32?-other:other)*sin(angle);
    }
    if(query)q[h*256+tid]=value;
    else {keys[position*512+h*256+tid]=DIKV(value);values[position*512+h*256+tid]=DIKV(v[h*256+tid]);}
}
kernel void prefill_scores(device const float *q [[buffer(0)]],device const DIKV *keys [[buffer(1)]],
 device float *scores [[buffer(2)]],constant PShape &s [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint t=group.x*4+sg,h=group.y,token=group.z;if(t>s.position+token)return;q+=token*4096;scores+=token*16*s.capacity;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=q[h*256+d]*float(keys[t*512+(h/8)*256+d]);
    sum=simd_sum(sum);if(lane==0)scores[h*s.capacity+t]=sum/16.0f;
}
kernel void prefill_attention(device const float *scores [[buffer(0)]],device const DIKV *values [[buffer(1)]],
 device const float *raw [[buffer(2)]],device float *out [[buffer(3)]],constant PShape &s [[buffer(4)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint h=group.x,token=group.y,position=s.position+token;scores+=token*16*s.capacity;raw+=token*8192;out+=token*4096;
    threadgroup float prob[2048];threadgroup float partials[8];threadgroup float maximum;float best=-INFINITY;
    for(uint t=tid;t<=position;t+=256)best=max(best,scores[h*s.capacity+t]);
    best=simd_max(best);if(lane==0)partials[sg]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float m=partials[0];for(uint i=1;i<8;i++)m=max(m,partials[i]);maximum=m;}
    threadgroup_barrier(mem_flags::mem_threadgroup);best=maximum;float sum=0;
    for(uint t=tid;t<=position;t+=256){float p=exp(scores[h*s.capacity+t]-best);prob[t]=p;sum+=p;}
    sum=reduce_sum(sum,partials,tid,lane,sg);
    for(uint t=tid;t<=position;t+=256)prob[t]/=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);float result=0;
    for(uint t=0;t<=position;t++)result+=prob[t]*float(values[t*512+(h/8)*256+tid]);
    out[h*256+tid]=result*sigmoid(raw[h*512+256+tid]);
}

#include <metal_simdgroup_matrix>
// Four query tokens x eight GQA heads share a KV tile. Register accumulators keep
// threadgroup storage bounded to 13 KiB; statistics never leave the chip until
// a complete 1024-key partition is finished. All math and storage are FP32.
kernel void prefill_fused_attention(device const float *q [[buffer(0)]],device const DIKV *keys [[buffer(1)]],
 device const DIKV *values [[buffer(2)]],device float *parts [[buffer(3)]],constant AttentionShape &s [[buffer(4)]],
 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Features=PREFILL_ATTENTION_FEATURES;
    uint part=group.x,kh=group.y,qbase=(group.z/(256/Features))*4,featureBase=(group.z%(256/Features))*Features;
    uint start=part*1024,end=min(start+1024,s.position+s.tokens);
    threadgroup float temp[32*64],scores[32*32],scales[32*8],maxima[32],sums[32],alpha[32];
    simdgroup_float8x8 accumulated[Features/8];
    #pragma clang loop unroll(full)
    for(uint j=0;j<Features/8;j++)accumulated[j]=simdgroup_float8x8(0.0f);
    if(tid<32){maxima[tid]=-INFINITY;sums[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(start<=s.position+min(qbase+4,s.tokens)-1)for(uint block=start;block<end;block+=32) {
        simdgroup_float8x8 dots[4];
        for(uint j=0;j<4;j++)dots[j]=simdgroup_float8x8(0.0f);
        for(uint dim=0;dim<256;dim+=32) {
            for(uint i=tid;i<32*32;i+=128){uint row=i/32,d=dim+i%32,t=qbase+row/8,h=kh*8+row%8;
                temp[i]=t<s.tokens?q[t*4096+h*256+d]:0;
                uint key=block+i%32,kdim=dim+i/32;
                temp[1024+i]=key<end?float(keys[key*512+kh*256+kdim]):0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint inner=0;inner<4;inner++) {
                simdgroup_float8x8 a,b;simdgroup_load(a,temp+sg*8*32+inner*8,32);
                for(uint j=0;j<4;j++){simdgroup_load(b,temp+1024+inner*8*32+j*8,32);simdgroup_multiply_accumulate(dots[j],a,b,dots[j]);}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for(uint j=0;j<4;j++)simdgroup_store(dots[j],scores+sg*8*32+j*8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=sg;row<32;row+=4) {
            uint t=qbase+row/8,key=block+lane;
            float value=t<s.tokens&&key<end&&key<=s.position+t?scores[row*32+lane]/16.0f:-INFINITY;
            float next=max(maxima[row],simd_max(value));
            float scale=maxima[row]==-INFINITY?0:exp(maxima[row]-next);
            float prob=value==-INFINITY?0:exp(value-next);
            float total=simd_sum(prob);
            scores[row*32+lane]=prob;
            if(lane==0){maxima[row]=next;sums[row]=sums[row]*scale+total;alpha[row]=scale;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<32*8;i+=128)scales[i]=alpha[i/8];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Load an identically shaped row-scale fragment. Elementwise scaling
        // therefore needs no assumption about the implementation's lane layout.
        simdgroup_float8x8 rescale;simdgroup_load(rescale,scales+sg*8*8,8);
        #pragma clang loop unroll(full)
        for(uint j=0;j<Features/8;j++)accumulated[j].thread_elements()*=rescale.thread_elements();
        #pragma clang loop unroll(full)
        for(uint feature=0;feature<Features;feature+=64) {
            for(uint i=tid;i<32*64;i+=128){uint key=block+i/64,d=featureBase+feature+i%64;
                temp[i]=key<end?float(values[key*512+kh*256+d]):0;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for(uint inner=0;inner<4;inner++){
                simdgroup_float8x8 a,b;simdgroup_load(a,scores+sg*8*32+inner*8,32);
                #pragma clang loop unroll(full)
                for(uint j=0;j<8;j++){simdgroup_load(b,temp+inner*8*64+j*8,64);
                    uint index=feature/8+j;simdgroup_multiply_accumulate(accumulated[index],a,b,accumulated[index]);}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if(qbase+sg<s.tokens) {
        device float *out=parts+(((qbase+sg)*16+kh*8)*s.partitions+part)*258+2+featureBase;
        #pragma clang loop unroll(full)
        for(uint j=0;j<Features/8;j++)simdgroup_store(accumulated[j],out+j*8,s.partitions*258);
    }
    if(featureBase==0&&tid<32){uint t=qbase+tid/8,h=kh*8+tid%8;
        if(t<s.tokens){uint at=((t*16+h)*s.partitions+part)*258;parts[at]=maxima[tid];parts[at+1]=sums[tid];}}

}
// One decode query: four SIMD groups split key columns for QK, then output
// feature columns for PV. All eight GQA heads are useful (no padded queries).
kernel void decode_fused_attention(device const float *q [[buffer(0)]],device const DIKV *keys [[buffer(1)]],
 device const DIKV *values [[buffer(2)]],device float *parts [[buffer(3)]],constant AttentionShape &s [[buffer(4)]],
 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint part=group.x,kh=group.y,start=part*128,end=min(start+128,s.position+1);
    threadgroup float temp[16*256],scores[8*32],scales[8*8],maxima[8],sums[8],alpha[8];
    simdgroup_float8x8 accumulated[8];
    #pragma clang loop unroll(full)
    for(uint j=0;j<8;j++)accumulated[j]=simdgroup_float8x8(0.0f);
    if(tid<8){maxima[tid]=-INFINITY;sums[tid]=0;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint block=start;block<end;block+=32) {
        simdgroup_float8x8 dots(0.0f);
        for(uint dim=0;dim<256;dim+=32) {
            for(uint i=tid;i<8*32;i+=128)temp[i]=q[(kh*8+i/32)*256+dim+i%32];
            for(uint i=tid;i<32*32;i+=128){uint key=block+i%32;temp[256+i]=key<end?float(keys[key*512+kh*256+dim+i/32]):0;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint inner=0;inner<4;inner++) {
                simdgroup_float8x8 a,b;simdgroup_load(a,temp+inner*8,32);
                simdgroup_load(b,temp+256+inner*8*32+sg*8,32);simdgroup_multiply_accumulate(dots,a,b,dots);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(dots,scores+sg*8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint row=sg;row<8;row+=4) {
            float value=block+lane<end?scores[row*32+lane]/16.0f:-INFINITY;
            float next=max(maxima[row],simd_max(value));
            float scale=maxima[row]==-INFINITY?0:exp(maxima[row]-next);
            float prob=value==-INFINITY?0:exp(value-next),total=simd_sum(prob);
            scores[row*32+lane]=prob;
            if(lane==0){maxima[row]=next;sums[row]=sums[row]*scale+total;alpha[row]=scale;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<64)scales[tid]=alpha[tid/8];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 rescale;simdgroup_load(rescale,scales,8);
        #pragma clang loop unroll(full)
        for(uint j=0;j<8;j++)accumulated[j].thread_elements()*=rescale.thread_elements();
        // Two 16-key value tiles keep total threadgroup memory below 18 KiB.
        for(uint keyHalf=0;keyHalf<32;keyHalf+=16) {
            for(uint i=tid;i<16*256;i+=128){uint key=block+keyHalf+i/256;temp[i]=key<end?float(values[key*512+kh*256+i%256]):0;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint inner=0;inner<16;inner+=8) {
                simdgroup_float8x8 a,b;simdgroup_load(a,scores+keyHalf+inner,32);
                #pragma clang loop unroll(full)
                for(uint j=0;j<8;j++){simdgroup_load(b,temp+inner*256+sg*64+j*8,256);simdgroup_multiply_accumulate(accumulated[j],a,b,accumulated[j]);}
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    device float *out=parts+(kh*8*s.partitions+part)*258+2+sg*64;
    #pragma clang loop unroll(full)
    for(uint j=0;j<8;j++)simdgroup_store(accumulated[j],out+j*8,s.partitions*258);
    if(tid<8){uint at=((kh*8+tid)*s.partitions+part)*258;parts[at]=maxima[tid];parts[at+1]=sums[tid];}
}

)metal";
struct PShape {uint32_t rows,columns,tokens,position,layer,trace;float epsilon,theta;uint32_t capacity=2048,bits=5;};
enum PP {PQ5,PQ8,PShared,PExperts,PExperts8,PDown,PEmbed,PNorm,PAdd,PMerge,PTop,PLinear,PDelta,PDeltaNorm,PHead,PScores,PAttention,PMatrix85,PMatrix88,PMatrix165,PMatrix168,PMatrix245,PMatrix248,PMatrix5,PMatrix8,PFusedAttention,PDecodeAttention,PBuildGroups,PGroupDispatch,PGroupedGate,PGroupedDown,PDeltaPrepare,PDeltaApply,PPCount};
struct GPrefill {
 std::array<id<MTLComputePipelineState>,PPCount> ps;
 id<MTLBuffer> x,norm,out,qkv,z,a,b,q,k,v,attention,conv,router,ids,prob,sharedAct,sharedRoute,sharedOut,expertAct,expertOut,scores,tokens,trace;
 id<MTLBuffer> groupedRoutes,groupedCounts,groupedTiles,groupedTileCount,groupedDispatch,deltaW,deltaU,deltaCausal,deltaFactors;
 bool groupedExperts=false,chunkedDelta=false,deviceDelta=false,bf16Gemm=false,expandedGemm=false,expandedExperts=false,packedExperts=false;
 bool directExperts=false,expertActDirect=false,pipelineExperts=false,adaptiveExperts=false,fusedExpertGate=false;id<MTLComputePipelineState> packedGatePs,packedDownPs,packedGateDirectPs;
 int submissions=1;
 bool attentionBf16=false;uint32_t attentionQueryTokens=4,attentionKeyTokens=32,attentionThreads=128,attentionBatchTokens=128,attentionPartitionTokens=1024;id<MTLComputePipelineState> attentionBf16Ps,attentionWidePs,queryPackPs;id<MTLBuffer> packedQueries;
 std::array<id<MTLComputePipelineState>,4> bf16ps,expandedPs;
 id<MTLBuffer> expandedWeight,expandedUp,expandedInput;
 bool matrix=true,fusedAttention=true,fusedAll=false;uint32_t matrixRows=64,matrixTokens=0,scoreCapacity=2048,attentionFeatures=256,maxTokens=2048;
};
static void g_prefill_destroy(GPrefill *p){delete p;}
static bool p_create(GModel *g,char *error,int capacity,uint32_t scratchTokens=0) {
 auto *p=new GPrefill{};
 p->groupedExperts=!std::getenv("DEEPINFER_METAL_PREFILL_EXPERTS")||strcmp(std::getenv("DEEPINFER_METAL_PREFILL_EXPERTS"),"grouped")==0;
 p->chunkedDelta=!std::getenv("DEEPINFER_METAL_PREFILL_DELTA")||strcmp(std::getenv("DEEPINFER_METAL_PREFILL_DELTA"),"chunked")==0;
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_DELTA_IMPL")){
  if(strcmp(v,"staged")&&strcmp(v,"device")){error_text(error,capacity,@"DELTA_IMPL must be staged or device");delete p;return false;}p->deviceDelta=strcmp(v,"device")==0;
  if(p->deviceDelta&&!p->chunkedDelta){error_text(error,capacity,@"Device DeltaNet requires chunked delta");delete p;return false;}
 }
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION")){p->fusedAttention=strcmp(v,"legacy")!=0;p->fusedAll=strcmp(v,"fused-all")==0;}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION_FEATURES")){if(strcmp(v,"64")==0)p->attentionFeatures=64;else if(strcmp(v,"128")==0)p->attentionFeatures=128;}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_CHUNK"))for(uint32_t n:{32u,64u,128u,256u,512u,1024u,2048u})if(std::string(v)==std::to_string(n))p->maxTokens=n;
 if(scratchTokens)p->maxTokens=scratchTokens;
 if(const char *gemm=std::getenv("DEEPINFER_METAL_PREFILL_GEMM"))if(strcmp(gemm,"fp32")&&strcmp(gemm,"bf16-nax")){error_text(error,capacity,@"DEEPINFER_METAL_PREFILL_GEMM must be fp32 or bf16-nax");delete p;return false;}
 p->bf16Gemm=std::getenv("DEEPINFER_METAL_PREFILL_GEMM")&&strcmp(std::getenv("DEEPINFER_METAL_PREFILL_GEMM"),"bf16-nax")==0;
 if(const char *impl=std::getenv("DEEPINFER_METAL_PREFILL_GEMM_IMPL")){
  if(strcmp(impl,"staged")&&strcmp(impl,"hybrid")&&strcmp(impl,"expanded")){error_text(error,capacity,@"GEMM_IMPL must be staged, hybrid or expanded");delete p;return false;}
  p->expandedGemm=strcmp(impl,"staged")!=0;p->expandedExperts=strcmp(impl,"expanded")==0;
 }
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL"))if(strcmp(v,"staged")&&strcmp(v,"packed")){error_text(error,capacity,@"EXPERT_IMPL must be staged or packed");delete p;return false;}
 p->packedExperts=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL")&&strcmp(std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL"),"packed")==0;
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_FLOW")){
  if(strcmp(v,"staged")&&strcmp(v,"direct")){error_text(error,capacity,@"EXPERT_FLOW must be staged or direct");delete p;return false;}
  p->directExperts=strcmp(v,"direct")==0;
 }
 if(p->directExperts&&!(p->bf16Gemm&&p->packedExperts&&p->expandedGemm&&!p->expandedExperts&&p->groupedExperts)){error_text(error,capacity,@"Direct expert flow requires hybrid packed BF16 experts");delete p;return false;}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE")){
  if(strcmp(v,"serial")&&strcmp(v,"double-buffer")){error_text(error,capacity,@"EXPERT_PIPELINE must be serial or double-buffer");delete p;return false;}
  p->pipelineExperts=strcmp(v,"double-buffer")==0;
 }
 if(p->pipelineExperts&&!(p->bf16Gemm&&p->packedExperts)){error_text(error,capacity,@"Expert pipeline requires packed BF16 experts");delete p;return false;}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_GATE")){
  if(strcmp(v,"split")&&strcmp(v,"fused")){error_text(error,capacity,@"EXPERT_GATE must be split or fused");delete p;return false;}
  p->fusedExpertGate=strcmp(v,"fused")==0;
  if(p->fusedExpertGate&&(!p->bf16Gemm||!p->packedExperts||p->pipelineExperts)){error_text(error,capacity,@"Fused gate requires serial packed BF16 experts");delete p;return false;}
 }
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_EXPERT_TILES")){
  if(strcmp(v,"fixed")&&strcmp(v,"adaptive")){error_text(error,capacity,@"EXPERT_TILES must be fixed or adaptive");delete p;return false;}
  p->adaptiveExperts=strcmp(v,"adaptive")==0;
  if(p->adaptiveExperts&&(!p->bf16Gemm||!p->packedExperts)){error_text(error,capacity,@"Adaptive expert tiles require packed BF16 experts");delete p;return false;}
 }
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION_BATCH")){
  bool valid=false;for(uint32_t n:{128u,256u,512u,1024u,2048u})if(std::string(v)==std::to_string(n)){p->attentionBatchTokens=n;valid=true;break;}
  if(!valid){error_text(error,capacity,@"ATTENTION_BATCH must be 128, 256, 512, 1024 or 2048");delete p;return false;}
 }
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 const char *row=std::getenv("DEEPINFER_METAL_PREFILL_ROWS");p->matrixRows=row&&strcmp(row,"32")==0?32:64;
 const char *tokenTile=std::getenv("DEEPINFER_METAL_PREFILL_TOKEN_TILE");p->matrixTokens=tokenTile&&strcmp(tokenTile,"8")==0?8:tokenTile&&strcmp(tokenTile,"32")==0?32:0;
 NSString *source=[[NSString stringWithFormat:@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS %u\n#define PREFILL_ROUTE_TILE %u\n",p->matrixRows,p->bf16Gemm?64:16] stringByAppendingString:[quant_source stringByAppendingString:prefill_source]];
 auto qlib=[g->compute.device newLibraryWithSource:q_math_source([source stringByAppendingString:prefill_moe_source]) options:options error:&failure];
 if(!qlib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
 NSString *attentionSource=[[NSString stringWithFormat:@"#define PREFILL_ATTENTION_FEATURES %u\n",p->attentionFeatures] stringByAppendingString:[g_kv_source(g) stringByAppendingString:prefill_graph_source]];
 auto glib=[g->compute.device newLibraryWithSource:q_math_source([attentionSource stringByAppendingString:[[NSString stringWithFormat:@"\n#define DELTA_DEVICE %d\n",p->deviceDelta?1:0] stringByAppendingString:prefill_delta_source]]) options:options error:&failure];
 if(!qlib||!glib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
 NSArray<NSString*> *names=@[@"prefill_gemv5",@"prefill_gemv8",@"prefill_shared",@"prefill_experts5",@"prefill_experts8",@"prefill_expert_down",@"prefill_embedding",@"prefill_norm",@"prefill_add_norm",@"prefill_merge",@"prefill_top8",@"prefill_linear",@"prefill_delta",@"prefill_delta_norm",@"prefill_head",@"prefill_scores",@"prefill_attention",@"prefill_matrix5_8",@"prefill_matrix8_8",@"prefill_matrix5_16",@"prefill_matrix8_16",@"prefill_matrix5_24",@"prefill_matrix8_24",@"prefill_matrix5_32",@"prefill_matrix8_32",@"prefill_fused_attention",@"decode_fused_attention",@"prefill_build_groups",@"prefill_group_dispatch",@"prefill_grouped_gate",@"prefill_grouped_down",@"prefill_delta_prepare",@"prefill_delta_apply"];
 for(int i=0;i<PPCount;i++){p->ps[i]=[g->compute.device newComputePipelineStateWithFunction:[(i<6||(i>=PMatrix85&&i<PFusedAttention)||(i>=PBuildGroups&&i<PDeltaPrepare)?qlib:glib) newFunctionWithName:names[i]] error:&failure];
 if(!p->ps[i]||p->ps[i].threadExecutionWidth!=32||p->ps[i].maxTotalThreadsPerThreadgroup<256){error_text(error,capacity,failure.localizedDescription ?: @"Unsupported prefill pipeline");delete p;return false;}}
 if(p->bf16Gemm){
  if(![g->compute.device supportsFamily:MTLGPUFamilyApple10]){error_text(error,capacity,@"BF16 NAX GEMM requires Apple10 (M5) or later");delete p;return false;}
  if(@available(macOS 26.0,*)){options.languageVersion=MTLLanguageVersion4_0;}else{error_text(error,capacity,@"BF16 NAX requires macOS 26 or later");delete p;return false;}
  auto lib=[g->compute.device newLibraryWithSource:q_math_source([source stringByAppendingString:prefill_bf16_source]) options:options error:&failure];
  if(!lib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
  NSArray<NSString*> *bfNames=@[@"prefill_bf16_matrix5",@"prefill_bf16_matrix8",@"prefill_bf16_gate",@"prefill_bf16_down"];
  for(int i=0;i<4;i++){p->bf16ps[i]=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:bfNames[i]] error:&failure];
   if(!p->bf16ps[i]||p->bf16ps[i].maxTotalThreadsPerThreadgroup<128){error_text(error,capacity,failure.localizedDescription ?: @"Unsupported BF16 GEMM pipeline");delete p;return false;}}
 }
 if(p->bf16Gemm&&p->expandedGemm){
  auto lib=[g->compute.device newLibraryWithSource:q_math_source([source stringByAppendingString:prefill_expanded_source]) options:options error:&failure];
  if(!lib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
  p->expandedWeight=g_buffer(g,p->expandedExperts?NSUInteger(256)*512*2048/2:NSUInteger(8192)*2048/2);p->expandedUp=g_buffer(g,p->expandedExperts?NSUInteger(256)*512*2048/2:NSUInteger(512)*2048/2);
  p->expandedInput=g_buffer(g,(p->expandedExperts||p->packedExperts)?NSUInteger((p->maxTokens*8+63)/64+256)*64*2048/2:NSUInteger(std::max(64u,p->maxTokens))*4096/2);
  if(!p->expandedWeight||!p->expandedUp||!p->expandedInput){error_text(error,capacity,@"Cannot allocate bounded BF16 expansion workspace");delete p;return false;}
  NSArray<NSString*> *names=@[@"expanded_weights",@"expanded_pack",@"expanded_matrix",@"expanded_gate"];
  for(int i=0;i<4;i++){p->expandedPs[i]=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&failure];if(!p->expandedPs[i]){error_text(error,capacity,failure.localizedDescription);delete p;return false;}}
 }
 if(p->bf16Gemm&&p->packedExperts){
  auto packedSource=[[NSString stringWithFormat:@"#define PACKED_EXPERT_PIPELINE %d\n#define PACKED_EXPERT_ADAPTIVE %d\n",p->pipelineExperts?1:0,p->adaptiveExperts?1:0] stringByAppendingString:[source stringByAppendingString:prefill_expert_packed_source]];
  auto lib=[g->compute.device newLibraryWithSource:q_math_source(packedSource) options:options error:&failure];if(!lib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
  p->packedGatePs=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:p->fusedExpertGate?@"packed_expert_gate_fused":@"packed_expert_gate"] error:&failure];p->packedDownPs=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"packed_expert_down"] error:&failure];
  p->packedGateDirectPs=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:p->fusedExpertGate?@"packed_expert_gate_fused_direct":@"packed_expert_gate_direct"] error:&failure];
  if(!p->packedGateDirectPs){error_text(error,capacity,failure.localizedDescription ?: @"Unsupported direct expert pipeline");delete p;return false;}
  if(!p->expandedGemm||p->expandedExperts||!p->groupedExperts||!p->packedGatePs||!p->packedDownPs){error_text(error,capacity,failure.localizedDescription ?: @"Packed experts require hybrid GEMM and grouped routing");delete p;return false;}
 }
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION_IMPL")){
  if(strcmp(v,"fp32")&&strcmp(v,"bf16-nax")){error_text(error,capacity,@"ATTENTION_IMPL must be fp32 or bf16-nax");delete p;return false;}p->attentionBf16=strcmp(v,"bf16-nax")==0;}
 if(p->attentionBatchTokens>128&&!p->attentionBf16){error_text(error,capacity,@"Attention batches above 128 require BF16 fused attention");delete p;return false;}
 if(p->attentionBf16&&!p->fusedAttention){error_text(error,capacity,@"BF16 attention requires fused attention mode");delete p;return false;}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION")){
  bool valid=false;for(uint32_t n:{1024u,2048u,4096u,8192u,16384u,32768u})if(std::string(v)==std::to_string(n)){p->attentionPartitionTokens=n;valid=true;break;}
  if(!valid||!p->attentionBf16||p->attentionQueryTokens!=4){error_text(error,capacity,@"ATTENTION_PARTITION requires BF16 attention and a supported size");delete p;return false;}
 }
 if(p->attentionBf16){
  if(!g->kvBf16){error_text(error,capacity,@"BF16 attention requires DEEPINFER_METAL_KV=bf16");delete p;return false;}
  if(@available(macOS 26.0,*)){options.languageVersion=MTLLanguageVersion4_0;}else{error_text(error,capacity,@"BF16 attention requires macOS 26+");delete p;return false;}
  if(![g->compute.device supportsFamily:MTLGPUFamilyApple10]){error_text(error,capacity,@"BF16 attention requires M5 or later");delete p;return false;}
  if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_QUERY_TILE"))if(strcmp(v,"8")==0)p->attentionQueryTokens=8;
  if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_KEY_TILE")){if(strcmp(v,"64")==0)p->attentionKeyTokens=64;else if(strcmp(v,"128")==0)p->attentionKeyTokens=128;}
  if(p->attentionQueryTokens==8&&p->attentionKeyTokens==128){error_text(error,capacity,@"Query tile 8 requires key tile <=64");delete p;return false;}
  if(p->attentionPartitionTokens!=1024&&(p->attentionQueryTokens!=4||p->attentionKeyTokens!=128)){error_text(error,capacity,@"Custom attention partitions require query tile 4 and key tile 128");delete p;return false;}
  auto attentionSource=[[NSString stringWithFormat:@"#define BF16_QUERY_TOKENS %u\n#define BF16_KEY_TOKENS %u\n",p->attentionQueryTokens,p->attentionKeyTokens] stringByAppendingString:[g_kv_source(g) stringByAppendingString:prefill_attention_bf16_source]];
  auto lib=[g->compute.device newLibraryWithSource:q_math_source(attentionSource) options:options error:&failure];
  if(!lib){error_text(error,capacity,failure.localizedDescription);delete p;return false;}
  p->packedQueries=g_buffer(g,NSUInteger(p->attentionBatchTokens)*4096/2);p->queryPackPs=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"prefill_pack_queries"] error:&failure];
  p->attentionBf16Ps=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"prefill_attention_bf16"] error:&failure];
  // More SIMD groups reduce live FP32 matrix fragments per thread on long
  // prefill tiles. Decode/tiny inputs retain the smaller original pipeline.
  if(p->attentionQueryTokens==4&&p->attentionKeyTokens==128){
   auto wideSource=[[NSString stringWithFormat:@"#define BF16_ATTENTION_SG 16\n#define BF16_ATTENTION_PARTITION %u\n",p->attentionPartitionTokens] stringByAppendingString:attentionSource];
   auto wideLib=[g->compute.device newLibraryWithSource:q_math_source(wideSource) options:options error:&failure];
   if(wideLib)p->attentionWidePs=[g->compute.device newComputePipelineStateWithFunction:[wideLib newFunctionWithName:@"prefill_attention_bf16"] error:&failure];
   if(!p->attentionWidePs||p->attentionWidePs.maxTotalThreadsPerThreadgroup<512){error_text(error,capacity,failure.localizedDescription ?: @"Unsupported wide BF16 attention pipeline");delete p;return false;}
  }
  if(!p->packedQueries||!p->queryPackPs||!p->attentionBf16Ps||p->attentionBf16Ps.maxTotalThreadsPerThreadgroup<128){error_text(error,capacity,failure.localizedDescription ?: @"Unsupported BF16 attention pipeline");delete p;return false;}
 }
 p->scoreCapacity=p->fusedAttention?2048:g->capacity;
 NSUInteger partsFloats=NSUInteger(std::min(p->attentionBatchTokens,p->maxTokens))*16*((g->capacity+1023)/1024)*258;
 if(!g->attentionParts||g->attentionParts.length<partsFloats*4)g->attentionParts=g_buffer(g,partsFloats);
 int M=p->maxTokens;auto b=[&](int n){return g_buffer(g,M*n);};
 p->x=b(2048);p->norm=b(2048);p->out=b(2048);p->qkv=b(8192);p->z=b(4096);p->a=b(32);p->b=b(32);p->q=b(4096);p->k=b(512);p->v=b(512);p->attention=b(4096);
 p->conv=b(8192);p->router=b(256);p->ids=b(8);p->prob=b(8);p->sharedAct=b(512);p->sharedRoute=b(1);p->sharedOut=b(2048);p->expertAct=g_buffer(g,p->directExperts?std::max(NSUInteger(M)*4096,NSUInteger((M*8+63)/64+256)*64*512/2):NSUInteger(M)*4096);p->expertOut=b(16384);p->scores=g_buffer(g,NSUInteger(std::min(128,M))*16*p->scoreCapacity);p->tokens=b(1);p->trace=g_buffer(g,1);
 // Gate output and packed gate input must remain disjoint while groups run.
 // At M=2048 the existing FP32 activation buffer already fits packed BF16 tails.
 p->groupedTileCount=g_buffer(g,1);
 if(p->groupedExperts){p->groupedRoutes=g_buffer(g,256*M);p->groupedCounts=g_buffer(g,256);p->groupedTiles=g_buffer(g,M*8*2);p->groupedDispatch=g_buffer(g,8);
  if(!p->groupedRoutes||!p->groupedCounts||!p->groupedTiles||!p->groupedDispatch){error_text(error,capacity,@"Cannot allocate grouped expert scratch");delete p;return false;}}
 if(p->chunkedDelta){p->deltaW=b(4096);p->deltaU=b(4096);p->deltaCausal=b(1024);p->deltaFactors=b(64);
  if(!p->deltaW||!p->deltaU||!p->deltaCausal||!p->deltaFactors){error_text(error,capacity,@"Cannot allocate chunkwise delta scratch");delete p;return false;}}
 if(const char *v=std::getenv("DEEPINFER_METAL_PREFILL_MATRIX"))p->matrix=strcmp(v,"1")==0;
 for(auto buffer:{p->x,p->norm,p->out,p->qkv,p->z,p->a,p->b,p->q,p->k,p->v,p->attention,p->conv,p->router,p->ids,p->prob,p->sharedAct,p->sharedRoute,p->sharedOut,p->expertAct,p->expertOut,p->scores,p->tokens,p->trace})
 if(!buffer||!g->attentionParts||!p->groupedTileCount){error_text(error,capacity,@"Cannot allocate bounded prefill scratch");delete p;return false;}
 g->prefill=p;return true;
}
static void p_run(GModel *g,id<MTLComputeCommandEncoder> e,int pipeline,MTLSize grid,int threads) {
 [e setComputePipelineState:g->prefill->ps[pipeline]];[e dispatchThreadgroups:grid threadsPerThreadgroup:MTLSizeMake(threads,1,1)];g->dispatches++;
}
static void p_build_groups(GModel *g,id<MTLComputeCommandEncoder> e,PShape shape,uint32_t gateRows,uint32_t downRows) {
 auto *p=g->prefill;
 g_bind(e,p->ids,0);g_bind(e,p->groupedRoutes,1);g_bind(e,p->groupedCounts,2);g_bind(e,p->groupedTiles,3);g_bind(e,p->groupedTileCount,4);
 [e setBytes:&shape length:sizeof(shape) atIndex:5];p_run(g,e,PBuildGroups,MTLSizeMake(256,1,1),32);g_barrier(e);
 uint32_t rows[2]{p->packedExperts&&p->bf16Gemm?gateRows*2:gateRows,p->packedExperts&&p->bf16Gemm?downRows*2:downRows};g_bind(e,p->groupedTileCount,0);g_bind(e,p->groupedDispatch,1);[e setBytes:rows length:sizeof(rows) atIndex:2];
 p_run(g,e,PGroupDispatch,MTLSizeMake(1,1,1),32);g_barrier(e);
}
static void p_delta(GModel *g,id<MTLComputeCommandEncoder> e,id<MTLBuffer> state,id<MTLBuffer> history,PShape shape) {
 auto *p=g->prefill;
 if(p->chunkedDelta&&shape.tokens>=64){struct {uint32_t tokens,blocks;} ds{shape.tokens,(shape.tokens+31)/32};
  g_bind(e,p->conv,0);g_bind(e,p->a,1);g_bind(e,p->b,2);g_bind(e,p->deltaW,3);g_bind(e,p->deltaU,4);g_bind(e,p->deltaCausal,5);g_bind(e,p->deltaFactors,6);[e setBytes:&ds length:sizeof(ds) atIndex:7];
  p_run(g,e,PDeltaPrepare,MTLSizeMake(32,ds.blocks,1),128);g_barrier(e);
  g_bind(e,state,0);g_bind(e,p->conv,1);g_bind(e,p->deltaW,2);g_bind(e,p->deltaU,3);g_bind(e,p->deltaCausal,4);g_bind(e,p->deltaFactors,5);g_bind(e,p->attention,6);g_bind(e,history,7);g_bind(e,p->qkv,8);[e setBytes:&ds length:sizeof(ds) atIndex:9];
  p_run(g,e,PDeltaApply,MTLSizeMake(32,4,1),128);g_barrier(e);
 } else {
  g_bind(e,state,0);g_bind(e,p->conv,1);g_bind(e,p->a,2);g_bind(e,p->b,3);g_bind(e,p->attention,4);g_bind(e,history,5);g_bind(e,p->qkv,6);[e setBytes:&shape length:sizeof(shape) atIndex:7];
  p_run(g,e,PDelta,MTLSizeMake(2048,1,1),64);g_barrier(e);
 }
}
static void p_fused_attention(GModel *g,id<MTLComputeCommandEncoder> e,id<MTLBuffer> q,id<MTLBuffer> keys,
 id<MTLBuffer> values,id<MTLBuffer> raw,id<MTLBuffer> out,uint32_t count,uint32_t offset=0) {
 uint32_t partition=g->prefill->attentionBf16&&count>=32?g->prefill->attentionPartitionTokens:1024;
 AttentionShape shape{g->position+offset,count,g->capacity,(g->position+offset+count+partition-1)/partition};
 if(g->prefill->attentionBf16){auto *p=g->prefill;
  [e setComputePipelineState:p->queryPackPs];g_bind(e,q,0,NSUInteger(offset)*4096*4);g_bind(e,p->packedQueries,1);[e setBytes:&count length:sizeof(count) atIndex:2];
  [e dispatchThreadgroups:MTLSizeMake((count*4096+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
  g_bind(e,p->packedQueries,0);
 }else g_bind(e,q,0,NSUInteger(offset)*4096*4);
 g_bind(e,keys,1);g_bind(e,values,2);g_bind(e,g->attentionParts,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];
 if(g->prefill->attentionBf16){auto *p=g->prefill;bool wide=count>=32&&p->attentionWidePs;
  [e setComputePipelineState:wide?p->attentionWidePs:p->attentionBf16Ps];[e dispatchThreadgroups:MTLSizeMake(shape.partitions,2,(count+p->attentionQueryTokens-1)/p->attentionQueryTokens) threadsPerThreadgroup:MTLSizeMake(wide?512:p->attentionThreads,1,1)];g->dispatches++;}
 else p_run(g,e,PFusedAttention,MTLSizeMake(shape.partitions,2,((count+3)/4)*(256/g->prefill->attentionFeatures)),128);g_barrier(e);
 g_bind(e,g->attentionParts,0);g_bind(e,raw,1,NSUInteger(offset)*8192*4);g_bind(e,out,2,NSUInteger(offset)*4096*4);[e setBytes:&shape length:sizeof(shape) atIndex:3];
 [e setComputePipelineState:g->ps[GAttentionCombine]];
 [e dispatchThreadgroups:MTLSizeMake(16,count,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
}
static void p_decode_attention(GModel *g,id<MTLComputeCommandEncoder> e,id<MTLBuffer> q,id<MTLBuffer> keys,
 id<MTLBuffer> values,id<MTLBuffer> raw,id<MTLBuffer> out) {
 AttentionShape shape{g->position,1,g->capacity,(g->position+128)/128};
 g_bind(e,q,0);g_bind(e,keys,1);g_bind(e,values,2);g_bind(e,g->attentionParts,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];
 p_run(g,e,PDecodeAttention,MTLSizeMake(shape.partitions,2,1),128);g_barrier(e);
 g_bind(e,g->attentionParts,0);g_bind(e,raw,1);g_bind(e,out,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];
 [e setComputePipelineState:g->ps[GAttentionCombine]];
 [e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
}
static bool g_try_fused_attention(GModel *g,id<MTLComputeCommandEncoder> e,GLayer &layer){
 if(!g->prefill)return false;
 if(g->decodeFusedFp32){p_decode_attention(g,e,g->q,layer.keys,layer.values,g->qkv,g->attention);return true;}
 if(!g->prefill->attentionBf16)return false;
 if(const char *v=std::getenv("DEEPINFER_METAL_DECODE_ATTENTION"))if(strcmp(v,"legacy")==0)return false;
 p_fused_attention(g,e,g->q,layer.keys,layer.values,g->qkv,g->attention,1);return true;
}
struct EShape {uint32_t rows,columns,tokens,experts,grouped,bits,maxTiles;};
static void p_pack_expert(GModel *g,id<MTLComputeCommandEncoder> e,PShape shape,bool gate,id<MTLBuffer> input){
 auto *p=g->prefill;uint32_t tiles=(shape.tokens*8+63)/64+256;EShape s{shape.rows,shape.columns,shape.tokens,256,gate?1u:2u,shape.bits,tiles};
 [e setComputePipelineState:p->expandedPs[1]];g_bind(e,input,0);g_bind(e,p->expandedInput,1);[e setBytes:&s length:sizeof(s) atIndex:2];g_bind(e,p->groupedRoutes,3);g_bind(e,p->groupedCounts,4);g_bind(e,p->groupedTiles,5);g_bind(e,p->groupedTileCount,6);
 [e dispatchThreadgroups:MTLSizeMake((shape.columns*64+255)/256,tiles,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
}
static void p_expanded(GModel *g,id<MTLComputeCommandEncoder> e,QMatrix &m,QMatrix *up,id<MTLBuffer> x,id<MTLBuffer> out,PShape shape,bool grouped){
 auto *p=g->prefill;uint32_t tiles=grouped?(shape.tokens*8+63)/64+256:(shape.tokens+63)/64;
 EShape s{m.rows,m.columns,shape.tokens,grouped?m.experts:1u,grouped?(up?1u:2u):0u,m.bits,tiles};
 auto expand=[&](QMatrix &matrix,id<MTLBuffer> target){[e setComputePipelineState:p->expandedPs[0]];g_bind(e,matrix.packed,0);g_bind(e,matrix.scales,1);g_bind(e,matrix.biases,2);g_bind(e,target,3);[e setBytes:&s length:sizeof(s) atIndex:4];g_bind(e,grouped?p->groupedCounts:p->groupedTileCount,5);
  [e dispatchThreadgroups:MTLSizeMake((NSUInteger(s.experts)*matrix.rows*matrix.columns/8+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;};
 expand(m,p->expandedWeight);if(up)expand(*up,p->expandedUp);
 [e setComputePipelineState:p->expandedPs[1]];g_bind(e,x,0);g_bind(e,p->expandedInput,1);[e setBytes:&s length:sizeof(s) atIndex:2];g_bind(e,p->groupedRoutes?:p->groupedTileCount,3);g_bind(e,p->groupedCounts?:p->groupedTileCount,4);g_bind(e,p->groupedTiles?:p->groupedTileCount,5);g_bind(e,p->groupedTileCount,6);
 [e dispatchThreadgroups:MTLSizeMake((m.columns*64+255)/256,tiles,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
 [e setComputePipelineState:p->expandedPs[up?3:2]];int at=0;g_bind(e,p->expandedWeight,at++);if(up)g_bind(e,p->expandedUp,at++);g_bind(e,p->expandedInput,at++);g_bind(e,out,at++);[e setBytes:&s length:sizeof(s) atIndex:at++];g_bind(e,p->groupedRoutes?:p->groupedTileCount,at++);g_bind(e,p->groupedCounts?:p->groupedTileCount,at++);g_bind(e,p->groupedTiles?:p->groupedTileCount,at++);
 if(grouped)[e dispatchThreadgroupsWithIndirectBuffer:p->groupedDispatch indirectBufferOffset:up?0:16 threadsPerThreadgroup:MTLSizeMake(128,1,1)];
 else [e dispatchThreadgroups:MTLSizeMake((m.rows+63)/64,tiles,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g->dispatches++;g_barrier(e);
}
static void p_gemv(GModel *g,id<MTLComputeCommandEncoder> e,QMatrix &m,id<MTLBuffer> x,id<MTLBuffer> out,PShape shape,bool experts=false) {
 shape.rows=m.rows;shape.columns=m.columns;shape.bits=m.bits;
 bool direct=experts&&g->prefill->expertActDirect&&x==g->prefill->expertAct&&shape.tokens>=64;
 if(experts&&g->prefill->bf16Gemm&&g->prefill->packedExperts&&shape.tokens>=64&&!direct)p_pack_expert(g,e,shape,false,x);
 if(g->prefill->bf16Gemm&&g->prefill->expandedGemm&&shape.tokens>=64&&m.rows>=32&&(!experts||(g->prefill->groupedExperts&&g->prefill->expandedExperts))){p_expanded(g,e,m,nullptr,x,out,shape,experts);return;}
 g_bind(e,m.packed,0);g_bind(e,m.scales,1);g_bind(e,m.biases,2);g_bind(e,!direct&&experts&&g->prefill->bf16Gemm&&g->prefill->packedExperts&&shape.tokens>=64?g->prefill->expandedInput:x,3);g_bind(e,out,4);[e setBytes:&shape length:sizeof(shape) atIndex:5];
 if(!experts&&g->prefill->bf16Gemm&&g->prefill->matrix&&m.rows>=32&&shape.tokens>=64){
  [e setComputePipelineState:g->prefill->bf16ps[m.bits==8?1:0]];
  [e dispatchThreadgroups:MTLSizeMake((m.rows+63)/64,(shape.tokens+63)/64,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g->dispatches++;return;}
 if(!experts&&g->prefill->matrix&&m.rows>=128){uint32_t tile=g->prefill->matrixTokens?g->prefill->matrixTokens:std::min(32u,((shape.tokens+7)/8)*8);
 int pipeline=PMatrix85+int((tile/8-1)*2)+(m.bits==8?1:0);
 p_run(g,e,pipeline,MTLSizeMake((m.rows+g->prefill->matrixRows-1)/g->prefill->matrixRows,(shape.tokens+tile-1)/tile,1),g->prefill->matrixRows*4);return;}
 if(experts&&g->prefill->groupedExperts&&shape.tokens>=64){auto *p=g->prefill;
  g_bind(e,p->groupedRoutes,6);g_bind(e,p->groupedCounts,7);g_bind(e,p->groupedTiles,8);[e setComputePipelineState:p->bf16Gemm?(p->packedExperts?p->packedDownPs:p->bf16ps[3]):p->ps[PGroupedDown]];
  [e dispatchThreadgroupsWithIndirectBuffer:p->groupedDispatch indirectBufferOffset:16 threadsPerThreadgroup:MTLSizeMake(p->bf16Gemm?(p->packedExperts?32:128):256,1,1)];g->dispatches++;return;}
 if(experts)g_bind(e,g->prefill->ids,6);
 p_run(g,e,experts?PDown:m.bits==5?PQ5:PQ8,MTLSizeMake((m.rows+3)/4,experts?8:(shape.tokens+3)/4,experts?shape.tokens:1),128);
}
static void p_gate(GModel *g,id<MTLComputeCommandEncoder> e,QMatrix &gate,QMatrix &up,id<MTLBuffer> out,PShape shape,bool experts) {
 shape.rows=gate.rows;shape.columns=gate.columns;shape.bits=gate.bits;
 bool direct=experts&&g->prefill->directExperts&&out==g->prefill->expertAct&&gate.rows<=512&&shape.tokens>=64;
 if(out==g->prefill->expertAct)g->prefill->expertActDirect=direct;
 if(experts&&g->prefill->bf16Gemm&&g->prefill->packedExperts&&shape.tokens>=64)p_pack_expert(g,e,shape,true,g->prefill->norm);
 g_bind(e,gate.packed,0);g_bind(e,gate.scales,1);g_bind(e,gate.biases,2);g_bind(e,up.packed,3);g_bind(e,up.scales,4);g_bind(e,up.biases,5);
 g_bind(e,experts&&g->prefill->bf16Gemm&&g->prefill->packedExperts&&shape.tokens>=64?g->prefill->expandedInput:g->prefill->norm,6);g_bind(e,out,7);[e setBytes:&shape length:sizeof(shape) atIndex:8]; if(experts)g_bind(e,g->prefill->ids,9);
 if(g->prefill->bf16Gemm&&g->prefill->expandedGemm&&shape.tokens>=64&&(!experts||(g->prefill->groupedExperts&&g->prefill->expandedExperts))){p_expanded(g,e,gate,&up,g->prefill->norm,out,shape,experts);return;}
 if(experts&&g->prefill->groupedExperts&&shape.tokens>=64){auto *p=g->prefill;
  g_bind(e,p->groupedRoutes,9);g_bind(e,p->groupedCounts,10);g_bind(e,p->groupedTiles,11);[e setComputePipelineState:p->bf16Gemm?(p->packedExperts?(direct?p->packedGateDirectPs:p->packedGatePs):p->bf16ps[2]):p->ps[PGroupedGate]];
  [e dispatchThreadgroupsWithIndirectBuffer:p->groupedDispatch indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(p->bf16Gemm?(p->packedExperts?64:128):256,1,1)];g->dispatches++;return;}
 p_run(g,e,experts?(gate.bits==8?PExperts8:PExperts):PShared,MTLSizeMake((gate.rows+3)/4,experts?8:(shape.tokens+3)/4,experts?shape.tokens:1),128);
}
extern "C" int di_graph_prefill_range(void *graph,const int32_t *tokens,int count,int classifier,int begin,int end,const float *hidden,float *output,float *logits,float *trace,
 int *nextToken,int *dispatches,double *ms,char *error,int capacity) {
 @autoreleasepool {
 if(!graph||!tokens||!((begin==0&&end==40)||(begin==20&&end==40&&hidden))||count<1||count>2048||!nextToken||!dispatches){error_text(error,capacity,@"Invalid prefill chunk");return -1;}
 auto *g=static_cast<GModel*>(graph);if(g->position+count>g->capacity||g->forcedRoutes){error_text(error,capacity,@"Invalid prefill context/diagnostic routing");return -1;}
 if(begin==0&&g->splitActive){error_text(error,capacity,@"Reset split graph before full prefill");return -1;}
 if(begin==20){
  if(g->splitTarget&&g->position+count>g->splitTarget){error_text(error,capacity,@"Split prefix target exceeded");return -1;}
  if(!g->splitActive){if(g->position){error_text(error,capacity,@"Reset before split prefill");return -1;}g->splitActive=true;}
  g->splitMissing=(1ull<<40)-1;
 }
 for(int t=0;t<count;t++)if(tokens[t]<0||tokens[t]>=248320){error_text(error,capacity,@"Invalid prefill token");return -1;}
 if(!g->prefill&&!p_create(g,error,capacity))return -1;auto *p=g->prefill;
 if((uint32_t)count>p->maxTokens){error_text(error,capacity,@"Prefill chunk exceeds configured scratch capacity");return -1;}
 if(trace&&p->trace.length<NSUInteger(count)*40*2048*4){p->trace=g_buffer(g,NSUInteger(count)*40*2048);if(!p->trace){error_text(error,capacity,@"Cannot allocate prefill trace");return -1;}}
 memcpy(p->tokens.contents,tokens,count*4);p->submissions=0;
 g->dispatches=0;g->profileCount=0;PShape shape{0,0,(uint32_t)count,g->position,0,(uint32_t)(trace!=nullptr),g->epsilon,g->theta,g->capacity};
 auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:g->concurrent?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial];
 if(!command||!e){error_text(error,capacity,@"Cannot create prefill command");return -1;}
 bool profiling=std::getenv("DEEPINFER_METAL_PREFILL_PROFILE")&&strcmp(std::getenv("DEEPINFER_METAL_PREFILL_PROFILE"),"1")==0&&g->position==4096;
 double profileMs=0;int profileDispatches=0;
 const char *routeSpec=std::getenv("DEEPINFER_METAL_ROUTE_PROBE");
 int routeLayer=0,routeTokenA=49,routeTokenB=91;
 if(routeSpec&&std::sscanf(routeSpec,"%d:%d",&routeLayer,&routeTokenA)==2)routeTokenB=-1;
 auto checkpoint=[&](const char *stage)->bool {
  bool routeProbe=trace&&routeSpec&&shape.layer==uint32_t(routeLayer)&&strcmp(stage,"routing")==0;
  if(!profiling&&!routeProbe)return true;
  [e endEncoding];double interval=0;if(!q_complete(command,&interval,error,capacity))return false;
  p->submissions++;profileMs+=interval;
  if(profiling)fprintf(stderr,"PREFILL_STAGE {\"position\":%u,\"tokens\":%d,\"layer\":%u,\"stage\":\"%s\",\"gpuMs\":%.6f,\"kernels\":%d}\n",g->position,count,shape.layer,stage,interval,g->dispatches-profileDispatches);
  if(routeProbe)for(int token:{routeTokenA,routeTokenB})if(token>=0&&g->position<=uint32_t(token)&&uint32_t(token)<g->position+count){
   auto *ids=(uint32_t*)p->ids.contents;auto *prob=(float*)p->prob.contents;
   int local=token-int(g->position);fprintf(stderr,"METAL_ROUTE token=%d layer=%u",token,shape.layer);
   for(int i=0;i<8;i++)fprintf(stderr," %u:%.9g",ids[local*8+i],prob[local*8+i]);
   fputc('\n',stderr);
   if(routeTokenB<0){
    auto *norm=(float*)p->norm.contents+local*2048,*logits=(float*)p->router.contents+local*256;
    fprintf(stderr,"METAL_ROUTE_NORM token=%d layer=%u",token,shape.layer);
    for(int i=0;i<2048;i++)fprintf(stderr," %.9g",norm[i]);
    fputc('\n',stderr);
    fprintf(stderr,"METAL_ROUTE_LOGITS token=%d layer=%u",token,shape.layer);
    for(int i=0;i<256;i++)fprintf(stderr," %.9g",logits[i]);
    fputc('\n',stderr);
   }
  }
  profileDispatches=g->dispatches;
  command=[g->compute.queue commandBuffer];e=[command computeCommandEncoderWithDispatchType:g->concurrent?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial];
  if(!command||!e){error_text(error,capacity,@"Cannot create profile command");return false;}return true;
 };


 if(begin==0){
 g_bind(e,g->embedding.packed,0);g_bind(e,g->embedding.scales,1);g_bind(e,g->embedding.biases,2);g_bind(e,p->tokens,3);g_bind(e,p->x,4);[e setBytes:&g->embedding.bits length:4 atIndex:5];
 p_run(g,e,PEmbed,MTLSizeMake(16,count,1),128);g_barrier(e);
 }else memcpy(p->x.contents,hidden,NSUInteger(count)*2048*4);
 g_bind(e,p->x,0);g_bind(e,g->layers[begin].parameters,1);g_bind(e,p->norm,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];p_run(g,e,PNorm,MTLSizeMake(count,1,1),256);g_barrier(e);
 if(!checkpoint("embedding-norm"))return -1;
 for(int index=begin;index<end;index++) {
  shape.layer=index;auto &l=g->layers[index];auto &m=l.m;
  p_gemv(g,e,m[GQ],p->norm,p->qkv,shape);
  if(l.linear){p_gemv(g,e,m[GZ],p->norm,p->z,shape);p_gemv(g,e,m[GA],p->norm,p->a,shape);p_gemv(g,e,m[GB],p->norm,p->b,shape);g_barrier(e);if(!checkpoint("linear-projections"))return -1;
   g_bind(e,p->qkv,0);g_bind(e,l.history,1);g_bind(e,l.parameters,2,4096*4);g_bind(e,p->conv,3);
   g_bind(e,p->a,4);g_bind(e,p->b,5);g_bind(e,l.parameters,6,36864*4);g_bind(e,l.parameters,7,36896*4);[e setBytes:&shape length:sizeof(shape) atIndex:8];
   p_run(g,e,PLinear,MTLSizeMake(64,count,1),128);g_barrier(e);if(!checkpoint("linear-convolution"))return -1;
   p_delta(g,e,l.state,l.history,shape);if(!checkpoint("delta"))return -1;
   g_bind(e,p->attention,0);g_bind(e,p->z,1);g_bind(e,l.parameters,2,36928*4);[e setBytes:&shape length:sizeof(shape) atIndex:3];p_run(g,e,PDeltaNorm,MTLSizeMake(32,count,1),128);g_barrier(e);if(!checkpoint("delta-norm"))return -1;
  } else {
   p_gemv(g,e,m[GK],p->norm,p->k,shape);p_gemv(g,e,m[GV],p->norm,p->v,shape);g_barrier(e);if(!checkpoint("attention-projections"))return -1;
   g_bind(e,p->qkv,0);g_bind(e,p->k,1);g_bind(e,p->v,2);g_bind(e,l.parameters,3,37056*4);g_bind(e,l.parameters,4,37312*4);
   g_bind(e,p->q,5);g_bind(e,l.keys,6);g_bind(e,l.values,7);[e setBytes:&shape length:sizeof(shape) atIndex:8];p_run(g,e,PHead,MTLSizeMake(18,count,1),256);g_barrier(e);if(!checkpoint("attention-head"))return -1;
   for(uint32_t offset=0;offset<(uint32_t)count;offset+=p->attentionBatchTokens){uint32_t n=std::min(p->attentionBatchTokens,(uint32_t)count-offset);
    if(p->fusedAttention&&(p->attentionBf16||p->fusedAll||g->position+offset+n>2048))p_fused_attention(g,e,p->q,l.keys,l.values,p->qkv,p->attention,n,offset);
    else {
     PShape as=shape;as.tokens=n;as.position+=offset;as.capacity=p->scoreCapacity;
     g_bind(e,p->q,0,NSUInteger(offset)*4096*4);g_bind(e,l.keys,1);g_bind(e,p->scores,2);[e setBytes:&as length:sizeof(as) atIndex:3];
     p_run(g,e,PScores,MTLSizeMake((as.position+n+3)/4,16,n),128);g_barrier(e);
     g_bind(e,p->scores,0);g_bind(e,l.values,1);g_bind(e,p->qkv,2,NSUInteger(offset)*8192*4);g_bind(e,p->attention,3,NSUInteger(offset)*4096*4);[e setBytes:&as length:sizeof(as) atIndex:4];
     if(as.position+n<=2048){p_run(g,e,PAttention,MTLSizeMake(16,n,1),256);g_barrier(e);}
     else g_long_attention(g,e,p->scores,l.values,p->qkv,p->attention,n,offset);
    }
   }
  }
  if(!l.linear&&!checkpoint("attention"))return -1;
  p_gemv(g,e,m[GO],p->attention,p->out,shape);g_barrier(e);if(!checkpoint("output-projection"))return -1;
  g_bind(e,p->x,0);g_bind(e,p->out,1);g_bind(e,l.parameters,2,2048*4);g_bind(e,p->norm,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];p_run(g,e,PAdd,MTLSizeMake(count,1,1),256);g_barrier(e);if(!checkpoint("residual-norm"))return -1;
  if(std::getenv("DEEPINFER_METAL_ROUTER_FP32")&&strcmp(std::getenv("DEEPINFER_METAL_ROUTER_FP32"),"1")==0){
   PShape route=shape;route.rows=m[GR].rows;route.columns=m[GR].columns;route.bits=m[GR].bits;
   g_bind(e,m[GR].packed,0);g_bind(e,m[GR].scales,1);g_bind(e,m[GR].biases,2);
   g_bind(e,p->norm,3);g_bind(e,p->router,4);[e setBytes:&route length:sizeof(route) atIndex:5];
   p_run(g,e,route.bits==8?PQ8:PQ5,MTLSizeMake((route.rows+3)/4,(count+3)/4,1),128);
  }else p_gemv(g,e,m[GR],p->norm,p->router,shape);
  p_gate(g,e,m[GSG],m[GSU],p->sharedAct,shape,false);p_gemv(g,e,m[GSR],p->norm,p->sharedRoute,shape);g_barrier(e);if(!checkpoint("router-shared-gate"))return -1;
  g_bind(e,p->router,0);g_bind(e,p->ids,1);g_bind(e,p->prob,2);g_bind(e,p->groupedTileCount,3);p_run(g,e,PTop,MTLSizeMake(count,1,1),32);g_barrier(e);
  if(p->groupedExperts&&count>=64)p_build_groups(g,e,shape,m[GEG].rows,m[GED].rows);if(!checkpoint("routing"))return -1;
  p_gate(g,e,m[GEG],m[GEU],p->expertAct,shape,true);g_barrier(e);if(!checkpoint("expert-gate"))return -1;
  p_gemv(g,e,m[GED],p->expertAct,p->expertOut,shape,true);p_gemv(g,e,m[GSD],p->sharedAct,p->sharedOut,shape);g_barrier(e);if(!checkpoint("expert-shared-down"))return -1;
  g_bind(e,p->x,0);g_bind(e,p->expertOut,1);g_bind(e,p->sharedOut,2);g_bind(e,p->sharedRoute,3);g_bind(e,p->prob,4);
  g_bind(e,index==39?g->finalNorm:g->layers[index+1].parameters,5);g_bind(e,p->norm,6);g_bind(e,p->trace,7);[e setBytes:&shape length:sizeof(shape) atIndex:8];p_run(g,e,PMerge,MTLSizeMake(count,1,1),256);g_barrier(e);if(!checkpoint("merge"))return -1;
 }
 // The 8-bit vocabulary matrix adds ~0.5 GiB to a command's resource set.
 // Finish the layer stack before binding it, so long KV + prefill scratch
 // need not be resident in the same command as the classifier. One extra
 // submission per request; no intermediate CPU readback or precision change.
 if(classifier&&g->classifier.bits==8){
  [e endEncoding];double layersMs=0;if(!q_complete(command,&layersMs,error,capacity))return -1;
  p->submissions++;profileMs+=layersMs;
  command=[g->compute.queue commandBuffer];e=[command computeCommandEncoder];
  if(!command||!e){error_text(error,capacity,@"Cannot create classifier command");return -1;}
 }
 if(classifier){QMatrix &m=g->classifier;struct {uint32_t experts,rows,columns,bits,group,count,batched;} qs{1,m.rows,m.columns,m.bits,64,1,0};
  [e setComputePipelineState:m.bits==8?(g->compute.rowTile==8?g->compute.wide8:g->compute.tiled8):(g->compute.rowTile==8?g->compute.wide5:g->compute.tiled5)];g_bind(e,m.packed,0);g_bind(e,m.scales,1);g_bind(e,m.biases,2);g_bind(e,p->norm,3,(count-1)*2048*4);g_bind(e,g->logits,4);[e setBytes:&qs length:sizeof(qs) atIndex:5];g_bind(e,g->zero,6);
  [e dispatchThreadgroups:MTLSizeMake((m.rows+g->compute.rowTile-1)/g->compute.rowTile,1,1) threadsPerThreadgroup:MTLSizeMake(g->compute.rowTile*32,1,1)];g->dispatches++;g_barrier(e);
  g_bind(e,g->logits,0);g_bind(e,g->argValues,1);g_bind(e,g->argIds,2);g_run(g,e,GArgBlocks,970,32);g_barrier(e);g_bind(e,g->argValues,0);g_bind(e,g->argIds,1);g_bind(e,g->nextToken,2);g_run(g,e,GArgFinal,1,32);
 }
 [e endEncoding];double finalMs=0;if(!q_complete(command,&finalMs,error,capacity))return -1;p->submissions++;if(ms)*ms=profileMs+finalMs;
 *nextToken=classifier?static_cast<const int32_t*>(g->nextToken.contents)[0]:-1;
 if(classifier&&logits)memcpy(logits,g->logits.contents,248320*4);if(trace)memcpy(trace,p->trace.contents,count*40*2048*4);
 if(output)memcpy(output,p->x.contents,NSUInteger(count)*2048*4);
 *dispatches=g->dispatches;g->position+=count;return 0;
 }
}

extern "C" int di_graph_prepare_prefill(void *graph,char *error,int capacity){
 @autoreleasepool {if(!graph){error_text(error,capacity,@"Invalid graph");return -1;}auto *g=static_cast<GModel*>(graph);return g->prefill||p_create(g,error,capacity)?0:-1;}
}
extern "C" int di_graph_prepare_prefill_scratch(void *graph,int tokens,char *error,int capacity){
 @autoreleasepool {
  if(!graph||(tokens!=32&&tokens!=64&&tokens!=128&&tokens!=256&&tokens!=512&&tokens!=1024&&tokens!=2048)){
   error_text(error,capacity,@"Invalid prefill scratch capacity");return -1;
  }
  auto *g=static_cast<GModel*>(graph);
  if(g->prefill){error_text(error,capacity,@"Prefill scratch already configured");return -1;}
  return p_create(g,error,capacity,uint32_t(tokens))?0:-1;
 }
}

// Only called between requests, after reset. Grow per layer so the previous
// full KV allocation need not coexist with the new full allocation.
static int g_reserve(void *graph,int tokens,char *error,int capacity,bool preserve) {
 @autoreleasepool {
 if(!graph||tokens<1||tokens>262144){error_text(error,capacity,@"Context must be 1..262144 tokens; no RoPE scaling is implemented");return -1;}
 auto *g=static_cast<GModel*>(graph);
 if(g->position&&!preserve){error_text(error,capacity,@"Reset graph before reserving context");return -1;}
 if((uint32_t)tokens<=g->capacity)return 0;
 // Reserve a bounded 2K page, not the next power of two: 128K + decode must not allocate 256K.
 uint32_t target=((uint32_t(tokens)+2047)/2048)*2048;
 for(auto &l:g->layers)if(l.loaded&&!l.linear){
  auto k=g_kv_buffer(g,NSUInteger(target)*512),v=g_kv_buffer(g,NSUInteger(target)*512);
  if(!k||!v){error_text(error,capacity,@"Cannot allocate KV cache for requested context");return -1;}
  if(preserve&&g->position){NSUInteger bytes=NSUInteger(g->position)*512*(g->kvBf16?2:4);memcpy(k.contents,l.keys.contents,bytes);memcpy(v.contents,l.values.contents,bytes);}
  l.keys=k;l.values=v;
 }
 auto scores=g_buffer(g,NSUInteger(16)*target);
 bool legacy=g->prefill&&!g->prefill->fusedAttention;
 auto pscores=legacy?g_buffer(g,NSUInteger(std::min(128u,g->prefill->maxTokens))*16*target):nil;
 auto parts=g_buffer(g,NSUInteger(g->prefill?std::min(g->prefill->attentionBatchTokens,g->prefill->maxTokens):32)*16*((target+1023)/1024)*258);
 if(!scores||(legacy&&!pscores)||!parts){error_text(error,capacity,@"Cannot allocate long-attention scratch");return -1;}
 g->scores=scores;if(legacy){g->prefill->scores=pscores;g->prefill->scoreCapacity=target;}g->attentionParts=parts;g->capacity=target;return 0;
 }
}
extern "C" int di_graph_reserve(void *graph,int tokens,char *error,int capacity){return g_reserve(graph,tokens,error,capacity,false);}
extern "C" int di_graph_grow(void *graph,int tokens,char *error,int capacity){return g_reserve(graph,tokens,error,capacity,true);}
// Read the effective native settings, not the managed environment cache. Version 1 is 14 ints.
extern "C" int di_graph_execution_profile(void *graph,int *values,int count) {
 if(!graph||!values||count!=14)return -1;
 auto *g=static_cast<GModel*>(graph);auto *p=g->prefill;if(!p)return -1;
 const int actual[]={g->kvBf16?2:4,p->bf16Gemm,p->matrix,p->chunkedDelta,p->groupedExperts,
  p->packedExperts,p->expandedGemm,p->deviceDelta,int(p->maxTokens),p->fusedAttention,
  p->fusedAll,int(p->attentionFeatures),int(p->attentionBatchTokens),int(p->matrixRows)};
 memcpy(values,actual,sizeof(actual));return 0;
}
extern "C" int di_graph_stats(void *graph,int *capacity,uint64_t *kvBytes,uint64_t *allocatedBytes) {
 if(!graph||!capacity||!kvBytes||!allocatedBytes)return -1;
 auto *g=static_cast<GModel*>(graph);*capacity=g->capacity;*kvBytes=uint64_t(g->capacity)*(g->kvBf16?20480:40960);
 *allocatedBytes=g->compute.device.currentAllocatedSize;return 0;
}

extern "C" int di_graph_prefill_submissions(void *graph){auto *g=static_cast<GModel*>(graph);return g&&g->prefill?g->prefill->submissions:0;}

// Existing full-model ABI remains unchanged.
extern "C" int di_graph_prefill(void *graph,const int32_t *tokens,int count,int classifier,float *logits,float *trace,
 int *nextToken,int *dispatches,double *ms,char *error,int capacity){
 return di_graph_prefill_range(graph,tokens,count,classifier,0,40,nullptr,nullptr,logits,trace,nextToken,dispatches,ms,error,capacity);
}
// Synchronous shared-memory copy after completed GPU submissions. State layout is
// [position,kv-head,dimension] BF16, or [value-head,value-dim,key-dim] FP32.
extern "C" int di_graph_prefix_state(void *graph,int layer,int kind,int position,void *data,int bytes,int importState){
 if(!graph||!data||layer<0||layer>=20||kind<0||kind>1||position<1)return -1;
 auto *g=static_cast<GModel*>(graph);if(g->position!=(uint32_t)position)return -1;
 auto &l=g->layers[layer];int expected=l.linear?(kind?3*8192*4:32*128*128*4):position*512*(g->kvBf16?2:4);
 if(bytes!=expected)return -1;
 uint64_t bit=1ull<<(layer*2+kind);
 if(importState&&(!g->splitActive||!(g->splitMissing&bit)||(!l.linear&&g->splitKvWritten[layer*2+kind]!=0)))return -1;
 auto buffer=l.linear?(kind?l.history:l.state):(kind?l.values:l.keys);
 if(importState){memcpy(buffer.contents,data,bytes);g->splitMissing&=~bit;}
 else memcpy(data,buffer.contents,bytes);
 return 0;
}

// Ordered, lossless KV slices, imported only after the corresponding suffix
// submission completes. Final recurrent states remain mandatory before decode.
extern "C" int di_graph_prefix_kv_range(void *graph,int layer,int kind,int position,int count,int total,const void *data,int bytes){
 if(!graph||!data||layer<0||layer>=20||(layer+1)%4!=0||kind<0||kind>1||position<0||count<1||count>512||
    total<1||total>262144||position>total-count)return -1;
 auto *g=static_cast<GModel*>(graph);auto &l=g->layers[layer];auto index=layer*2+kind;uint64_t bit=1ull<<index;
 if(!g->splitActive||!l.loaded||l.linear||!(g->splitMissing&bit)||(uint32_t)total>g->capacity||
    (uint32_t)(position+count)>g->position||g->splitKvWritten[index]!=(uint32_t)position||
    (g->splitTarget&&g->splitTarget!=(uint32_t)total)||bytes!=count*512*(g->kvBf16?2:4))return -1;
 auto buffer=kind?l.values:l.keys;if(!buffer)return -1;
 g->splitTarget=uint32_t(total);
 memcpy(static_cast<unsigned char*>(buffer.contents)+size_t(position)*512*(g->kvBf16?2:4),data,bytes);
 g->splitKvWritten[index]+=uint32_t(count);
 if(g->splitKvWritten[index]==g->splitTarget)g->splitMissing&=~bit;
 return 0;
}

// Explicit configuration avoids relying on managed environment changes reaching libc.
extern "C" int di_graph_decode_fused_fp32(void *graph,int enabled){
 if(!graph||(enabled!=0&&enabled!=1))return -1;
 auto *g=static_cast<GModel*>(graph);
 if(g->position||g->splitActive||(enabled&&(!g->prefill||g->kvBf16||g->prefill->attentionBf16)))return -1;
 g->decodeFusedFp32=enabled!=0;return 0;
}
