// Original fixed-geometry Qwen3.6 GPU executor. One submitted command buffer per
// token; tensors, convolution history, recurrent state and KV remain on device.
#include <array>
#include <atomic>
static NSString *graph_source = @R"metal(
#include <metal_stdlib>
using namespace metal;
#ifdef DEEPINFER_KV_BF16
typedef bfloat DIKV;
#else
typedef float DIKV;
#endif
struct Step { uint position,layer,trace; float epsilon,theta; uint capacity; };
inline float sigmoid(float x){return 1.0f/(1.0f+exp(-x));}
inline float silu(float x){return x*sigmoid(x);}
// 256-thread reductions use eight SIMD groups and one shared scalar.
inline float reduce_sum(float value,threadgroup float *partials,uint tid,uint lane,uint sg) {
    value=simd_sum(value);if(lane==0)partials[sg]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float total=0;for(uint i=0;i<8;i++)total+=partials[i];partials[0]=total;}
    threadgroup_barrier(mem_flags::mem_threadgroup);return partials[0];
}
kernel void graph_embedding(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]],
 device const ushort *biases [[buffer(2)]],device float *x [[buffer(3)]],constant uint &token [[buffer(4)]],constant uint &bits [[buffer(5)]],
 uint c [[thread_position_in_grid]]) {
    if(c>=2048)return;
    uint bit=c*bits,at=bit>>5,shift=bit&31;
    ulong base=ulong(token)*(2048*bits/32);uint q=w[base+at]>>shift;
    if(shift+bits>32)q|=w[base+at+1]<<(32-shift);
    uint p=token*32+c/64;
    x[c]=float(q&((1u<<bits)-1))*as_type<float>(uint(scales[p])<<16)+as_type<float>(uint(biases[p])<<16);
}
kernel void graph_norm(device const float *x [[buffer(0)]],device const float *weight [[buffer(1)]],
 device float *out [[buffer(2)]],constant Step &s [[buffer(3)]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];float total=0;
    for(uint d=tid;d<2048;d+=256)total+=x[d]*x[d];
    float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
    for(uint d=tid;d<2048;d+=256)out[d]=x[d]*scale*weight[d];
}
kernel void graph_add_norm(device float *x [[buffer(0)]],device const float *add [[buffer(1)]],
 device const float *weight [[buffer(2)]],device float *out [[buffer(3)]],constant Step &s [[buffer(4)]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];float values[8],total=0;
    for(uint i=0;i<8;i++){uint d=tid+256*i;values[i]=x[d]+add[d];total+=values[i]*values[i];}
    float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
    for(uint i=0;i<8;i++){uint d=tid+256*i;x[d]=values[i];out[d]=values[i]*scale*weight[d];}
}
kernel void graph_merge_norm(device float *x [[buffer(0)]],device const float *expert [[buffer(1)]],
 device const float *shared [[buffer(2)]],device const float *route [[buffer(3)]],device const float *prob [[buffer(4)]],
 device const float *weight [[buffer(5)]],device float *out [[buffer(6)]],device float *trace [[buffer(7)]],constant Step &s [[buffer(8)]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];float values[8],total=0;
    for(uint i=0;i<8;i++) {
        uint d=tid+256*i;float y=shared[d]*sigmoid(route[0]);
        for(uint e=0;e<8;e++)y+=expert[e*2048+d]*prob[e];
        values[i]=x[d]+y;total+=values[i]*values[i];
    }
    float scale=rsqrt(reduce_sum(total,partials,tid,lane,sg)/2048.0f+s.epsilon);
    for(uint i=0;i<8;i++){uint d=tid+256*i;x[d]=values[i];out[d]=values[i]*scale*weight[d];
        if(s.trace)trace[s.layer*2048+d]=values[i];}
}
kernel void graph_linear_prepare(device const float *raw [[buffer(0)]],device float *history [[buffer(1)]],
 device const float *conv [[buffer(2)]],device float *out [[buffer(3)]],device float *a [[buffer(4)]],
 device float *b [[buffer(5)]],device const float *alog [[buffer(6)]],device const float *dt [[buffer(7)]],
 uint head [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint d=head*128+tid;
    float value=silu(history[d]*conv[d*4]+history[8192+d]*conv[d*4+1]+
        history[16384+d]*conv[d*4+2]+raw[d]*conv[d*4+3]);
    history[d]=history[8192+d];history[8192+d]=history[16384+d];history[16384+d]=raw[d];
    if(head<32) {
        threadgroup float partials[4];float sum=simd_sum(value*value);if(lane==0)partials[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0)partials[0]=rsqrt(partials[0]+partials[1]+partials[2]+partials[3]+1e-6f)*(head<16?rsqrt(128.0f):1.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);value*=partials[0];
    }
    out[d]=value;
    if(head==0&&tid<32){float t=a[tid]+dt[tid];float softplus=t>20?t:log(1.0f+exp(t));
        a[tid]=exp(-exp(alog[tid])*softplus);b[tid]=sigmoid(b[tid]);}
}
kernel void graph_delta_norm(device float *x [[buffer(0)]],device const float *z [[buffer(1)]],
 device const float *weight [[buffer(2)]],constant Step &s [[buffer(3)]],
 uint head [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[4];uint d=head*128+tid;float value=x[d];
    float sum=simd_sum(value*value);if(lane==0)partials[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)partials[0]=rsqrt((partials[0]+partials[1]+partials[2]+partials[3])/128.0f+s.epsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);x[d]=value*partials[0]*weight[tid]*silu(z[d]);
}
kernel void graph_head_prepare(device const float *raw [[buffer(0)]],device const float *k [[buffer(1)]],
 device const float *v [[buffer(2)]],device const float *qw [[buffer(3)]],device const float *kw [[buffer(4)]],
 device float *q [[buffer(5)]],device DIKV *keys [[buffer(6)]],device DIKV *values [[buffer(7)]],constant Step &s [[buffer(8)]],
 uint head [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[8];bool query=head<16;uint h=query?head:head-16;
    device const float *input=query?raw+h*512:k+h*256;
    device const float *weight=query?qw:kw;
    float value=input[tid];float scale=rsqrt(reduce_sum(value*value,partials,tid,lane,sg)/256.0f+s.epsilon);
    value=value*scale*weight[tid];
    if(tid<64) {
        uint frequency=tid%32,partner=tid<32?tid+32:tid-32;
        float angle=float(s.position)/pow(s.theta,float(frequency)/32.0f);
        float other=input[partner]*scale*weight[partner];
        value=value*cos(angle)+(tid<32?-other:other)*sin(angle);
    }
    if(query)q[h*256+tid]=value;
    else {keys[s.position*512+h*256+tid]=DIKV(value);values[s.position*512+h*256+tid]=DIKV(v[h*256+tid]);}
}
kernel void graph_attention_scores(device const float *q [[buffer(0)]],device const DIKV *keys [[buffer(1)]],
 device float *scores [[buffer(2)]],constant Step &s [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint t=group.x*4+sg,h=group.y;if(t>s.position)return;
    float sum=0;for(uint d=lane;d<256;d+=32)sum+=q[h*256+d]*float(keys[t*512+(h/8)*256+d]);
    sum=simd_sum(sum);if(lane==0)scores[h*s.capacity+t]=sum/16.0f;
}
kernel void graph_attention_output(device const float *scores [[buffer(0)]],device const DIKV *values [[buffer(1)]],
 device const float *raw [[buffer(2)]],device float *out [[buffer(3)]],constant Step &s [[buffer(4)]],
 uint h [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float prob[2048];threadgroup float partials[8];threadgroup float maximum;float best=-INFINITY;
    for(uint t=tid;t<=s.position;t+=256)best=max(best,scores[h*s.capacity+t]);
    best=simd_max(best);if(lane==0)partials[sg]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float m=partials[0];for(uint i=1;i<8;i++)m=max(m,partials[i]);maximum=m;}
    threadgroup_barrier(mem_flags::mem_threadgroup);best=maximum;float sum=0;
    for(uint t=tid;t<=s.position;t+=256){float p=exp(scores[h*s.capacity+t]-best);prob[t]=p;sum+=p;}
    sum=reduce_sum(sum,partials,tid,lane,sg);
    for(uint t=tid;t<=s.position;t+=256)prob[t]/=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);float result=0;
    for(uint t=0;t<=s.position;t++)result+=prob[t]*float(values[t*512+(h/8)*256+tid]);
    out[h*256+tid]=result*sigmoid(raw[h*512+256+tid]);
}
kernel void graph_top8(device const float *logits [[buffer(0)]],device uint *ids [[buffer(1)]],
 device float *prob [[buffer(2)]],uint lane [[thread_index_in_simdgroup]]) {
    float candidates[8],chosen[8];for(uint j=0;j<8;j++)candidates[j]=logits[lane+32*j];
    for(uint n=0;n<8;n++) {
        float best=candidates[0];uint id=lane;
        for(uint j=1;j<8;j++)if(candidates[j]>best){best=candidates[j];id=lane+32*j;}
        float maximum=simd_max(best);uint selected=simd_min(best==maximum?id:0xffffffffu);
        if(lane==0){ids[n]=selected;chosen[n]=maximum;}
        if((selected&31)==lane)candidates[selected>>5]=-INFINITY;
    }
    if(lane==0){float total=0;for(uint i=0;i<8;i++){prob[i]=exp(chosen[i]-chosen[0]);total+=prob[i];}
        for(uint i=0;i<8;i++)prob[i]/=total;}
}
kernel void graph_debug_router(device const float *router [[buffer(0)]],device const uint *ids [[buffer(1)]],
 device float *out [[buffer(2)]],constant Step &s [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    if(tid<256)out[s.layer*264+tid]=router[tid];if(tid<8)out[s.layer*264+256+tid]=float(ids[tid]);
}
// Differential verification only: hold discrete choices fixed while comparing
// long-state floating-point operators. Normal inference never binds this path.
kernel void graph_force_route(device const float *router [[buffer(0)]],device uint *ids [[buffer(1)]],
 device float *prob [[buffer(2)]],device const uint *forced [[buffer(3)]],constant Step &s [[buffer(4)]],uint lane [[thread_index_in_simdgroup]]) {
    if(lane==0){uint offset=(s.position*40+s.layer)*8;float best=-INFINITY;
        for(uint i=0;i<8;i++){ids[i]=forced[offset+i];best=max(best,router[ids[i]]);}
        float sum=0;for(uint i=0;i<8;i++){prob[i]=exp(router[ids[i]]-best);sum+=prob[i];}
        for(uint i=0;i<8;i++)prob[i]/=sum;}
}
kernel void graph_argmax_blocks(device const float *logits [[buffer(0)]],device float *values [[buffer(1)]],
 device uint *ids [[buffer(2)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    float best=-INFINITY;uint id=0xffffffffu;bool invalid=false;
    for(uint i=0;i<8;i++){uint at=group*256+lane+32*i;float v=logits[at];invalid|=!isfinite(v);
        if(v>best||(v==best&&at<id)){best=v;id=at;}}
    float maximum=simd_max(best);uint chosen=simd_min(best==maximum?id:0xffffffffu);invalid=simd_any(invalid);
    if(lane==0){values[group]=maximum;ids[group]=invalid?0xffffffffu:chosen;}
}
kernel void graph_argmax_final(device const float *values [[buffer(0)]],device const uint *ids [[buffer(1)]],
 device uint *out [[buffer(2)]],uint lane [[thread_index_in_simdgroup]]) {
    float best=-INFINITY;uint id=0xffffffffu;bool invalid=false;
    for(uint i=lane;i<970;i+=32){float v=values[i];invalid|=ids[i]==0xffffffffu;
        if(v>best||(v==best&&ids[i]<id)){best=v;id=ids[i];}}
    float maximum=simd_max(best);uint chosen=simd_min(best==maximum?id:0xffffffffu);invalid=simd_any(invalid);
    if(lane==0)out[0]=invalid?0xffffffffu:chosen;
}

// Long-context softmax uses 1024-key partitions, then combines their stable
// (max, sum, weighted value) statistics. No context-sized threadgroup arrays.
struct AttentionShape {uint position,tokens,capacity,partitions;};
kernel void graph_attention_split(device const float *scores [[buffer(0)]],device const DIKV *values [[buffer(1)]],
 device float *parts [[buffer(2)]],constant AttentionShape &s [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint part=group.x,h=group.y,token=group.z,position=s.position+token,start=part*1024;
    device float *out=parts+((token*16+h)*s.partitions+part)*258;
    if(start>position){if(tid==0){out[0]=-INFINITY;out[1]=0;}out[2+tid]=0;return;}
    scores+=(token*16+h)*s.capacity;uint count=min(1024u,position+1-start);
    threadgroup float prob[1024],partials[8];threadgroup float maximum;
    float best=-INFINITY;for(uint i=tid;i<count;i+=256)best=max(best,scores[start+i]);
    best=simd_max(best);if(lane==0)partials[sg]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float m=partials[0];for(uint i=1;i<8;i++)m=max(m,partials[i]);maximum=m;}
    threadgroup_barrier(mem_flags::mem_threadgroup);float sum=0;
    for(uint i=tid;i<count;i+=256){float p=exp(scores[start+i]-maximum);prob[i]=p;sum+=p;}
    sum=reduce_sum(sum,partials,tid,lane,sg);
    if(tid==0){out[0]=maximum;out[1]=sum;}
    float result=0;for(uint i=0;i<count;i++)result+=prob[i]*float(values[(start+i)*512+(h/8)*256+tid]);
    out[2+tid]=result;
}
kernel void graph_attention_combine(device const float *parts [[buffer(0)]],device const float *raw [[buffer(1)]],
 device float *out [[buffer(2)]],constant AttentionShape &s [[buffer(3)]],uint3 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint h=group.x,token=group.y;parts+=(token*16+h)*s.partitions*258;
    threadgroup float partials[8];threadgroup float maximum;
    float best=-INFINITY;for(uint p=tid;p<s.partitions;p+=256)best=max(best,parts[p*258]);
    best=simd_max(best);if(lane==0)partials[sg]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0){float m=partials[0];for(uint i=1;i<8;i++)m=max(m,partials[i]);maximum=m;}
    threadgroup_barrier(mem_flags::mem_threadgroup);float total=0;
    for(uint p=tid;p<s.partitions;p+=256)total+=exp(parts[p*258]-maximum)*parts[p*258+1];
    total=reduce_sum(total,partials,tid,lane,sg);float result=0;
    for(uint p=0;p<s.partitions;p++)result+=exp(parts[p*258]-maximum)*parts[p*258+2+tid];
    out[token*4096+h*256+tid]=result/total*sigmoid(raw[token*8192+h*512+256+tid]);
}

)metal";

enum GM { GQ,GK,GV,GZ,GA,GB,GO,GR,GEG,GEU,GED,GSG,GSU,GSD,GSR,GMCount };
enum GP { GEmbed,GNorm,GAddNorm,GMerge,GLinear,GDeltaNorm,GHead,GScores,GAttention,GTop8,GDebug,GForce,GArgBlocks,GArgFinal,GAttentionSplit,GAttentionCombine,GPCount };
struct GStep {uint32_t position,layer,trace;float epsilon,theta;uint32_t capacity=2048;};
struct AttentionShape {uint32_t position,tokens,capacity,partitions;};
struct GLayer {
    std::array<QMatrix,GMCount> m{};
    id<MTLBuffer> parameters,history,state,keys,values;
    bool linear=false,loaded=false;
};
struct GPrefill;
static void g_prefill_destroy(GPrefill *p);
struct GSnapshotStore;
static void g_snapshot_destroy(GSnapshotStore *s);
static uint64_t g_next_graph_identity(){static std::atomic<uint64_t> next{1};return next.fetch_add(1);}
struct GModel {
    QContext compute;
    std::array<id<MTLComputePipelineState>,GPCount> ps;
    QMatrix embedding{},classifier{};
    std::array<GLayer,40> layers;
    id<MTLBuffer> finalNorm,x,norm,attention,out,qkv,z,a,b,conv,q,k,v,router,ids,prob,sharedAct,sharedRoute,
        sharedOut,expertAct,expertOut,scores,logits,zero,trace;
    float epsilon,theta;
    uint32_t position=0,capacity=2048;
    uint64_t splitMissing=0; // Two state tensors per prefix layer; decode requires all 40.
    bool splitActive=false;
    bool decodeFusedFp32=false;
    uint32_t splitTarget=0;
    std::array<uint32_t,40> splitKvWritten{};
    id<MTLBuffer> attentionParts;
    int dispatches=0;
    id<MTLCounterSampleBuffer> counters;
    std::array<int,1024> profileTags{};
    uint32_t profileCount=0;
    id<MTLBuffer> debugRouter;
    uint32_t debugPosition=511;
    id<MTLBuffer> forcedRoutes;
    uint32_t forcedPositions=0;
    id<MTLBuffer> argValues,argIds,nextToken;
    bool concurrent=true,kvBf16=false;
    GPrefill *prefill=nullptr;
    // Opt-in DFlash2 target feature taps (gpu_dflash.mm). All stay empty unless di_dflash_create ran.
    id<MTLBuffer> tapBuffer;id<MTLComputePipelineState> tapPipeline;uint32_t tapRows=0,tapValidRows=0,tapBasePosition=0;bool tapsEnabled=false;
    GSnapshotStore *snapshots=nullptr;
    uint64_t instanceId=g_next_graph_identity(),epoch=1; // epoch changes on every reset
    ~GModel(){g_prefill_destroy(prefill);g_snapshot_destroy(snapshots);}
};
static bool g_try_fused_attention(GModel *g,id<MTLComputeCommandEncoder> e,GLayer &layer);
static bool g_profiling(GModel *g){return g->counters&&g->position==32;}
static void g_profile_begin(GModel *g,id<MTLComputeCommandEncoder> e,int tag){
    if(g_profiling(g)&&g->profileCount<g->profileTags.size()){g->profileTags[g->profileCount]=tag;[e sampleCountersInBuffer:g->counters atSampleIndex:g->profileCount*2 withBarrier:YES];}
}
static void g_profile_end(GModel *g,id<MTLComputeCommandEncoder> e){
    if(g_profiling(g)&&g->profileCount<g->profileTags.size()){[e sampleCountersInBuffer:g->counters atSampleIndex:g->profileCount*2+1 withBarrier:YES];g->profileCount++;}
}
static id<MTLBuffer> g_buffer(GModel *g,NSUInteger floats){auto b=[g->compute.device newBufferWithLength:floats*4 options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);return b;}
static id<MTLBuffer> g_kv_buffer(GModel *g,NSUInteger elements){auto b=[g->compute.device newBufferWithLength:elements*(g->kvBf16?2:4) options:MTLResourceStorageModeShared];q_trace_alloc(g->compute.device);return b;}
static NSString *g_kv_source(GModel *g){return g->kvBf16?[@"#define DEEPINFER_KV_BF16\n" stringByAppendingString:graph_source]:graph_source;}
static void g_bind(id<MTLComputeCommandEncoder> e,id<MTLBuffer> b,int index,NSUInteger offset=0){[e setBuffer:b offset:offset atIndex:index];}
static void g_run(GModel *g,id<MTLComputeCommandEncoder> e,int pipeline,int groups,int threads){
    [e setComputePipelineState:g->ps[pipeline]];
    g_profile_begin(g,e,pipeline);
    [e dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];g->dispatches++;
    g_profile_end(g,e);
}
static void g_barrier(id<MTLComputeCommandEncoder> e){[e memoryBarrierWithScope:MTLBarrierScopeBuffers];}
static void g_long_attention(GModel *g,id<MTLComputeCommandEncoder> e,id<MTLBuffer> scores,id<MTLBuffer> values,
    id<MTLBuffer> raw,id<MTLBuffer> out,uint32_t count,uint32_t offset=0) {
    AttentionShape shape{g->position+offset,count,g->capacity,(g->position+offset+count+1023)/1024};
    g_bind(e,scores,0);g_bind(e,values,1);g_bind(e,g->attentionParts,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];
    [e setComputePipelineState:g->ps[GAttentionSplit]];
    [e dispatchThreadgroups:MTLSizeMake(shape.partitions,16,count) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
    g_bind(e,g->attentionParts,0);g_bind(e,raw,1,NSUInteger(offset)*8192*4);g_bind(e,out,2,NSUInteger(offset)*4096*4);[e setBytes:&shape length:sizeof(shape) atIndex:3];
    [e setComputePipelineState:g->ps[GAttentionCombine]];
    [e dispatchThreadgroups:MTLSizeMake(16,count,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
}
static void g_gemv(GModel *g,id<MTLComputeCommandEncoder> e,QMatrix &m,id<MTLBuffer> x,id<MTLBuffer> out,bool experts=false){
    auto *c=&g->compute;bool small=m.rows<128;
    struct {uint32_t experts,rows,columns,bits,group,count,batched;} shape{m.experts,m.rows,m.columns,m.bits,m.group,experts?8u:1u,experts?1u:0u};
    [e setComputePipelineState:m.bits==5?(small?c->small5:c->rowTile==8?c->wide5:c->tiled5):(small?c->small8:c->rowTile==8?c->wide8:c->tiled8)];
    g_bind(e,m.packed,0);g_bind(e,m.scales,1);g_bind(e,m.biases,2);g_bind(e,x,3);g_bind(e,out,4);
    [e setBytes:&shape length:sizeof(shape) atIndex:5];g_bind(e,experts?g->ids:g->zero,6);
    g_profile_begin(g,e,m.rows==248320?GPCount+2:experts?GPCount+3:m.bits==8?GPCount+1:GPCount);
    [e dispatchThreadgroups:MTLSizeMake((m.rows+(small?0:c->rowTile-1))/(small?1:c->rowTile),experts?8:1,1) threadsPerThreadgroup:MTLSizeMake(small?32:c->rowTile*32,1,1)];g->dispatches++;
    g_profile_end(g,e);
}
// Expert gate/up consumes one shared layer input, even with eight selected slots.
static void g_gate_up(GModel *g,id<MTLComputeCommandEncoder> e,QMatrix &gate,QMatrix &up,id<MTLBuffer> out,bool selected){
    struct {uint32_t experts,rows,columns,bits,group,count,batched;} shape{gate.experts,gate.rows,gate.columns,gate.bits,gate.group,selected?8u:1u,0};
    [e setComputePipelineState:gate.bits==5?g->compute.gateUp5:g->compute.gateUp8];
    g_bind(e,gate.packed,0);g_bind(e,gate.scales,1);g_bind(e,gate.biases,2);
    g_bind(e,up.packed,3);g_bind(e,up.scales,4);g_bind(e,up.biases,5);g_bind(e,g->norm,6);g_bind(e,out,7);
    [e setBytes:&shape length:sizeof(shape) atIndex:8];g_bind(e,selected?g->ids:g->zero,9);
    g_profile_begin(g,e,selected?GPCount+5:GPCount+4);
    [e dispatchThreadgroups:MTLSizeMake((gate.rows+3)/4,selected?8:1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g->dispatches++;
    g_profile_end(g,e);
}
// Gate/up reads a shared vector; the GEMV flag selects per-slot inputs only for down.
static void g_router(GModel *g,id<MTLComputeCommandEncoder> e,GLayer &l,GStep &s,id<MTLBuffer> nextNorm){
    auto &m=l.m;
    g_gemv(g,e,m[GR],g->norm,g->router);g_gate_up(g,e,m[GSG],m[GSU],g->sharedAct,false);
    g_gemv(g,e,m[GSR],g->norm,g->sharedRoute);g_barrier(e);
    g_bind(e,g->router,0);g_bind(e,g->ids,1);g_bind(e,g->prob,2);g_run(g,e,GTop8,1,32);g_barrier(e);
    if(g->forcedRoutes){g_bind(e,g->router,0);g_bind(e,g->ids,1);g_bind(e,g->prob,2);g_bind(e,g->forcedRoutes,3);
        [e setBytes:&s length:sizeof(s) atIndex:4];g_run(g,e,GForce,1,32);g_barrier(e);}
    if(g->debugRouter&&g->position==g->debugPosition){g_bind(e,g->router,0);g_bind(e,g->ids,1);g_bind(e,g->debugRouter,2);[e setBytes:&s length:sizeof(s) atIndex:3];g_run(g,e,GDebug,2,128);g_barrier(e);}
    g_gate_up(g,e,m[GEG],m[GEU],g->expertAct,true);g_barrier(e);
    g_gemv(g,e,m[GED],g->expertAct,g->expertOut,true);g_gemv(g,e,m[GSD],g->sharedAct,g->sharedOut);g_barrier(e);
    g_bind(e,g->x,0);g_bind(e,g->expertOut,1);g_bind(e,g->sharedOut,2);g_bind(e,g->sharedRoute,3);g_bind(e,g->prob,4);
    g_bind(e,nextNorm,5);g_bind(e,g->norm,6);g_bind(e,g->trace,7);[e setBytes:&s length:sizeof(s) atIndex:8];
    g_run(g,e,GMerge,1,256);g_barrier(e);
}
extern "C" {
void di_graph_destroy(void *p){delete static_cast<GModel*>(p);}
void *di_graph_create(void *ctx,QMatrix *embedding,QMatrix *classifier,const float *norm,float epsilon,float theta,char *error,int capacity){
    @autoreleasepool {
        if(!ctx||!embedding||!classifier||!norm||embedding->rows!=248320||embedding->columns!=2048||(embedding->bits!=5&&embedding->bits!=8)||
            classifier->rows!=248320||classifier->columns!=2048||(classifier->bits!=5&&classifier->bits!=8)){error_text(error,capacity,@"Unsupported graph weights");return nullptr;}
        auto *g=new GModel{};g->compute=*static_cast<QContext*>(ctx);g->embedding=*embedding;g->classifier=*classifier;g->epsilon=epsilon;g->theta=theta;
        if(const char *serial=std::getenv("DEEPINFER_METAL_SERIAL_GRAPH"))g->concurrent=strcmp(serial,"1")!=0;
        const char *profile=std::getenv("DEEPINFER_METAL_GPU_PROFILE");
        if(profile&&strcmp(profile,"1")==0) {
            if([g->compute.device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
                for(id<MTLCounterSet> set in g->compute.device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    MTLCounterSampleBufferDescriptor *d=[MTLCounterSampleBufferDescriptor new];d.counterSet=set;d.storageMode=MTLStorageModeShared;d.sampleCount=2048;
                    g->counters=[g->compute.device newCounterSampleBufferWithDescriptor:d error:nil];
                }
            }
            if(!g->counters)fprintf(stderr,"GPU per-dispatch timestamps unavailable\n");
        }
        NSError *failure=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        if(const char *kv=std::getenv("DEEPINFER_METAL_KV")){
            if(strcmp(kv,"bf16")&&strcmp(kv,"fp32")){error_text(error,capacity,@"DEEPINFER_METAL_KV must be fp32 or bf16");delete g;return nullptr;}
            g->kvBf16=strcmp(kv,"bf16")==0;
        }
        if(g->kvBf16){if(@available(macOS 15.0,*)){options.languageVersion=MTLLanguageVersion3_1;}}
        auto library=[g->compute.device newLibraryWithSource:g_kv_source(g) options:options error:&failure];
        if(!library){error_text(error,capacity,failure.localizedDescription);delete g;return nullptr;}
        NSArray<NSString*> *names=@[@"graph_embedding",@"graph_norm",@"graph_add_norm",@"graph_merge_norm",@"graph_linear_prepare",@"graph_delta_norm",@"graph_head_prepare",@"graph_attention_scores",@"graph_attention_output",@"graph_top8",@"graph_debug_router",@"graph_force_route",@"graph_argmax_blocks",@"graph_argmax_final",@"graph_attention_split",@"graph_attention_combine"];
        for(int i=0;i<GPCount;i++) {
            g->ps[i]=[g->compute.device newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]] error:&failure];
            if(!g->ps[i]||g->ps[i].threadExecutionWidth!=32||g->ps[i].maxTotalThreadsPerThreadgroup<256){
                error_text(error,capacity,failure.localizedDescription ?: @"Graph kernels require SIMD32/256 threads");delete g;return nullptr;}
        }
        auto buffer=[&](NSUInteger n){return g_buffer(g,n);};
        g->finalNorm=buffer(2048);g->x=buffer(2048);g->norm=buffer(2048);g->attention=buffer(4096);g->out=buffer(2048);
        g->qkv=buffer(8192);g->z=buffer(4096);g->a=buffer(32);g->b=buffer(32);g->conv=buffer(8192);
        g->q=buffer(4096);g->k=buffer(512);g->v=buffer(512);g->router=buffer(256);g->ids=buffer(8);g->prob=buffer(8);
        g->sharedAct=buffer(512);g->sharedRoute=buffer(1);g->sharedOut=buffer(2048);g->expertAct=buffer(4096);g->expertOut=buffer(16384);
        g->scores=buffer(16*2048);g->logits=buffer(248320);g->zero=buffer(1);g->trace=buffer(40*2048);
        g->argValues=buffer(970);g->argIds=buffer(970);g->nextToken=buffer(1);
        if(std::getenv("DEEPINFER_GRAPH_DEBUG_DIR")){g->debugRouter=buffer(40*264);
            if(const char *p=std::getenv("DEEPINFER_GRAPH_DEBUG_POSITION"))g->debugPosition=(uint32_t)std::atoi(p);}
        if(const char *path=std::getenv("DEEPINFER_GRAPH_FORCE_ROUTES")) {
            NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
            if(data.length==0||data.length%(40*8*4)||data.length>2048*40*8*4){error_text(error,capacity,@"Invalid diagnostic route file");delete g;return nullptr;}
            auto *ids=static_cast<const uint32_t*>(data.bytes);
            for(NSUInteger i=0;i<data.length/4;i++)if(ids[i]>=256){error_text(error,capacity,@"Invalid diagnostic expert ID");delete g;return nullptr;}
            g->forcedRoutes=buffer(data.length/4);if(!g->forcedRoutes){delete g;return nullptr;}
            memcpy(g->forcedRoutes.contents,data.bytes,data.length);g->forcedPositions=(uint32_t)(data.length/(40*8*4));
        }
        if(!g->finalNorm||!g->x||!g->norm||!g->attention||!g->out||!g->qkv||!g->z||!g->a||!g->b||!g->conv||!g->q||!g->k||!g->v||
           !g->router||!g->ids||!g->prob||!g->sharedAct||!g->sharedRoute||!g->sharedOut||!g->expertAct||!g->expertOut||!g->scores||!g->logits||!g->zero||!g->trace||!g->argValues||!g->argIds||!g->nextToken){
            error_text(error,capacity,@"Cannot allocate graph buffers");delete g;return nullptr;}
        memcpy(g->finalNorm.contents,norm,2048*4);memset(g->zero.contents,0,4);return g;
    }
}
int di_graph_layer(void *graph,int index,QMatrix **matrices,const float *parameters,int count,char *error,int capacity){
    @autoreleasepool {
        if(!graph||index<0||index>=40||!matrices||!parameters||count!=37568){error_text(error,capacity,@"Invalid graph layer");return -1;}
        auto *g=static_cast<GModel*>(graph);auto &l=g->layers[index];if(l.loaded){error_text(error,capacity,@"Layer already registered");return -1;}
        l.linear=(index+1)%4!=0;
        for(int i=0;i<GMCount;i++)if(matrices[i])l.m[i]=*matrices[i];
        l.parameters=g_buffer(g,37568);
        if(l.linear){l.history=g_buffer(g,3*8192);l.state=g_buffer(g,32*128*128);}
        else {l.keys=g_kv_buffer(g,2048*512);l.values=g_kv_buffer(g,2048*512);}
        if(!l.parameters||(l.linear?(!l.history||!l.state):(!l.keys||!l.values))){error_text(error,capacity,@"Cannot allocate graph layer state");return -1;}
        memcpy(l.parameters.contents,parameters,37568*4);l.loaded=true;return 0;
    }
}
int di_graph_reset(void *graph){
    if(!graph)return -1;auto *g=static_cast<GModel*>(graph);
    for(auto &l:g->layers){if(!l.loaded)return -1;if(l.linear){memset(l.history.contents,0,3*8192*4);memset(l.state.contents,0,32*128*128*4);}}
    g->position=0;g->splitMissing=0;g->splitActive=false;g->splitTarget=0;g->splitKvWritten.fill(0);g->tapValidRows=0;g->epoch++;return 0;
}
int di_graph_forward(void *graph,int token,int position,int classifier,float *logits,float *trace,int *nextToken,int *dispatches,double *ms,char *error,int capacity){
    @autoreleasepool {
        if(!graph||token<0||token>=248320||position<0||position>=262144||!nextToken) {error_text(error,capacity,@"Invalid graph input");return -1;}
        auto *g=static_cast<GModel*>(graph);
        if(g->splitMissing){error_text(error,capacity,@"Split prefix state is incomplete");return -1;}
        if((uint32_t)position>=g->capacity){error_text(error,capacity,@"Reserve graph context before execution");return -1;}
        if((uint32_t)position!=g->position){error_text(error,capacity,@"Graph token positions must be contiguous; reset between requests");return -1;}
        if(g->forcedRoutes&&(uint32_t)position>=g->forcedPositions){error_text(error,capacity,@"Diagnostic route sequence exhausted");return -1;}
        if(g->tapsEnabled)g->tapValidRows=0; // Scalar decode does not populate the prefill/verify tap buffer.
        g->dispatches=0;g->profileCount=0;GStep s{(uint32_t)position,0,(uint32_t)(trace!=nullptr),g->epsilon,g->theta,g->capacity};
        auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:g->concurrent?MTLDispatchTypeConcurrent:MTLDispatchTypeSerial];
        if(!command||!e){error_text(error,capacity,@"Cannot create graph encoder");return -1;}
        g_bind(e,g->embedding.packed,0);g_bind(e,g->embedding.scales,1);g_bind(e,g->embedding.biases,2);g_bind(e,g->x,3);
        [e setBytes:&token length:4 atIndex:4];[e setBytes:&g->embedding.bits length:4 atIndex:5];g_run(g,e,GEmbed,16,128);g_barrier(e);
        g_bind(e,g->x,0);g_bind(e,g->layers[0].parameters,1);g_bind(e,g->norm,2);[e setBytes:&s length:sizeof(s) atIndex:3];
        g_run(g,e,GNorm,1,256);g_barrier(e);
        for(int index=0;index<40;index++) {
            s.layer=index;auto &l=g->layers[index];auto &m=l.m;
            g_gemv(g,e,m[GQ],g->norm,g->qkv);
            if(l.linear) {
                g_gemv(g,e,m[GZ],g->norm,g->z);g_gemv(g,e,m[GA],g->norm,g->a);g_gemv(g,e,m[GB],g->norm,g->b);g_barrier(e);
                g_bind(e,g->qkv,0);g_bind(e,l.history,1);g_bind(e,l.parameters,2,4096*4);g_bind(e,g->conv,3);
                g_bind(e,g->a,4);g_bind(e,g->b,5);g_bind(e,l.parameters,6,36864*4);g_bind(e,l.parameters,7,36896*4);
                g_run(g,e,GLinear,64,128);g_barrier(e);
                [e setComputePipelineState:g->compute.delta];g_bind(e,l.state,0);g_bind(e,g->conv,1);g_bind(e,g->conv,2,2048*4);
                g_bind(e,g->conv,3,4096*4);g_bind(e,g->a,4);g_bind(e,g->b,5);g_bind(e,g->attention,6);
                g_profile_begin(g,e,GPCount+6);
                [e dispatchThreadgroups:MTLSizeMake(2048,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];g->dispatches++;g_barrier(e);
                g_profile_end(g,e);
                g_bind(e,g->attention,0);g_bind(e,g->z,1);g_bind(e,l.parameters,2,36928*4);[e setBytes:&s length:sizeof(s) atIndex:3];
                g_run(g,e,GDeltaNorm,32,128);g_barrier(e);
            } else {
                g_gemv(g,e,m[GK],g->norm,g->k);g_gemv(g,e,m[GV],g->norm,g->v);g_barrier(e);
                g_bind(e,g->qkv,0);g_bind(e,g->k,1);g_bind(e,g->v,2);g_bind(e,l.parameters,3,37056*4);g_bind(e,l.parameters,4,37312*4);
                g_bind(e,g->q,5);g_bind(e,l.keys,6);g_bind(e,l.values,7);[e setBytes:&s length:sizeof(s) atIndex:8];
                g_run(g,e,GHead,18,256);g_barrier(e);
                if(!g_try_fused_attention(g,e,l)){
                g_bind(e,g->q,0);g_bind(e,l.keys,1);g_bind(e,g->scores,2);[e setBytes:&s length:sizeof(s) atIndex:3];
                [e setComputePipelineState:g->ps[GScores]];
                g_profile_begin(g,e,GScores);
                [e dispatchThreadgroups:MTLSizeMake((position+4)/4,16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g->dispatches++;g_barrier(e);
                g_profile_end(g,e);
                g_bind(e,g->scores,0);g_bind(e,l.values,1);g_bind(e,g->qkv,2);g_bind(e,g->attention,3);[e setBytes:&s length:sizeof(s) atIndex:4];
                if(position<2048){g_run(g,e,GAttention,16,256);g_barrier(e);}
                else g_long_attention(g,e,g->scores,l.values,g->qkv,g->attention,1);
                }
            }
            g_gemv(g,e,m[GO],g->attention,g->out);g_barrier(e);
            g_bind(e,g->x,0);g_bind(e,g->out,1);g_bind(e,l.parameters,2,2048*4);g_bind(e,g->norm,3);[e setBytes:&s length:sizeof(s) atIndex:4];
            g_run(g,e,GAddNorm,1,256);g_barrier(e);
            // The next input norm is at offset zero; final layer uses the final norm.
            g_router(g,e,l,s,index==39?g->finalNorm:g->layers[index+1].parameters);
        }
        if(classifier){g_gemv(g,e,g->classifier,g->norm,g->logits);g_barrier(e);
            g_bind(e,g->logits,0);g_bind(e,g->argValues,1);g_bind(e,g->argIds,2);g_run(g,e,GArgBlocks,970,32);g_barrier(e);
            g_bind(e,g->argValues,0);g_bind(e,g->argIds,1);g_bind(e,g->nextToken,2);g_run(g,e,GArgFinal,1,32);}
        [e endEncoding];if(!q_complete(command,ms,error,capacity))return -1;
        *nextToken=classifier?static_cast<const int32_t*>(g->nextToken.contents)[0]:-1;
        if(classifier&&logits)memcpy(logits,g->logits.contents,248320*4);
        if(trace)memcpy(trace,g->trace.contents,40*2048*4);
        if(g->debugRouter&&g->position==g->debugPosition){char path[4096];snprintf(path,sizeof(path),"%s/gpu-routers.f32",std::getenv("DEEPINFER_GRAPH_DEBUG_DIR"));
            if(FILE *f=fopen(path,"wb")){fwrite(g->debugRouter.contents,4,40*264,f);fclose(f);}}
        if(g_profiling(g)) {
            NSData *data=[g->counters resolveCounterRange:NSMakeRange(0,g->profileCount*2)];
            if(data.length>=g->profileCount*2*sizeof(MTLCounterResultTimestamp)) {
                auto *samples=static_cast<const MTLCounterResultTimestamp*>(data.bytes);std::array<uint64_t,GPCount+7> totals{};
                for(uint32_t i=0;i<g->profileCount;i++)if(samples[i*2+1].timestamp>=samples[i*2].timestamp)
                    totals[g->profileTags[i]]+=samples[i*2+1].timestamp-samples[i*2].timestamp;
                fprintf(stderr,"GPU_TIMESTAMP_PROFILE position=32 ");for(int i=0;i<GPCount+7;i++)fprintf(stderr,"%d:%llu ",i,(unsigned long long)totals[i]);fprintf(stderr,"\n");
            }
        }
        *dispatches=g->dispatches;g->position++;return 0;
    }
}
}
