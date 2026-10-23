// Original affine 5/8-bit selected-expert GEMV and gated-delta recurrence.
// No MLX, ggml or other inference library is linked or invoked.
#include <cstdlib>
#include "residency.mm"
struct QContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> gemv, delta;
    id<MTLComputePipelineState> tiled5, tiled8, small5, small8;
    id<MTLComputePipelineState> gateUp5, gateUp8;
    id<MTLComputePipelineState> wide5,wide8;
    uint32_t rowTile;
    bool generic;
    std::shared_ptr<QResidency> residency;
};
struct QMatrix {
    id<MTLBuffer> packed, scales, biases, input, output;
    uint32_t experts, rows, columns, bits, group;
    std::shared_ptr<QResidencyLease> residency;
};
struct DeltaState {
    id<MTLBuffer> state, q, k, v, decay, beta, output;
};
struct QuantJob {
    QMatrix *matrix;
    const float *input;
    float *output;
    const uint32_t *experts;
    int32_t count, batched;
};
// Diagnostic parity experiment: match CUDA's explicit XOR reduction order.
// The production default retains Metal's built-in reduction.
static NSString *q_math_source(NSString *source) {
    const char *mode=getenv("DEEPINFER_METAL_MATH");
    if(!mode||strcmp(mode,"ordered")!=0)return source;
    return [@R"metal(
#include <metal_stdlib>
using namespace metal;
template<typename T> inline T di_ordered_sum(T value) {
    for(uint offset=16;offset>0;offset/=2)
        value += simd_shuffle_xor(value,offset);
    return value;
}
#define simd_sum di_ordered_sum
)metal" stringByAppendingString:source];
}
static NSString *quant_source = @R"metal(
#include <metal_stdlib>
using namespace metal;
struct QShape { uint experts, rows, columns, bits, group, count, batched; };
kernel void quant_gemv(device const uint *w [[buffer(0)]],
 device const ushort *scales [[buffer(1)]], device const ushort *biases [[buffer(2)]],
 device const float *x [[buffer(3)]], device float *out [[buffer(4)]],
 constant QShape &s [[buffer(5)]], constant uint *experts [[buffer(6)]],
 uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint slot=g/s.rows, row=g%s.rows, expert=experts[slot];
    ulong base=(ulong(expert)*s.rows+row)*(s.columns*s.bits/32);
    ulong params=(ulong(expert)*s.rows+row)*(s.columns/s.group);
    float sum=0; uint mask=(1u<<s.bits)-1;
    for(uint c=tid;c<s.columns;c+=64) {
        uint bit=c*s.bits, at=bit/32, shift=bit%32;
        uint q=w[base+at]>>shift;
        if(shift+s.bits>32) q |= w[base+at+1]<<(32-shift);
        uint p=c/s.group;
        float scale=as_type<float>(uint(scales[params+p])<<16);
        float bias=as_type<float>(uint(biases[params+p])<<16);
        float value=float(q&mask)*scale+bias;
        sum += value*x[(s.batched?slot*s.columns:0)+c];
    }
    threadgroup float partial[2];
    sum=simd_sum(sum); if(lane==0)partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)out[slot*s.rows+row]=partial[0]+partial[1];
}
// Affine groups contain 64 values. The vector path gives each lane eight
// adjacent values; the fallback handles column counts below/not divisible by 256.
template<uint Bits>
inline uint quant_code(device const uint *w, uint word, uint shift) {
    uint q=w[word]>>shift;
    if(shift+Bits>32)q |= w[word+1]<<(32-shift);
    return q&((1u<<Bits)-1);
}
// Normalize a lane's consecutive eight codes once. Their subsequent bit
// positions are compile-time constants, avoiding eight variable cross-word
// extraction branches for every affine group iteration.
inline uint quant_align_low(uint low,uint high,uint shift) {
    return (low>>shift)|(shift?high<<(32-shift):0u);
}
template<uint Bits>
inline uint quant_octet(uint low,uint high,uint index) {
    if(Bits==5){
        if(index<6)return (low>>(index*5))&31;
        if(index==6)return ((low>>30)|(high<<2))&31;
        return (high>>3)&31;
    }
    return (index<4?low>>(index*8):high>>((index-4)*8))&255;
}
template<uint Bits>
inline float quant_dot(device const uint *w, device const ushort *scales,
 device const ushort *biases, device const float *x, uint groups, uint lane) {
#ifdef VECTOR_QUANT
    if(groups%4==0) {
    float sum=0;
    #pragma clang loop unroll_count(2)
    for(uint g=0;g<groups;g+=4) {
        uint local=lane*8,bit=local*Bits,word=bit/32,shift=bit%32;
        uint low=w[word],high=w[word+1];
        low=quant_align_low(low,high,shift);high>>=shift;
        float scale=as_type<float>(uint(scales[g+lane/8])<<16);
        float bias=as_type<float>(uint(biases[g+lane/8])<<16);
        #pragma clang loop unroll(full)
        for(uint j=0;j<8;j++) {
            uint code=quant_octet<Bits>(low,high,j);
            sum+=(float(code)*scale+bias)*x[g*64+local+j];
        }
        w+=Bits*8;
    }
    return simd_sum(sum);
    }
#endif
    uint bit=lane*Bits, word=bit>>5, shift=bit&31;
    float sum=0;
    #pragma clang loop unroll_count(4)
    for(uint g=0;g<groups;g++) {
        float scale=as_type<float>(uint(scales[g])<<16);
        float bias=as_type<float>(uint(biases[g])<<16);
        uint q0=quant_code<Bits>(w,word,shift),q1=quant_code<Bits>(w,word+Bits,shift);
        sum += (float(q0)*scale+bias)*x[g*64+lane];
        sum += (float(q1)*scale+bias)*x[g*64+lane+32];
        w+=Bits*2;
    }
    return simd_sum(sum);
}
template<uint Bits,uint RowsPerGroup>
inline void quant_rows(device const uint *w,device const ushort *scales,
 device const ushort *biases,device const float *x,device float *out,
 constant QShape &s,constant uint *experts,uint2 group,uint lane,uint sg) {
    uint row=group.x*RowsPerGroup+sg,slot=group.y;
    if(row>=s.rows)return; // Entire SIMD group takes the same branch; no threadgroup barrier.
    uint groups=s.columns>>6;
    ulong params=(ulong(experts[slot])*s.rows+row)*groups;
    float sum=quant_dot<Bits>(w+params*(Bits*2),scales+params,biases+params,
        x+(s.batched?slot*s.columns:0),groups,lane);
    if(lane==0)out[slot*s.rows+row]=sum;
}
#define QUANT_ROWS(NAME,BITS,ROWS) \
kernel void NAME(device const uint *w [[buffer(0)]], \
 device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]], \
 device const float *x [[buffer(3)]],device float *out [[buffer(4)]], \
 constant QShape &s [[buffer(5)]],constant uint *experts [[buffer(6)]], \
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]], \
 uint sg [[simdgroup_index_in_threadgroup]]) { \
 quant_rows<BITS,ROWS>(w,scales,biases,x,out,s,experts,group,lane,sg); }
QUANT_ROWS(quant_rows5,5,4)
QUANT_ROWS(quant_rows8,8,4)
QUANT_ROWS(quant_small5,5,1)
QUANT_ROWS(quant_small8,8,1)
QUANT_ROWS(quant_wide5,5,8)
QUANT_ROWS(quant_wide8,8,8)

// Selected expert gate/up share input loads. Activation stays on the GPU and
// writes contiguous [selected expert, intermediate] values for the down GEMV.
template<uint Bits>
inline void expert_gate_up(device const uint *gw,device const ushort *gs,device const ushort *gb,
 device const uint *uw,device const ushort *us,device const ushort *ub,
 device const float *x,device float *out,constant QShape &s,constant uint *experts,
 uint2 group,uint lane,uint sg) {
    uint row=group.x*4+sg,slot=group.y;
    if(row>=s.rows)return;
    uint groups=s.columns>>6,bit=lane*Bits,word=bit>>5,shift=bit&31;
    ulong params=(ulong(experts[slot])*s.rows+row)*groups;
#ifdef VECTOR_QUANT
    float gate=quant_dot<Bits>(gw+params*(Bits*2),gs+params,gb+params,x,groups,lane);
    float up=quant_dot<Bits>(uw+params*(Bits*2),us+params,ub+params,x,groups,lane);
    if(lane==0)out[slot*s.rows+row]=(gate*(1.0f/(1.0f+exp(-gate))))*up;
#else
    gw+=params*(Bits*2);uw+=params*(Bits*2);
    gs+=params;gb+=params;us+=params;ub+=params;
    float gate=0,up=0;
    #pragma clang loop unroll_count(4)
    for(uint g=0;g<groups;g++) {
        float x0=x[g*64+lane],x1=x[g*64+lane+32];
        float sgain=as_type<float>(uint(gs[g])<<16),gbias=as_type<float>(uint(gb[g])<<16);
        float ugain=as_type<float>(uint(us[g])<<16),ubias=as_type<float>(uint(ub[g])<<16);
        gate+=(float(quant_code<Bits>(gw,word,shift))*sgain+gbias)*x0;
        gate+=(float(quant_code<Bits>(gw,word+Bits,shift))*sgain+gbias)*x1;
        up+=(float(quant_code<Bits>(uw,word,shift))*ugain+ubias)*x0;
        up+=(float(quant_code<Bits>(uw,word+Bits,shift))*ugain+ubias)*x1;
        gw+=Bits*2;uw+=Bits*2;
    }
    gate=simd_sum(gate);up=simd_sum(up);
    if(lane==0)out[slot*s.rows+row]=(gate*(1.0f/(1.0f+exp(-gate))))*up;
#endif
}
#define EXPERT_GATE_UP(NAME,BITS) \
kernel void NAME(device const uint *gw [[buffer(0)]],device const ushort *gs [[buffer(1)]], \
 device const ushort *gb [[buffer(2)]],device const uint *uw [[buffer(3)]], \
 device const ushort *us [[buffer(4)]],device const ushort *ub [[buffer(5)]], \
 device const float *x [[buffer(6)]],device float *out [[buffer(7)]], \
 constant QShape &s [[buffer(8)]],constant uint *experts [[buffer(9)]], \
 uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]], \
 uint sg [[simdgroup_index_in_threadgroup]]) {expert_gate_up<BITS>(gw,gs,gb,uw,us,ub,x,out,s,experts,group,lane,sg);}
EXPERT_GATE_UP(expert_gate_up5,5)
EXPERT_GATE_UP(expert_gate_up8,8)
// One SIMD group per (value head, value channel), four state elements per lane.
kernel void delta_step(device float *state [[buffer(0)]],
 device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],
 device const float *v [[buffer(3)]], device const float *decay [[buffer(4)]],
 device const float *beta [[buffer(5)]], device float *out [[buffer(6)]],
 uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
 uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row=group*2+sg, head=row/128, kh=head/2;
    float values[4]; float prediction=0;
    for(uint j=0;j<4;j++) {
        uint d=lane+32*j;
        values[j]=state[row*128+d]*decay[head];
        prediction += values[j]*k[kh*128+d];
    }
    prediction=simd_sum(prediction);
    float correction=(v[row]-prediction)*beta[head]; float result=0;
    for(uint j=0;j<4;j++) {
        uint d=lane+32*j;
        float value=values[j]+k[kh*128+d]*correction;
        state[row*128+d]=value;
        result += value*q[kh*128+d];
    }
    result=simd_sum(result); if(lane==0)out[row]=result;
}
)metal";
static bool q_complete(id<MTLCommandBuffer> command, double *ms, char *error, int capacity) {
    auto started=std::chrono::steady_clock::now();
    [command commit];auto committed=std::chrono::steady_clock::now();
    [command waitUntilCompleted];auto completed=std::chrono::steady_clock::now();
    double wall=std::chrono::duration<double,std::milli>(completed-started).count();
    if(wall>100)fprintf(stderr,"METAL_WAIT commitMs=%.3f waitMs=%.3f gpuMs=%.3f\n",
        std::chrono::duration<double,std::milli>(committed-started).count(),
        std::chrono::duration<double,std::milli>(completed-committed).count(),
        std::max(0.0,(command.GPUEndTime-command.GPUStartTime)*1000.0));
    if(command.status!=MTLCommandBufferStatusCompleted) {
        error_text(error,capacity,command.error.localizedDescription ?: @"Quantized Metal command failed"); return false;
    }
    if(ms)*ms=std::max(0.0,(command.GPUEndTime-command.GPUStartTime)*1000.0);
    return true;
}
static void q_encode(QContext *c,id<MTLComputeCommandEncoder> e,QMatrix *m,
 id<MTLBuffer> input,const uint32_t *experts,int count,bool batched) {
    struct {uint32_t experts,rows,columns,bits,group,count,batched;} shape{m->experts,m->rows,m->columns,m->bits,m->group,(uint32_t)count,(uint32_t)batched};
    bool small=m->rows<128;
    [e setComputePipelineState:c->generic?c->gemv:(m->bits==5?(small?c->small5:c->rowTile==8?c->wide5:c->tiled5):(small?c->small8:c->rowTile==8?c->wide8:c->tiled8))];
    [e setBuffer:m->packed offset:0 atIndex:0];[e setBuffer:m->scales offset:0 atIndex:1];[e setBuffer:m->biases offset:0 atIndex:2];
    [e setBuffer:input offset:0 atIndex:3];[e setBuffer:m->output offset:0 atIndex:4];
    [e setBytes:&shape length:sizeof(shape) atIndex:5];[e setBytes:experts length:count*4 atIndex:6];
    if(c->generic)[e dispatchThreadgroups:MTLSizeMake((NSUInteger)m->rows*count,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
    else [e dispatchThreadgroups:MTLSizeMake((m->rows+(small?0:c->rowTile-1))/(small?1:c->rowTile),count,1) threadsPerThreadgroup:MTLSizeMake(small?32:c->rowTile*32,1,1)];
}
extern "C" {
void *di_quant_create_configured(int resident,int refresh,char *error,int capacity) {
    if((resident!=0&&resident!=1)||refresh<0||refresh>60000){error_text(error,capacity,@"Invalid residency configuration");return nullptr;}
    @autoreleasepool {
        id<MTLDevice> d=MTLCreateSystemDefaultDevice(); if(!d){error_text(error,capacity,@"No Metal device");return nullptr;}
        NSError *failure=nil; MTLCompileOptions *options=[MTLCompileOptions new]; options.mathMode=MTLMathModeSafe;
        const char *vector=std::getenv("DEEPINFER_METAL_VECTOR_QUANT");
        NSString *source=!vector||strcmp(vector,"0")!=0?[@"#define VECTOR_QUANT\n" stringByAppendingString:quant_source]:quant_source;
        id<MTLLibrary> library=[d newLibraryWithSource:q_math_source(source) options:options error:&failure];
        if(!library){error_text(error,capacity,failure.localizedDescription);return nullptr;}
        id<MTLComputePipelineState> gemv=[d newComputePipelineStateWithFunction:[library newFunctionWithName:@"quant_gemv"] error:&failure];
        id<MTLComputePipelineState> delta=[d newComputePipelineStateWithFunction:[library newFunctionWithName:@"delta_step"] error:&failure];
        auto pipeline=[&](NSString *name){return [d newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&failure];};
        auto tiled5=pipeline(@"quant_rows5"),tiled8=pipeline(@"quant_rows8");
        auto small5=pipeline(@"quant_small5"),small8=pipeline(@"quant_small8");
        auto gateUp5=pipeline(@"expert_gate_up5"),gateUp8=pipeline(@"expert_gate_up8");
        auto wide5=pipeline(@"quant_wide5"),wide8=pipeline(@"quant_wide8");
        id<MTLCommandQueue> queue=[d newCommandQueue];
        if(!gemv||!delta||!tiled5||!tiled8||!small5||!small8||!gateUp5||!gateUp8||!wide5||!wide8||!queue||gemv.threadExecutionWidth!=32||delta.threadExecutionWidth!=32||
           tiled5.threadExecutionWidth!=32||tiled8.threadExecutionWidth!=32||
           tiled5.maxTotalThreadsPerThreadgroup<128||tiled8.maxTotalThreadsPerThreadgroup<128||
           gateUp5.threadExecutionWidth!=32||gateUp8.threadExecutionWidth!=32||
           gateUp5.maxTotalThreadsPerThreadgroup<128||gateUp8.maxTotalThreadsPerThreadgroup<128||
           wide5.maxTotalThreadsPerThreadgroup<256||wide8.maxTotalThreadsPerThreadgroup<256||
           gemv.maxTotalThreadsPerThreadgroup<64||delta.maxTotalThreadsPerThreadgroup<64) {
            error_text(error,capacity,failure.localizedDescription ?: @"Quant kernels require 64 threads/SIMD32");return nullptr;
        }
        const char *generic=std::getenv("DEEPINFER_METAL_GENERIC");
        const char *tile=std::getenv("DEEPINFER_METAL_ROW_TILE");uint32_t rowTile=tile&&strcmp(tile,"8")==0?8:4;
        std::shared_ptr<QResidency> residency;
        if(resident) {
            uint64_t reserve=512ULL<<20;
            uint64_t limit=std::min<uint64_t>(d.recommendedMaxWorkingSetSize,NSProcessInfo.processInfo.physicalMemory*4/5);
            if(limit<=reserve){error_text(error,capacity,@"Insufficient Metal residency budget");return nullptr;}
            residency=std::make_shared<QResidency>(d,refresh,limit-reserve);
            if(!residency->valid()){error_text(error,capacity,residency->error());return nullptr;}
        }
        return new QContext{d,queue,gemv,delta,tiled5,tiled8,small5,small8,gateUp5,gateUp8,wide5,wide8,rowTile,generic&&strcmp(generic,"1")==0,residency};
    }
}
// Legacy experiments may still configure creation through inherited native environment.
void *di_quant_create(char *error,int capacity) {
    const char *mode=std::getenv("DEEPINFER_METAL_RESIDENCY");
    const char *value=std::getenv("DEEPINFER_METAL_RESIDENCY_REFRESH_MS");
    return di_quant_create_configured(mode&&strcmp(mode,"1")==0,value?atoi(value):1000,error,capacity);
}
uint64_t di_quant_residency_bytes(void *p) {
    auto *c=static_cast<QContext*>(p);return c&&c->residency?c->residency->allocated():0;
}
void di_quant_residency_prepare(void *p) {
    if(p){auto *c=static_cast<QContext*>(p);if(c->residency)c->residency->start();}
}
void di_quant_destroy(void *p){delete static_cast<QContext*>(p);}
void *di_quant_matrix_create(void *ctx,int experts,int rows,int columns,int bits,int group,char *error,int capacity) {
    @autoreleasepool {
        if(!ctx||experts<1||experts>256||rows<1||columns<1||(bits!=5&&bits!=8)||group!=64||columns%64) {
            error_text(error,capacity,@"Unsupported quantized matrix shape");return nullptr;
        }
        auto *c=static_cast<QContext*>(ctx);
        auto buffer=[&](NSUInteger bytes){return [c->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];};
        QMatrix *m=new QMatrix{buffer((NSUInteger)experts*rows*columns*bits/8),
            buffer((NSUInteger)experts*rows*(columns/group)*2),buffer((NSUInteger)experts*rows*(columns/group)*2),
            buffer((NSUInteger)columns*8*4),buffer((NSUInteger)rows*8*4),
            (uint32_t)experts,(uint32_t)rows,(uint32_t)columns,(uint32_t)bits,(uint32_t)group};
        if(!m->packed||!m->scales||!m->biases||!m->input||!m->output) {
            delete m;error_text(error,capacity,@"Quantized buffer allocation failed");return nullptr;
        }
        if(c->residency) {
            std::vector<id<MTLBuffer>> buffers{m->packed,m->scales,m->biases};
            if(!c->residency->add(buffers)){delete m;error_text(error,capacity,@"Model weights exceed safe Metal residency budget");return nullptr;}
            m->residency=std::make_shared<QResidencyLease>(c->residency,std::move(buffers));
        }
        return m;
    }
}
void *di_quant_matrix_buffer(void *matrix,int index){
    if(!matrix)return nullptr;auto *m=static_cast<QMatrix*>(matrix);
    return index==0?m->packed.contents:index==1?m->scales.contents:index==2?m->biases.contents:nullptr;
}
void di_quant_matrix_destroy(void *matrix){delete static_cast<QMatrix*>(matrix);}
int di_quant_batch(void *ctx,const QuantJob *jobs,int jobCount,double *ms,char *error,int capacity) {
    @autoreleasepool {
        if(!ctx||!jobs||jobCount<1||jobCount>8){error_text(error,capacity,@"Invalid GEMV batch");return -1;}
        for(int j=0;j<jobCount;j++) {
            const auto &job=jobs[j];auto *m=job.matrix;
            if(!m||!job.input||!job.output||!job.experts||job.count<1||job.count>8) {
                error_text(error,capacity,@"Invalid quantized GEMV arguments");return -1;
            }
            for(int i=0;i<job.count;i++)if(job.experts[i]>=m->experts){error_text(error,capacity,@"Expert ID out of bounds");return -1;}
            for(int i=0;i<j;i++)if(jobs[i].matrix==m){error_text(error,capacity,@"Batch matrices must be distinct");return -1;}
        }
        auto *c=static_cast<QContext*>(ctx);
        id<MTLCommandBuffer> command=[c->queue commandBuffer];id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
        if(!command||!e){error_text(error,capacity,@"Cannot create quantized encoder");return -1;}
        for(int j=0;j<jobCount;j++) {
            const auto &job=jobs[j];auto *m=job.matrix;
            memcpy(m->input.contents,job.input,(size_t)m->columns*(job.batched?job.count:1)*4);
            q_encode(c,e,m,m->input,job.experts,job.count,job.batched!=0);
        }
        [e endEncoding];
        if(!q_complete(command,ms,error,capacity))return -1;
        for(int j=0;j<jobCount;j++)memcpy(jobs[j].output,jobs[j].matrix->output.contents,(size_t)jobs[j].matrix->rows*jobs[j].count*4);
        return 0;
    }
}
int di_quant_matvec(void *ctx,void *matrix,const float *x,float *out,const uint32_t *experts,int count,int batched,double *ms,char *error,int capacity) {
    QuantJob job{static_cast<QMatrix*>(matrix),x,out,experts,count,batched};
    return di_quant_batch(ctx,&job,1,ms,error,capacity);
}
int di_quant_experts(void *ctx,QMatrix *gate,QMatrix *up,QMatrix *down,QMatrix *shared,
 const float *input,const float *sharedInput,const uint32_t *experts,int count,
 float *expertOutput,float *sharedOutput,double *ms,char *error,int capacity) {
    @autoreleasepool {
        if(!ctx||!gate||!up||!down||!shared||!input||!sharedInput||!experts||!expertOutput||!sharedOutput||count<1||count>8||
           gate==up||gate==down||gate==shared||up==down||up==shared||down==shared||
           gate->rows!=up->rows||gate->columns!=up->columns||gate->experts!=up->experts||gate->bits!=up->bits||
           down->experts!=gate->experts||down->columns!=gate->rows||down->rows!=gate->columns||
           shared->experts!=1||shared->rows!=down->rows) {
            error_text(error,capacity,@"Incompatible fused expert matrices/arguments");return -1;
        }
        for(int i=0;i<count;i++)if(experts[i]>=gate->experts){error_text(error,capacity,@"Expert ID out of bounds");return -1;}
        auto *c=static_cast<QContext*>(ctx);
        memcpy(gate->input.contents,input,gate->columns*4);
        memcpy(shared->input.contents,sharedInput,shared->columns*4);
        id<MTLCommandBuffer> command=[c->queue commandBuffer];id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
        if(!command||!e){error_text(error,capacity,@"Cannot create expert encoder");return -1;}
        [e setComputePipelineState:gate->bits==5?c->gateUp5:c->gateUp8];
        [e setBuffer:gate->packed offset:0 atIndex:0];[e setBuffer:gate->scales offset:0 atIndex:1];[e setBuffer:gate->biases offset:0 atIndex:2];
        [e setBuffer:up->packed offset:0 atIndex:3];[e setBuffer:up->scales offset:0 atIndex:4];[e setBuffer:up->biases offset:0 atIndex:5];
        [e setBuffer:gate->input offset:0 atIndex:6];[e setBuffer:gate->output offset:0 atIndex:7];
        struct {uint32_t experts,rows,columns,bits,group,count,batched;} shape{gate->experts,gate->rows,gate->columns,gate->bits,gate->group,(uint32_t)count,0};
        [e setBytes:&shape length:sizeof(shape) atIndex:8];[e setBytes:experts length:count*4 atIndex:9];
        [e dispatchThreadgroups:MTLSizeMake((gate->rows+3)/4,count,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        q_encode(c,e,down,gate->output,experts,count,true);
        const uint32_t zero=0;q_encode(c,e,shared,shared->input,&zero,1,false);
        [e endEncoding];
        if(!q_complete(command,ms,error,capacity))return -1;
        memcpy(expertOutput,down->output.contents,(size_t)down->rows*count*4);
        memcpy(sharedOutput,shared->output.contents,shared->rows*4);
        return 0;
    }
}
int di_quant_embedding(void *matrix,int token,float *out){
    if(!matrix||!out)return -1;auto *m=static_cast<QMatrix*>(matrix);
    if(token<0||(uint32_t)token>=m->rows||m->experts!=1)return -1;
    auto *w=static_cast<const uint32_t*>(m->packed.contents)+(size_t)token*m->columns*m->bits/32;
    auto *s=static_cast<const uint16_t*>(m->scales.contents)+(size_t)token*m->columns/m->group;
    auto *b=static_cast<const uint16_t*>(m->biases.contents)+(size_t)token*m->columns/m->group;
    for(uint32_t i=0;i<m->columns;i++) {
        uint32_t bit=i*m->bits,at=bit/32,shift=bit%32,q=w[at]>>shift;
        if(shift+m->bits>32)q|=w[at+1]<<(32-shift);
        uint32_t scaleBits=uint32_t(s[i/m->group])<<16,biasBits=uint32_t(b[i/m->group])<<16;float scale,bias;
        memcpy(&scale,&scaleBits,4);memcpy(&bias,&biasBits,4);
        out[i]=float(q&((1u<<m->bits)-1))*scale+bias;
    }return 0;
}
void *di_delta_create(void *ctx){
    @autoreleasepool {
        if(!ctx)return nullptr;auto *c=static_cast<QContext*>(ctx);
        auto buffer=[&](NSUInteger n){return [c->device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];};
        auto *s=new DeltaState{buffer(32*128*128),buffer(16*128),buffer(16*128),buffer(32*128),buffer(32),buffer(32),buffer(32*128)};
        if(!s->state||!s->q||!s->k||!s->v||!s->decay||!s->beta||!s->output){delete s;return nullptr;}
        memset(s->state.contents,0,32*128*128*4);return s;
    }
}
void di_delta_destroy(void *state){delete static_cast<DeltaState*>(state);}
int di_delta_step(void *ctx,void *state,const float *q,const float *k,const float *v,const float *decay,const float *beta,float *out,double *ms,char *error,int capacity){
    @autoreleasepool {
        if(!ctx||!state||!q||!k||!v||!decay||!beta||!out){error_text(error,capacity,@"Null delta argument");return -1;}
        auto *c=static_cast<QContext*>(ctx);auto *s=static_cast<DeltaState*>(state);
        memcpy(s->q.contents,q,2048*4);memcpy(s->k.contents,k,2048*4);memcpy(s->v.contents,v,4096*4);
        memcpy(s->decay.contents,decay,32*4);memcpy(s->beta.contents,beta,32*4);
        id<MTLCommandBuffer> command=[c->queue commandBuffer];id<MTLComputeCommandEncoder> e=[command computeCommandEncoder];
        if(!command||!e){error_text(error,capacity,@"Cannot create delta encoder");return -1;}
        [e setComputePipelineState:c->delta];
        [e setBuffer:s->state offset:0 atIndex:0];[e setBuffer:s->q offset:0 atIndex:1];[e setBuffer:s->k offset:0 atIndex:2];
        [e setBuffer:s->v offset:0 atIndex:3];[e setBuffer:s->decay offset:0 atIndex:4];[e setBuffer:s->beta offset:0 atIndex:5];[e setBuffer:s->output offset:0 atIndex:6];
        [e dispatchThreadgroups:MTLSizeMake(2048,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];
        if(!q_complete(command,ms,error,capacity))return -1;memcpy(out,s->output.contents,4096*4);return 0;
    }
}
}
#include "gpu_graph.mm"
#include "gpu_prefill.mm"
