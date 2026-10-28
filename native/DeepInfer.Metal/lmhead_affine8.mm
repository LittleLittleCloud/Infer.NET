#include "lmhead_affine8_layout.h"

static NSString *lmhead_source = @R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void lmhead_tile4(device const uint2 *codes [[buffer(0)]],
 device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]],
 device const float *x [[buffer(3)]],device float *out [[buffer(4)]],
 constant uint &rows [[buffer(5)]],uint group [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint tile=group*4+sg,row=tile*4;
    if(row>=rows)return;
    float sums[4]={0,0,0,0};
    #pragma clang loop unroll_count(2)
    for(uint block=0;block<8;block++) {
        uint2 q[4];float scale[4],bias[4];
        #pragma clang loop unroll(full)
        for(uint r=0;r<4;r++) {
            q[r]=codes[(tile*8+block)*128+lane*4+r];
            uint p=(tile*32+block*4+lane/8)*4+r;
            scale[r]=as_type<float>(uint(scales[p])<<16);
            bias[r]=as_type<float>(uint(biases[p])<<16);
        }
        #pragma clang loop unroll(full)
        for(uint j=0;j<8;j++) {
            float input=x[block*256+lane*8+j];
            #pragma clang loop unroll(full)
            for(uint r=0;r<4;r++) {
                uint code=(j<4?q[r].x>>(j*8):q[r].y>>((j-4)*8))&255;
                sums[r]+=(float(code)*scale[r]+bias[r])*input;
            }
        }
    }
    #pragma clang loop unroll(full)
    for(uint r=0;r<4;r++) {
        float sum=simd_sum(sums[r]);
        if(lane==0&&row+r<rows)out[row+r]=sum;
    }
}
)metal";

struct LmHead {
    id<MTLBuffer> packed;
    id<MTLComputePipelineState> pipeline;
    std::shared_ptr<QResidencyLease> residency;
    int mode=0;
    bool configured=false;
};
static id<MTLComputePipelineState> lmhead_pipeline(id<MTLDevice> device,char *error,int capacity) {
    NSError *failure=nil;
    auto *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    auto library=[device newLibraryWithSource:lmhead_source options:options error:&failure];
    auto pipeline=library?[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"lmhead_tile4"] error:&failure]:nil;
    if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128) {
        error_text(error,capacity,failure.localizedDescription ?: @"LM head requires SIMD32/128 threads");return nil;
    }
    return pipeline;
}
static bool lmhead_prepare(LmHead &head,QContext &compute,const QMatrix &m,int mode,char *error,int capacity) {
    if(head.configured||mode<0||mode>1) {
        error_text(error,capacity,@"LM head selector must be off(0) or tile4(1), configured once before execution");return false;
    }
    if(mode==0){head.configured=true;return true;}
    if(!lmhead_affine8::geometry(m.experts,m.rows,m.columns,m.bits,m.group)) {
        error_text(error,capacity,@"tile4 LM head requires experts=1, N=248320, K=2048, affine8/group64");return false;
    }
    lmhead_affine8::Layout layout(m.rows);
    if(!m.packed||!m.scales||!m.biases||m.packed.length!=size_t(m.rows)*2048||
        m.scales.length!=size_t(m.rows)*64||m.biases.length!=m.scales.length||
        layout.bytes>compute.device.maxBufferLength) {
        error_text(error,capacity,@"Invalid LM head source buffers or device allocation limit");return false;
    }
    fprintf(stderr,"LMHEAD_PREFLIGHT extraPayloadBytes=%zu currentMetalBytes=%llu projectedMetalBytes=%llu repackScratchBytes=0\n",
        layout.bytes,(unsigned long long)compute.device.currentAllocatedSize,
        (unsigned long long)(compute.device.currentAllocatedSize+layout.bytes));
    auto pipeline=lmhead_pipeline(compute.device,error,capacity);if(!pipeline)return false;
    auto packed=[compute.device newBufferWithLength:layout.bytes options:MTLResourceStorageModeShared];
    if(!packed){error_text(error,capacity,@"Cannot allocate tile4 LM head");return false;}
    if(!lmhead_affine8::pack(m.rows,m.packed.contents,m.packed.length,m.scales.contents,m.scales.length,
        m.biases.contents,m.biases.length,packed.contents,packed.length)) {
        error_text(error,capacity,@"LM head packing rejected source sizes");return false;
    }
    std::shared_ptr<QResidencyLease> lease;
    if(compute.residency) {
        std::vector<id<MTLBuffer>> buffers{packed};
        if(!compute.residency->add(buffers)){error_text(error,capacity,@"LM head exceeds safe Metal residency budget");return false;}
        lease=std::make_shared<QResidencyLease>(compute.residency,std::move(buffers));
    }
    head.packed=packed;head.pipeline=pipeline;head.residency=std::move(lease);head.mode=mode;head.configured=true;
    return true;
}
static void lmhead_encode(const LmHead &head,id<MTLComputeCommandEncoder> e,id<MTLBuffer> x,
    id<MTLBuffer> out,uint32_t rows=lmhead_affine8::Rows) {
    lmhead_affine8::Layout layout(rows);
    [e setComputePipelineState:head.pipeline];
    [e setBuffer:head.packed offset:0 atIndex:0];
    [e setBuffer:head.packed offset:layout.scalesOffset atIndex:1];
    [e setBuffer:head.packed offset:layout.biasesOffset atIndex:2];
    [e setBuffer:x offset:0 atIndex:3];[e setBuffer:out offset:0 atIndex:4];
    [e setBytes:&rows length:4 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((rows+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
}
