// Independent output and timestamp coverage check for encoder-boundary profiling.
#include "bridge.mm"
int main(){@autoreleasepool {
    auto device=MTLCreateSystemDefaultDevice();
    if(![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]){
        fprintf(stderr,"SKIP: encoder timestamp sampling unavailable\n");return 0;
    }
    NSError *error=nil;
    auto library=[device newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal; kernel void add(device uint *x [[buffer(0)]], constant uint &v [[buffer(1)]], uint i [[thread_position_in_grid]]) { x[i]+=v; }" options:nil error:&error];
    auto pipeline=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add"] error:&error];
    if(!pipeline){fprintf(stderr,"%s\n",error.localizedDescription.UTF8String);return 1;}
    for(bool active:{false,true}) {
        GModel g{};g.compute.device=device;g.compute.queue=[device newCommandQueue];g.counterMode=2;g.profileActive=active;
        for(id<MTLCounterSet> set in device.counterSets)if([set.name isEqualToString:MTLCommonCounterSetTimestamp]){
            auto d=[MTLCounterSampleBufferDescriptor new];d.counterSet=set;d.storageMode=MTLStorageModeShared;d.sampleCount=2048;
            g.counters=[device newCounterSampleBufferWithDescriptor:d error:&error];
        }
        if(!g.counters)return 1;
        auto x=[device newBufferWithLength:64*sizeof(uint32_t) options:MTLResourceStorageModeShared];memset(x.contents,0,x.length);
        auto cb=[g.compute.queue commandBuffer];id<MTLComputeCommandEncoder> e=nil,first=nil;
        for(uint32_t v:{1u,2u}){
            e=g_stage_encoder(&g,cb,e,v==1?GPStageEmbedding:GPStageArgmax);if(v==1)first=e;
            [e setComputePipelineState:pipeline];[e setBuffer:x offset:0 atIndex:0];[e setBytes:&v length:4 atIndex:1];
            [e dispatchThreadgroups:MTLSizeMake(2,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];g.dispatches++;g_barrier(e);
        }
        if(active)g.profileCoveredDispatches+=g.dispatches-g.profileDispatchStart;
        [e endEncoding];double ms;char message[1024]{};if(!q_complete(cb,&ms,message,sizeof(message)))return 1;
        for(int i=0;i<64;i++)if(static_cast<uint32_t*>(x.contents)[i]!=3)return 1;
        if(active){
            if(first==e||g.profileCount!=2||g.profileCoveredDispatches!=2)return 1;
            NSData *data=[g.counters resolveCounterRange:NSMakeRange(0,4)];if(data.length!=4*sizeof(MTLCounterResultTimestamp))return 1;
            auto *samples=static_cast<const MTLCounterResultTimestamp*>(data.bytes);
            for(int i=0;i<2;i++)if(samples[2*i].timestamp==MTLCounterErrorValue||samples[2*i+1].timestamp==MTLCounterErrorValue||samples[2*i+1].timestamp<=samples[2*i].timestamp)return 1;
        }else if(first!=e||g.profileCount||g.profileCoveredDispatches)return 1;
    }
    puts("PASS: identical outputs with profiling off/on; two encoder regions, valid timestamps and exact dispatch coverage");
}}
