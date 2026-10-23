// Unified-memory transport microbenchmark; matches the model's shared MTLBuffers.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> data, errors;
static id<MTLComputePipelineState> fillPipeline, checkPipeline;
static char errorMessage[1024];
#define API extern "C"
static const char *source=R"(
#include <metal_stdlib>
using namespace metal;
struct Args {uint count,seed;};
kernel void fill(device uint *p [[buffer(0)]],constant Args &a [[buffer(1)]],uint i [[thread_position_in_grid]]){if(i<a.count)p[i]=(i*1664525u+1013904223u)^a.seed;}
kernel void check(device uint *p [[buffer(0)]],constant Args &a [[buffer(1)]],device atomic_uint *bad [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<a.count && p[i]!=((i*1664525u+1013904223u)^a.seed))atomic_fetch_add_explicit(bad,1u,memory_order_relaxed);}
)";
API const char *st_error(){return errorMessage;}
static int run(id<MTLComputePipelineState> pipeline,size_t bytes,uint32_t seed){
 @autoreleasepool {
 if(bytes>data.length || bytes%4)return -1;
 struct {uint32_t count,seed;} args={(uint32_t)(bytes/4),seed};
 auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];
 [enc setComputePipelineState:pipeline];[enc setBuffer:data offset:0 atIndex:0];
 [enc setBytes:&args length:sizeof(args) atIndex:1];[enc setBuffer:errors offset:0 atIndex:2];
 [enc dispatchThreads:MTLSizeMake(bytes/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
 [cb commit];[cb waitUntilCompleted];
 if(cb.status!=MTLCommandBufferStatusCompleted){snprintf(errorMessage,sizeof(errorMessage),"%s",cb.error.description.UTF8String);return -1;}return 0;
 }
}
API int st_open(size_t bytes,size_t chunk){
 @autoreleasepool {
 device=MTLCreateSystemDefaultDevice();queue=[device newCommandQueue];
 data=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
 errors=[device newBufferWithLength:4 options:MTLResourceStorageModeShared];
 NSError *e=nil;auto lib=[device newLibraryWithSource:[NSString stringWithUTF8String:source] options:nil error:&e];
 if(!lib){snprintf(errorMessage,sizeof(errorMessage),"%s",e.description.UTF8String);return -1;}
 fillPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"fill"] error:&e];
 checkPipeline=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"check"] error:&e];
 return data && errors && queue && fillPipeline && checkPipeline?0:-1;
 }
}
API int st_prepare(size_t bytes,uint32_t seed,int isSource){if(bytes>data.length)return -1;if(isSource)return run(fillPipeline,bytes,seed);memset(data.contents,0,bytes);return 0;}
API void *st_pointer(size_t offset){return (unsigned char*)data.contents+offset;}
API int st_read(size_t offset,size_t bytes){return offset<=data.length && bytes<=data.length-offset?0:-1;}
API int st_write(size_t offset,size_t bytes){return offset<=data.length && bytes<=data.length-offset?0:-1;}
API int st_validate(size_t bytes,uint32_t seed){*(uint32_t*)errors.contents=0;if(run(checkPipeline,bytes,seed))return -1;auto bad=*(uint32_t*)errors.contents;if(bad){snprintf(errorMessage,sizeof(errorMessage),"%u mismatched uint32 words",bad);return -1;}return 0;}
API void st_close(){data=nil;errors=nil;fillPipeline=nil;checkPipeline=nil;queue=nil;device=nil;}
