// Original Metal GEMV implementation. The transformer/tokenizer run in the .NET worker.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <algorithm>
#include <new>
#include <cmath>

static void error_text(char *target, int capacity, NSString *message) {
    if (target && capacity > 0) {
        const char *value = message.UTF8String ?: "Unknown Metal error";
        std::strncpy(target, value, (size_t)capacity - 1);
        target[capacity - 1] = 0;
    }
}

struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLComputePipelineState> bf16Pipeline;
};
struct Matrix {
    id<MTLBuffer> weights;
    id<MTLBuffer> input;
    id<MTLBuffer> output;
    uint32_t rows, columns;
    bool bf16;
};

static NSString *source = @R"metal(
#include <metal_stdlib>
using namespace metal;
// One threadgroup per row, two SIMD groups, FP32 accumulation. Handles non-multiple-of-64 widths.
kernel void gemv(device const float *matrix [[buffer(0)]],
                 device const float *x [[buffer(1)]],
                 device float *y [[buffer(2)]],
                 constant uint &columns [[buffer(3)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[2];
    float accum = 0.0f;
    for (uint column = tid; column < columns; column += 64)
        accum += matrix[row * columns + column] * x[column];
    float sum = simd_sum(accum);
    if (lane == 0) partials[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) y[row] = partials[0] + partials[1];
}
kernel void gemv_bf16(device const ushort *matrix [[buffer(0)]],
                 device const float *x [[buffer(1)]],
                 device float *y [[buffer(2)]],
                 constant uint &columns [[buffer(3)]],
                 uint row [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]],
                 uint group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[2];
    float accum = 0.0f;
    for (uint column = tid; column < columns; column += 64)
        accum += as_type<float>(uint(matrix[row * columns + column]) << 16) * x[column];
    float sum = simd_sum(accum);
    if (lane == 0) partials[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) y[row] = partials[0] + partials[1];
}
)metal";

extern "C" {
void *di_metal_create(char *error, int capacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { error_text(error, capacity, @"No Metal device."); return nullptr; }
        NSError *failure = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&failure];
        if (!library) { error_text(error, capacity, failure.localizedDescription); return nullptr; }
        id<MTLFunction> function = [library newFunctionWithName:@"gemv"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&failure];
        if (!pipeline || pipeline.threadExecutionWidth != 32 || pipeline.maxTotalThreadsPerThreadgroup < 64) {
            error_text(error, capacity, failure.localizedDescription ?: @"Kernel requires SIMD width 32 and 64 threads per group.");
            return nullptr;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) { error_text(error, capacity, @"Cannot create Metal command queue."); return nullptr; }
        id<MTLComputePipelineState> bf16 = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"gemv_bf16"] error:&failure];
        if (!bf16) { error_text(error, capacity, failure.localizedDescription); return nullptr; }
        return new Context{device, queue, pipeline, bf16};
    }
}
void di_metal_destroy(void *context) { delete static_cast<Context *>(context); }
void *di_metal_matrix_create(void *opaque, const void *weights, int rows, int columns, int bf16, char *error, int capacity) {
    @autoreleasepool {
        if (!opaque || !weights || rows <= 0 || columns <= 0) {
            error_text(error, capacity, @"Invalid GEMV shape or context."); return nullptr;
        }
        auto *context = static_cast<Context *>(opaque);
        id<MTLBuffer> matrix = [context->device newBufferWithBytes:weights length:(NSUInteger)rows * columns * (bf16 ? sizeof(uint16_t) : sizeof(float))
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> input = [context->device newBufferWithLength:(NSUInteger)columns * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [context->device newBufferWithLength:(NSUInteger)rows * sizeof(float) options:MTLResourceStorageModeShared];
        if (!matrix || !input || !output) { error_text(error, capacity, @"Metal buffer allocation failed."); return nullptr; }
        return new Matrix{matrix, input, output, (uint32_t)rows, (uint32_t)columns, bf16 != 0};
    }
}
void di_metal_matrix_destroy(void *matrix) { delete static_cast<Matrix *>(matrix); }
int di_metal_matvec(void *opaque, void *matrix, const float *input, float *output, double *gpu_ms, char *error, int capacity) {
    @autoreleasepool {
        if (!opaque || !matrix || !input || !output) { error_text(error, capacity, @"Null GEMV argument."); return -1; }
        auto *context = static_cast<Context *>(opaque);
        auto *m = static_cast<Matrix *>(matrix);
        std::memcpy(m->input.contents, input, (size_t)m->columns * sizeof(float));
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) { error_text(error, capacity, @"Cannot create Metal command encoder."); return -1; }
        [encoder setComputePipelineState:m->bf16 ? context->bf16Pipeline : context->pipeline];
        [encoder setBuffer:m->weights offset:0 atIndex:0];
        [encoder setBuffer:m->input offset:0 atIndex:1];
        [encoder setBuffer:m->output offset:0 atIndex:2];
        [encoder setBytes:&m->columns length:sizeof(uint32_t) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(m->rows,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            error_text(error, capacity, command.error.localizedDescription ?: @"Metal command failed."); return -1;
        }
        std::memcpy(output, m->output.contents, (size_t)m->rows * sizeof(float));
        if (gpu_ms) *gpu_ms = std::max(0.0, (command.GPUEndTime - command.GPUStartTime) * 1000.0);
        return 0;
    }
}
}
#include "quantized.mm"
