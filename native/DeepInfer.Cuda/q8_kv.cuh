#pragma once
#include <cuda_fp16.h>

template<class T> __device__ float kv_read(const T *p, size_t index) { return float(p[index]); }
__device__ float kv_read(const unsigned char *p, size_t index) {
    const unsigned char *block = p + (index / 32) * 34;
    return __half2float(*reinterpret_cast<const __half *>(block)) * float(static_cast<signed char>(block[2 + index % 32]));
}
template<class T> __device__ void kv_write(T *p, size_t index, float value, int *) { p[index] = T(value); }
// All 32 lanes write one contiguous block after sharing its absolute maximum.
__device__ void kv_write(unsigned char *p, size_t index, float value, int *error) {
    float maximum = fabsf(value);
    int invalid = !isfinite(value);
    for (int n = 16; n; n /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, n));
        invalid |= __shfl_xor_sync(0xffffffff, invalid, n);
    }
    __half h = __float2half_rn(maximum / 127.f);
    float scale = __half2float(h);
    invalid |= !isfinite(scale) || (maximum != 0 && scale == 0);
    if (invalid) { if (error) atomicExch(error, 1); return; }
    unsigned char *block = p + (index / 32) * 34;
    if ((index & 31) == 0) *reinterpret_cast<__half *>(block) = h;
    int q = scale == 0 ? 0 : max(-127, min(127, __float2int_rn(value / scale)));
    block[2 + index % 32] = static_cast<unsigned char>(static_cast<signed char>(q));
}
