// Environment validation only: this is not the Infer.NET CUDA inference backend.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); return 1; } } while (0)

__global__ void matmul(const float* a, const float* b, float* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float sum = 0;
    for (int j = 0; j < k; ++j) sum = fmaf(a[row*k+j], b[j*n+col], sum);
    c[row*n+col] = sum;
}

int main() {
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 8 || properties.minor != 9) {
        std::fprintf(stderr, "Expected SM89, found SM%d%d\n", properties.major, properties.minor);
        return 1;
    }
    constexpr int m = 37, n = 23, k = 53;
    std::vector<float> a(m*k), b(k*n), actual(m*n);
    for (int i = 0; i < m*k; ++i) a[i] = float(i%17-8) / 9;
    for (int i = 0; i < k*n; ++i) b[i] = float(i%13-6) / 7;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, actual.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice));
    matmul<<<dim3((n+15)/16, (m+15)/16), dim3(16,16)>>>(da, db, dc, m, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), dc, actual.size()*sizeof(float), cudaMemcpyDeviceToHost));
    double maxError = 0;
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        double expected = 0;
        for (int j = 0; j < k; ++j) expected += double(a[row*k+j])*double(b[j*n+col]);
        double error = std::abs(double(actual[row*n+col])-expected);
        if (!std::isfinite(actual[row*n+col]) || error > 1e-5) {
            std::fprintf(stderr, "Mismatch at (%d,%d): %.9g vs %.9g\n", row, col, actual[row*n+col], expected);
            return 1;
        }
        maxError = std::max(maxError, error);
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    std::printf("PASS CUDA environment: %s, SM%d%d, %d GEMM outputs, max_abs_error=%.9g\n",
        properties.name, properties.major, properties.minor, m*n, maxError);
    return 0;
}
