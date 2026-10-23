// Original resident Qwen3.6 affine8 prefix executor. Correctness-first, serial-token
// CUDA kernels; this is NOT the tensor-core batched prefill performance path.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
API int di_cuda_prefix_abi_version() { return 6; }
static void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Matrix {
    unsigned char *w = nullptr;
    unsigned short *s = nullptr, *b = nullptr;
    int experts = 0, rows = 0, cols = 0;
};
struct Layer {
    Matrix m[15];
    float *p = nullptr, *state = nullptr, *history = nullptr;
    void *keys = nullptr, *values = nullptr;
};
struct ChunkWorkspace {
    bool prepared=false;int scoreCapacity=0,partitions=0;
    float *parts=nullptr;
    float *x=nullptr,*norm=nullptr,*qkv=nullptr,*z=nullptr,*a=nullptr,*b=nullptr,*conv=nullptr,*attention=nullptr,
        *k=nullptr,*v=nullptr,*q=nullptr,*scores=nullptr,*out=nullptr,*router=nullptr,*prob=nullptr,
        *eg=nullptr,*eu=nullptr,*eo=nullptr,*sg=nullptr,*su=nullptr,*so=nullptr,*sr=nullptr;
    int *ids=nullptr,*tokens=nullptr;
};
struct Model {
    std::vector<void *> memory;
    Matrix embedding;
    Layer layers[20];
    int position = 0, capacity;
    bool kvBf16 = true, batchReadback = false, layerBatch=false, bf16Projections=false;
    ChunkWorkspace chunk;
    bool profileEnabled=false,denseReuse=false,densePacked=false,expertPacked=false;double profileTotals[9]{};
    int attentionPartitionThreshold=-1;
    // Experimental read-only route probe, enabled only before a native prefix run.
    int debugRouteLayer=-1,debugRoutePosition=-1,debugRoutes[8]{};
    float debugRouter[256]{},debugRouteProb[8]{};bool debugRouteCaptured=false;
    float eps, theta;
    float *x, *norm, *qkv, *z, *a, *b, *conv, *attention, *k, *v, *q, *scores, *out, *router, *prob,
        *eg, *eu, *eo, *sg, *su, *so, *sr, *batchHidden;
    int *ids;
    ~Model() {
        for (auto p : memory)
            cudaFree(p);
    }
    template <class T> T *alloc(size_t n) {
        T *p = nullptr;
        ck(cudaMalloc(&p, n * sizeof(T)));
        memory.push_back(p);
        ck(cudaMemset(p, 0, n * sizeof(T)));
        return p;
    }
};
__device__ float bf(unsigned short x) { return __uint_as_float(unsigned(x) << 16); }
__device__ float sig(float x) { return 1.f / (1.f + expf(-x)); }
__device__ float sil(float x) { return x * sig(x); }
__device__ float sum32(float x) {
    for (int n = 16; n; n /= 2)
        x += __shfl_xor_sync(0xffffffff, x, n);
    return x;
}
__device__ float blocksum(float x) {
    __shared__ float part[8];
    int t = threadIdx.x;
    x = sum32(x);
    if ((t & 31) == 0)
        part[t / 32] = x;
    __syncthreads();
    if (t == 0) {
        float v = 0;
        for (int i = 0; i < blockDim.x / 32; i++)
            v += part[i];
        part[0] = v;
    }
    __syncthreads();
    return part[0];
}
__device__ float weight(Matrix m, size_t r, int c) {
    size_t i = r * m.cols + c, p = i / 64;
    return float(m.w[i]) * bf(m.s[p]) + bf(m.b[p]);
}
__global__ void embed(Matrix m, int token, float *x) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < 2048)
        x[c] = weight(m, token, c);
}
__global__ void norm(float *x, const float *add, const float *w, float *y, float eps, int dim) {
    int t = threadIdx.x;
    float total = 0;
    for (int d = t; d < dim; d += blockDim.x) {
        float v = x[d] + (add ? add[d] : 0);
        if (add)
            x[d] = v;
        total += v * v;
    }
    float scale = rsqrtf(blocksum(total) / dim + eps);
    for (int d = t; d < dim; d += blockDim.x)
        y[d] = x[d] * scale * w[d];
}
__global__ void gemv(Matrix m, const float *x, float *y, const int *ids, int batch) {
    int lane = threadIdx.x & 31, r = blockIdx.x * 4 + threadIdx.x / 32, e = blockIdx.y;
    if (r >= m.rows)
        return;
    int expert = ids ? ids[e] : 0;
    size_t row = size_t(expert) * m.rows + r;
    float val = 0;
    // Match the Metal vectorized affine GEMV lane partition (8 values/lane).
    for (int base = lane * 8; base < m.cols; base += 256)
        for (int j = 0; j < 8; j++)
            val += weight(m, row, base + j) * x[(batch ? e * m.cols : 0) + base + j];
    val = sum32(val);
    if (lane == 0)
        y[e * m.rows + r] = val;
}
__global__ void activation(float *g, const float *u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        g[i] = sil(g[i]) * u[i];
}
__global__ void linear(float *raw, float *history, const float *p, float *c, float *a, float *b) {
    int t = threadIdx.x, h = blockIdx.x, d = h * 128 + t;
    const float *w = p + 4096;
    float v = sil(history[d] * w[d * 4] + history[8192 + d] * w[d * 4 + 1] +
                  history[16384 + d] * w[d * 4 + 2] + raw[d] * w[d * 4 + 3]);
    history[d] = history[8192 + d];
    history[8192 + d] = history[16384 + d];
    history[16384 + d] = raw[d];
    if (h < 32)
        v *= rsqrtf(blocksum(v * v) + 1e-6f) * (h < 16 ? rsqrtf(128.f) : 1.f);
    c[d] = v;
    if (h == 0 && t < 32) {
        float z = a[t] + p[36896 + t];
        a[t] = expf(-expf(p[36864 + t]) * (z > 20 ? z : logf(1.f + expf(z))));
        b[t] = sig(b[t]);
    }
}
__global__ void delta(float *state, const float *c, const float *a, const float *b, float *out) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 2 + threadIdx.x / 32, head = row / 128,
        kh = head / 2;
    float vals[4], pred = 0;
    for (int j = 0; j < 4; j++) {
        int d = lane + j * 32;
        vals[j] = state[row * 128 + d] * a[head];
        pred += vals[j] * c[2048 + kh * 128 + d];
    }
    pred = sum32(pred);
    float correction = (c[4096 + row] - pred) * b[head], result = 0;
    for (int j = 0; j < 4; j++) {
        int d = lane + j * 32;
        float v = vals[j] + c[2048 + kh * 128 + d] * correction;
        state[row * 128 + d] = v;
        result += v * c[kh * 128 + d];
    }
    result = sum32(result);
    if (lane == 0)
        out[row] = result;
}
__global__ void deltaNorm(float *x, const float *z, const float *w, float eps) {
    int t = threadIdx.x, d = blockIdx.x * 128 + t;
    float v = x[d], scale = rsqrtf(blocksum(v * v) / 128.f + eps);
    x[d] = v * scale * w[t] * sil(z[d]);
}
template <class KV>
__global__ void head(const float *raw, const float *k, const float *v, const float *p, float *q,
                     KV *keys, KV *values, int pos, float eps, float theta) {
    int t = threadIdx.x, h = blockIdx.x;
    bool query = h < 16;
    int index = query ? h : h - 16;
    const float *input = query ? raw + index * 512 : k + index * 256;
    const float *w = p + (query ? 37056 : 37312);
    float scale = rsqrtf(blocksum(input[t] * input[t]) / 256.f + eps);
    float val = input[t] * scale * w[t];
    if (t < 64) {
        int f = t % 32, partner = t < 32 ? t + 32 : t - 32;
        float angle = float(pos) / powf(theta, float(f) / 32.f),
              other = input[partner] * scale * w[partner];
        val = val * cosf(angle) + (t < 32 ? -other : other) * sinf(angle);
    }
    if (query)
        q[index * 256 + t] = val;
    else {
        keys[pos * 512 + index * 256 + t] = KV(val);
        values[pos * 512 + index * 256 + t] = KV(v[index * 256 + t]);
    }
}
template <class KV>
__global__ void scores(const float *q, const KV *k, float *s, int pos, int capacity) {
    int t = blockIdx.x * 4 + threadIdx.x / 32, h = blockIdx.y, lane = threadIdx.x & 31;
    if (t > pos)
        return;
    float v = 0;
    for (int d = lane; d < 256; d += 32)
        v += q[h * 256 + d] * float(k[t * 512 + (h / 8) * 256 + d]);
    v = sum32(v);
    if (lane == 0)
        s[h * capacity + t] = v / 16.f;
}
template <class KV>
__global__ void attend(float *s, const KV *v, const float *raw, float *out, int pos,
                       int capacity) {
    float *prob = s + blockIdx.x * capacity;
    __shared__ float parts[8];
    int h = blockIdx.x, t = threadIdx.x;
    float best = -INFINITY;
    for (int i = t; i <= pos; i += 256)
        best = fmaxf(best, s[h * capacity + i]);
    for (int n = 16; n; n /= 2)
        best = fmaxf(best, __shfl_xor_sync(0xffffffff, best, n));
    if ((t & 31) == 0)
        parts[t / 32] = best;
    __syncthreads();
    if (t == 0) {
        float m = parts[0];
        for (int i = 1; i < 8; i++)
            m = fmaxf(m, parts[i]);
        parts[0] = m;
    }
    __syncthreads();
    best = parts[0];
    float total = 0;
    for (int i = t; i <= pos; i += 256) {
        prob[i] = expf(s[h * capacity + i] - best);
        total += prob[i];
    }
    float den = blocksum(total);
    for (int i = t; i <= pos; i += 256)
        prob[i] /= den;
    __syncthreads();
    float result = 0;
    for (int i = 0; i <= pos; i++)
        result += prob[i] * float(v[i * 512 + (h / 8) * 256 + t]);
    out[h * 256 + t] = result * sig(raw[h * 512 + 256 + t]);
}
__global__ void top8(const float *logits, int *ids, float *prob) {
    if (threadIdx.x)
        return;
    float chosen[8];
    for (int n = 0; n < 8; n++) {
        float best = -INFINITY;
        int selected = 0;
        for (int i = 0; i < 256; i++) {
            bool used = false;
            for (int j = 0; j < n; j++)
                used |= ids[j] == i;
            if (!used && logits[i] > best) {
                best = logits[i];
                selected = i;
            }
        }
        ids[n] = selected;
        chosen[n] = best;
    }
    float total = 0;
    for (int n = 0; n < 8; n++) {
        prob[n] = expf(chosen[n] - chosen[0]);
        total += prob[n];
    }
    for (int n = 0; n < 8; n++)
        prob[n] /= total;
}
__global__ void merge(float *x, const float *eo, const float *so, const float *sr,
                      const float *prob, float *hidden = nullptr) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < 2048) {
        float v = so[d] * sig(sr[0]);
        for (int i = 0; i < 8; i++)
            v += eo[i * 2048 + d] * prob[i];
        x[d] += v;
        if (hidden) hidden[d] = x[d];
    }
}
static void mv(Matrix m, float *x, float *y, int *ids = nullptr, int count = 1,
               bool batch = false) {
    gemv<<<dim3((m.rows + 3) / 4, count), 128>>>(m, x, y, ids, batch);
}
#include "prefill_batch.cuh"
#include "prefill_chunk.cuh"
#include "prefill_attention.cuh"
#include "prefill_executor.cuh"
static void read(std::ifstream &f, void *p, size_t n) {
    if (!f.read((char *)p, n))
        throw std::runtime_error("Truncated prefix bundle");
}
template <class T> static T *upload(Model *g, std::ifstream &f, size_t n) {
    T *p = g->alloc<T>(n);
    std::vector<char> b(8 << 20);
    size_t offset = 0, bytes = n * sizeof(T);
    while (offset < bytes) {
        size_t take = std::min(b.size(), bytes - offset);
        read(f, b.data(), take);
        ck(cudaMemcpy((char *)p + offset, b.data(), take, cudaMemcpyHostToDevice));
        offset += take;
    }
    return p;
}
static Matrix matrix(Model *g, std::ifstream &f, int experts, int rows, int cols) {
    unsigned shape[3];
    read(f, shape, 12);
    if (shape[0] != experts || shape[1] != rows || shape[2] != cols)
        throw std::runtime_error("Prefix matrix geometry mismatch");
    Matrix m;
    m.experts = experts;
    m.rows = rows;
    m.cols = cols;
    if (!experts)
        return m;
    size_t n = size_t(experts) * rows * cols;
    m.w = upload<unsigned char>(g, f, n);
    m.s = upload<unsigned short>(g, f, n / 64);
    m.b = upload<unsigned short>(g, f, n / 64);
    return m;
}
static void err(char *error, int cap, const std::exception &e) {
    if (error && cap > 0)
        snprintf(error, cap, "%s", e.what());
}
API void *di_cuda_prefix_create(const char *path, int capacity, int kvBytes, char *error, int cap) {
    Model *g = nullptr;
    try {
        if ((kvBytes != 2 && kvBytes != 4) || capacity < 1 || capacity > 262144)
            throw std::runtime_error("CUDA prefix capacity exceeds the pinned 262144-token model window");
        cudaDeviceProp prop;
        ck(cudaGetDeviceProperties(&prop, 0));
        if (prop.major != 8 || prop.minor != 9)
            throw std::runtime_error("Expected CUDA SM89 device");
        g = new Model();
        g->capacity = capacity;
        g->kvBf16 = kvBytes == 2;
        std::ifstream f(path, std::ios::binary);
        char header[48];
        read(f, header, 48);
        if (memcmp(header, "DIPFX001e06a74e6236a60c8367e1a3214e83d8b61b637b0", 48))
            throw std::runtime_error("Prefix format/revision mismatch");
        read(f, &g->eps, 4);
        read(f, &g->theta, 4);
        if (g->eps != 1e-6f || g->theta != 10000000.f)
            throw std::runtime_error("Prefix norm/RoPE mismatch");
        g->embedding = matrix(g, f, 1, 248320, 2048);
        for (int l = 0; l < 20; l++) {
            auto &v = g->layers[l];
            bool lin = (l + 1) % 4 != 0;
            for (int i = 0; i < 15; i++) {
                int e = 1, r = 0, c = 2048;
                switch (i) {
                case 0:
                    r = 8192;
                    break;
                case 1:
                case 2:
                    r = lin ? 0 : 512;
                    break;
                case 3:
                    r = lin ? 4096 : 0;
                    break;
                case 4:
                case 5:
                    r = lin ? 32 : 0;
                    break;
                case 6:
                    r = 2048;
                    c = 4096;
                    break;
                case 7:
                    r = 256;
                    break;
                case 8:
                case 9:
                    e = 256;
                    r = 512;
                    break;
                case 10:
                    e = 256;
                    r = 2048;
                    c = 512;
                    break;
                case 11:
                case 12:
                    r = 512;
                    break;
                case 13:
                    r = 2048;
                    c = 512;
                    break;
                case 14:
                    r = 1;
                    break;
                }
                v.m[i] = matrix(g, f, r ? e : 0, r, r ? c : 0);
            }
            v.p = upload<float>(g, f, 37568);
            if (lin) {
                v.state = g->alloc<float>(32 * 128 * 128);
                v.history = g->alloc<float>(3 * 8192);
            } else {
                v.keys = g->alloc<unsigned char>(size_t(capacity) * 512 * kvBytes);
                v.values = g->alloc<unsigned char>(size_t(capacity) * 512 * kvBytes);
            }
        }
        if (f.peek() != EOF)
            throw std::runtime_error("Trailing prefix bundle data");
#define BUF(name, n) g->name = g->alloc<float>(n)
        BUF(x, 2048);
        BUF(batchHidden, 32 * 2048);
        BUF(norm, 2048);
        BUF(qkv, 8192);
        BUF(z, 4096);
        BUF(a, 32);
        BUF(b, 32);
        BUF(conv, 8192);
        BUF(attention, 4096);
        BUF(k, 512);
        BUF(v, 512);
        BUF(q, 4096);
        BUF(scores, 16 * capacity);
        BUF(out, 2048);
        BUF(router, 256);
        BUF(prob, 8);
        BUF(eg, 4096);
        BUF(eu, 4096);
        BUF(eo, 16384);
        BUF(sg, 512);
        BUF(su, 512);
        BUF(so, 2048);
        BUF(sr, 1);
        g->ids = g->alloc<int>(8);
#undef BUF
        ck(cudaDeviceSynchronize());
        return g;
    } catch (const std::exception &e) {
        delete g;
        err(error, cap, e);
        return nullptr;
    }
}
API void di_cuda_prefix_destroy(void *ptr) { delete static_cast<Model *>(ptr); }
// Grow only between requests; model weights remain resident. Allocate transactionally.
API int di_cuda_prefix_reserve(void *ptr, int capacity, char *error, int cap) {
    std::vector<void *> allocated;
    try {
        auto *g = static_cast<Model *>(ptr);
        if (!g || capacity < 1 || capacity > 262144 || g->position != 0)
            throw std::runtime_error("Reset CUDA session before reserving 1..262144 tokens");
        if (capacity <= g->capacity) return 0;
        capacity = ((capacity + 2047) / 2048) * 2048;
        ck(cudaDeviceSynchronize());
        for (int i=0; i<11; ++i) {
            void *p=nullptr; size_t bytes=i==10?size_t(16)*capacity*4:size_t(capacity)*512*(g->kvBf16?2:4);
            ck(cudaMalloc(&p,bytes)); allocated.push_back(p); ck(cudaMemset(p,0,bytes));
        }
        int at=0;
        auto replace=[&](void *old,void *next){ auto found=std::find(g->memory.begin(),g->memory.end(),old); if(found!=g->memory.end()) *found=next; cudaFree(old); };
        for(int l=3;l<20;l+=4) {
            replace(g->layers[l].keys,allocated[at]); g->layers[l].keys=allocated[at++];
            replace(g->layers[l].values,allocated[at]); g->layers[l].values=allocated[at++];
        }
        replace(g->scores,allocated[10]); g->scores=static_cast<float *>(allocated[10]); g->capacity=capacity;
        allocated.clear(); return 0;
    } catch (const std::exception &e) { for(auto p:allocated) cudaFree(p); err(error,cap,e); return -1; }
}
API int di_cuda_prefix_reset(void *ptr, char *error, int cap) {
    try {
        auto *g = static_cast<Model *>(ptr);
        if (!g) throw std::runtime_error("Invalid prefix session");
        ck(cudaDeviceSynchronize());
        for (int i = 0; i < 20; ++i) {
            auto &l = g->layers[i];
            if ((i + 1) % 4) {
                ck(cudaMemset(l.state, 0, 32 * 128 * 128 * sizeof(float)));
                ck(cudaMemset(l.history, 0, 3 * 8192 * sizeof(float)));
            } else {
                ck(cudaMemset(l.keys, 0, size_t(g->capacity) * 512 * (g->kvBf16 ? 2 : 4)));
                ck(cudaMemset(l.values, 0, size_t(g->capacity) * 512 * (g->kvBf16 ? 2 : 4)));
            }
        }
        ck(cudaDeviceSynchronize());
        g->position = 0;
        g->debugRouteCaptured=false;
        return 0;
    } catch (const std::exception &e) {
        if (error && cap > 0) snprintf(error, cap, "%s", e.what());
        return -1;
    }
}
API int di_cuda_prefix_readback_mode(void *ptr, int batched) {
    auto *g=static_cast<Model*>(ptr);
    if(!g || (batched!=0 && batched!=1)) return -1;
    g->batchReadback=batched!=0;return 0;
}
API int di_cuda_prefix_execution_mode(void *ptr,int mode) {
    auto *g=static_cast<Model*>(ptr);if(!g||mode<0||mode>5||g->position!=0)return -1;
    g->layerBatch=mode!=0;g->bf16Projections=mode==2;g->denseReuse=mode>=3;g->densePacked=mode>=4;g->expertPacked=mode==5;return 0;
}
// Experimental, disabled by default. Zero forces every chunk for numerical tests.
API int di_cuda_prefix_partition_threshold(void *ptr,int threshold) {
    auto *g=static_cast<Model*>(ptr);
    if(!g||g->position!=0||threshold < -1||threshold>262144)return -1;
    g->attentionPartitionThreshold=threshold;return 0;
}
API int di_cuda_prefix_route_probe(void *ptr,int layer,int position) {
    auto *g=static_cast<Model*>(ptr);
    if(!g||g->position!=0||layer<0||layer>=20||position<0||position>=g->capacity)return -1;
    g->debugRouteLayer=layer;g->debugRoutePosition=position;g->debugRouteCaptured=false;return 0;
}
API int di_cuda_prefix_route_read(void *ptr,int *ids,int idCount,float *logits,int logitCount,float *prob,int probCount) {
    auto *g=static_cast<Model*>(ptr);
    if(!g||!g->debugRouteCaptured||!ids||!logits||!prob||idCount!=8||logitCount!=256||probCount!=8)return -1;
    std::copy(g->debugRoutes,g->debugRoutes+8,ids);
    std::copy(g->debugRouter,g->debugRouter+256,logits);
    std::copy(g->debugRouteProb,g->debugRouteProb+8,prob);return 0;
}
// Diagnostic phase timing; disabled in every production session by default.
API int di_cuda_prefix_profile_mode(void *ptr,int enabled) {
    auto *g=static_cast<Model*>(ptr);if(!g||(enabled!=0&&enabled!=1)||g->position!=0)return -1;
    g->profileEnabled=enabled!=0;std::fill(g->profileTotals,g->profileTotals+9,0.0);return 0;
}
API int di_cuda_prefix_profile_read(void *ptr,double *values,int count) {
    auto *g=static_cast<Model*>(ptr);if(!g||!values||count!=9)return -1;
    std::copy(g->profileTotals,g->profileTotals+9,values);return 0;
}
API int di_cuda_prefix_run(void *ptr, const int *tokens, int count, int position, float *hidden,
                           float *trace, char *error, int cap) {
    try {
        auto *g = static_cast<Model *>(ptr);
        if (!g || !tokens || !hidden || count < 1 || count > 32 || position != g->position ||
            position + count > g->capacity)
            throw std::runtime_error("Invalid prefix sequence");
        for (int i = 0; i < count; i++)
            if (tokens[i] < 0 || tokens[i] >= 248320)
                throw std::runtime_error("Invalid token");
        if(g->layerBatch){run_chunk(g,tokens,count,hidden,trace);return 0;}
        for (int i = 0; i < count; i++) {
            int pos = g->position;
            embed<<<16, 128>>>(g->embedding, tokens[i], g->x);
            for (int l = 0; l < 20; l++) {
                auto &v = g->layers[l];
                auto *m = v.m;
                norm<<<1, 256>>>(g->x, nullptr, v.p, g->norm, g->eps, 2048);
                mv(m[0], g->norm, g->qkv);
                if ((l + 1) % 4) {
                    mv(m[3], g->norm, g->z);
                    mv(m[4], g->norm, g->a);
                    mv(m[5], g->norm, g->b);
                    linear<<<64, 128>>>(g->qkv, v.history, v.p, g->conv, g->a, g->b);
                    delta<<<2048, 64>>>(v.state, g->conv, g->a, g->b, g->attention);
                    deltaNorm<<<32, 128>>>(g->attention, g->z, v.p + 36928, g->eps);
                } else {
                    mv(m[1], g->norm, g->k);
                    mv(m[2], g->norm, g->v);
                    if (g->kvBf16) {
                        head<<<18, 256>>>(g->qkv, g->k, g->v, v.p, g->q, (__nv_bfloat16 *)v.keys,
                                          (__nv_bfloat16 *)v.values, pos, g->eps, g->theta);
                        scores<<<dim3((pos + 4) / 4, 16), 128>>>(g->q, (__nv_bfloat16 *)v.keys,
                                                                 g->scores, pos, g->capacity);
                        attend<<<16, 256>>>(g->scores, (__nv_bfloat16 *)v.values, g->qkv,
                                            g->attention, pos, g->capacity);
                    } else {
                        head<<<18, 256>>>(g->qkv, g->k, g->v, v.p, g->q, (float *)v.keys,
                                          (float *)v.values, pos, g->eps, g->theta);
                        scores<<<dim3((pos + 4) / 4, 16), 128>>>(g->q, (float *)v.keys, g->scores,
                                                                 pos, g->capacity);
                        attend<<<16, 256>>>(g->scores, (float *)v.values, g->qkv, g->attention, pos,
                                            g->capacity);
                    }
                }
                mv(m[6], g->attention, g->out);
                norm<<<1, 256>>>(g->x, g->out, v.p + 2048, g->norm, g->eps, 2048);
                mv(m[7], g->norm, g->router);
                mv(m[11], g->norm, g->sg);
                mv(m[12], g->norm, g->su);
                mv(m[14], g->norm, g->sr);
                top8<<<1, 32>>>(g->router, g->ids, g->prob);
                activation<<<4, 128>>>(g->sg, g->su, 512);
                mv(m[8], g->norm, g->eg, g->ids, 8);
                mv(m[9], g->norm, g->eu, g->ids, 8);
                activation<<<32, 128>>>(g->eg, g->eu, 4096);
                mv(m[10], g->eg, g->eo, g->ids, 8, true);
                mv(m[13], g->sg, g->so);
                merge<<<16, 128>>>(g->x, g->eo, g->so, g->sr, g->prob,
                    g->batchReadback && l == 19 ? g->batchHidden + i * 2048 : nullptr);
                if (trace)
                    ck(cudaMemcpy(trace + (i * 20 + l) * 2048, g->x, 2048 * 4,
                                  cudaMemcpyDeviceToHost));
            }
            if (!g->batchReadback) ck(cudaMemcpy(hidden + i * 2048, g->x, 2048 * 4, cudaMemcpyDeviceToHost));
            g->position++;
        }
        if (g->batchReadback) ck(cudaMemcpy(hidden, g->batchHidden, size_t(count) * 2048 * 4, cudaMemcpyDeviceToHost));
        ck(cudaGetLastError());
        return 0;
    } catch (const std::exception &e) {
        err(error, cap, e);
        return -1;
    }
}
API int di_cuda_prefix_state(void *ptr, int layer, int kind, int position, void *dest, int bytes,
                             char *error, int cap) {
    try {
        auto *g = static_cast<Model *>(ptr);
        if (!g || layer < 0 || layer >= 20 || kind < 0 || kind > 1 || position != g->position ||
            !dest)
            throw std::runtime_error("Invalid state identity");
        auto &l = g->layers[layer];
        bool lin = (layer + 1) % 4;
        int expected =
            lin ? (kind ? 3 * 8192 * 4 : 32 * 128 * 128 * 4) : position * 512 * (g->kvBf16 ? 2 : 4);
        if (bytes != expected)
            throw std::runtime_error("Invalid state size");
        void *src = lin ? (void *)(kind ? l.history : l.state) : (void *)(kind ? l.values : l.keys);
        ck(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception &e) {
        err(error, cap, e);
        return -1;
    }
}
// Export only the newly computed causal KV slice. No recurrence snapshot here.
API int di_cuda_prefix_kv_range(void *ptr,int layer,int kind,int position,int count,void *dest,int bytes,char *error,int cap) {
    try {
        auto *g=static_cast<Model*>(ptr);
        if(!g||layer<0||layer>=20||(layer+1)%4!=0||kind<0||kind>1||!dest||position<0||count<1||count>32||
           position>g->position-count||bytes!=count*512*(g->kvBf16?2:4))throw std::runtime_error("Invalid KV export range");
        auto &l=g->layers[layer];auto *src=static_cast<unsigned char*>(kind?l.values:l.keys);
        ck(cudaMemcpy(dest,src+size_t(position)*512*(g->kvBf16?2:4),bytes,cudaMemcpyDeviceToHost));return 0;
    }catch(const std::exception &e){err(error,cap,e);return -1;}
}
// Deterministic operator checks independent of model files. Outputs [GEMV, Delta]
// max absolute errors against scalar double accumulation, checked by the caller.
API int di_cuda_prefix_selftest(double *errors, char *error, int cap) {
    try {
        Model g;
        Matrix m;
        m.experts = 2;
        m.rows = 13;
        m.cols = 256;
        std::vector<unsigned char> w(2 * 13 * 256);
        std::vector<unsigned short> s(2 * 13 * 4, 0x3b80), b(2 * 13 * 4, 0xbe80);
        std::vector<float> x(256), actual(26);
        for (size_t i = 0; i < w.size(); i++)
            w[i] = (i * 17 + 3) % 256;
        for (int i = 0; i < 256; i++)
            x[i] = float((i * 11) % 23 - 11) / 23;
        m.w = g.alloc<unsigned char>(w.size());
        m.s = g.alloc<unsigned short>(s.size());
        m.b = g.alloc<unsigned short>(b.size());
        ck(cudaMemcpy(m.w, w.data(), w.size(), cudaMemcpyHostToDevice));
        ck(cudaMemcpy(m.s, s.data(), s.size() * 2, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(m.b, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
        auto *input = g.alloc<float>(256);
        auto *out = g.alloc<float>(26);
        auto *ids = g.alloc<int>(2);
        int chosen[2] = {1, 0};
        ck(cudaMemcpy(input, x.data(), 1024, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(ids, chosen, 8, cudaMemcpyHostToDevice));
        mv(m, input, out, ids, 2);
        ck(cudaMemcpy(actual.data(), out, 104, cudaMemcpyDeviceToHost));
        errors[0] = 0;
        for (int e = 0; e < 2; e++)
            for (int r = 0; r < 13; r++) {
                double expected = 0;
                for (int c = 0; c < 256; c++)
                    expected += (double(w[(chosen[e] * 13 + r) * 256 + c]) / 256 - .25) * x[c];
                errors[0] = std::max(errors[0], fabs(expected - actual[e * 13 + r]));
            }
        std::vector<float> state(32 * 128 * 128), conv(8192), a(32, .97f), beta(32, .3f),
            result(4096);
        for (size_t i = 0; i < state.size(); i++)
            state[i] = float(int(i % 19) - 9) / 100;
        for (size_t i = 0; i < conv.size(); i++)
            conv[i] = float(int(i % 17) - 8) / 100;
        auto *ds = g.alloc<float>(state.size());
        auto *dc = g.alloc<float>(8192);
        auto *da = g.alloc<float>(32);
        auto *db = g.alloc<float>(32);
        auto *dy = g.alloc<float>(4096);
        ck(cudaMemcpy(ds, state.data(), state.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(dc, conv.data(), 8192 * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(da, a.data(), 128, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(db, beta.data(), 128, cudaMemcpyHostToDevice));
        delta<<<2048, 64>>>(ds, dc, da, db, dy);
        ck(cudaMemcpy(result.data(), dy, 4096 * 4, cudaMemcpyDeviceToHost));
        std::vector<float> updated(state.size());
        ck(cudaMemcpy(updated.data(), ds, state.size() * 4, cudaMemcpyDeviceToHost));
        errors[1] = 0;
        for (int row = 0; row < 4096; row++) {
            int head = row / 128, kh = head / 2;
            double pred = 0;
            for (int d = 0; d < 128; d++)
                pred += double(state[row * 128 + d]) * a[head] * conv[2048 + kh * 128 + d];
            double correction = (conv[4096 + row] - pred) * beta[head], expected = 0;
            for (int d = 0; d < 128; d++) {
                double v =
                    double(state[row * 128 + d]) * a[head] + conv[2048 + kh * 128 + d] * correction;
                errors[1] = std::max(errors[1], fabs(v - updated[row * 128 + d]));
                expected += v * conv[kh * 128 + d];
            }
            errors[1] = std::max(errors[1], fabs(expected - result[row]));
        }
        return 0;
    } catch (const std::exception &e) {
        err(error, cap, e);
        return -1;
    }
}
// Analytic long attention test; no model weights. Exercises positions beyond the old 2K boundary.
API int di_cuda_prefix_attention_test(int length, int kvBytes, double *maxError, char *error, int cap) {
    try {
        if(length<1||length>16384||(kvBytes!=2&&kvBytes!=4)) throw std::runtime_error("Invalid attention check geometry");
        Model g; std::vector<float> score(16*length), values(length*512), raw(8192,0), result(4096);
        for(int h=0;h<16;h++)for(int i=0;i<length;i++)score[h*length+i]=float((i*7+h*3)%37-18)/8;
        for(size_t i=0;i<values.size();i++)values[i]=float(int(i%29)-14)/16;
        auto *ds=g.alloc<float>(score.size()),*dv=g.alloc<float>(values.size()),*dr=g.alloc<float>(raw.size()),*out=g.alloc<float>(4096);
        ck(cudaMemcpy(ds,score.data(),score.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
        if(kvBytes==4){ck(cudaMemcpy(dv,values.data(),values.size()*4,cudaMemcpyHostToDevice));attend<<<16,256>>>(ds,dv,dr,out,length-1,length);}
        else {std::vector<__nv_bfloat16> v(values.size());for(size_t i=0;i<v.size();i++)v[i]=__float2bfloat16(values[i]);ck(cudaMemcpy(dv,v.data(),v.size()*2,cudaMemcpyHostToDevice));attend<<<16,256>>>(ds,(__nv_bfloat16*)dv,dr,out,length-1,length);}
        ck(cudaMemcpy(result.data(),out,4096*4,cudaMemcpyDeviceToHost));*maxError=0;
        for(int h=0;h<16;h++){
            double best=-INFINITY,total=0;for(int i=0;i<length;i++)best=std::max(best,double(score[h*length+i]));
            std::vector<double> p(length);for(int i=0;i<length;i++){p[i]=std::exp(double(score[h*length+i])-best);total+=p[i];}
            for(int d=0;d<256;d++){double expected=0;for(int i=0;i<length;i++)expected+=p[i]*values[i*512+(h/8)*256+d];expected*=.5/total;
                *maxError=std::max(*maxError,std::abs(expected-result[h*256+d]));}
        }
        return 0;
    } catch(const std::exception &e){err(error,cap,e);return -1;}
}
