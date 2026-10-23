#pragma once
// Layer-major chunk helpers. Preserve FP32 reduction order until BF16 is explicitly selected.

__global__ void chunk_norm(float *x, const float *add, const float *w, float *y, float eps, int dim) {
    int token=blockIdx.x; x+=token*dim; if(add)add+=token*dim; y+=token*dim;

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

__global__ void chunk_delta_norm(float *x, const float *z, const float *w, float eps) {
    x+=blockIdx.y*4096;z+=blockIdx.y*4096;

    int t = threadIdx.x, d = blockIdx.x * 128 + t;
    float v = x[d], scale = rsqrtf(blocksum(v * v) / 128.f + eps);
    x[d] = v * scale * w[t] * sil(z[d]);
}

template <class KV>
__global__ void chunk_head(const float *raw, const float *k, const float *v, const float *p, float *q,
                     KV *keys, KV *values, int pos, float eps, float theta) {
    int seq=blockIdx.y;raw+=seq*8192;k+=seq*512;v+=seq*512;q+=seq*4096;pos+=seq;

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
__global__ void chunk_scores(const float *q, const KV *k, float *s, int pos, int capacity) {
    int seq=blockIdx.z;q+=seq*4096;s+=size_t(seq)*16*capacity;pos+=seq;

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
__global__ void chunk_attend(float *s, const KV *v, const float *raw, float *out, int pos,
                       int capacity) {
    int seq=blockIdx.y;s+=size_t(seq)*16*capacity;raw+=seq*8192;out+=seq*4096;pos+=seq;

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

__global__ void chunk_top8(const float *logits, int *ids, float *prob) {
    logits+=blockIdx.x*256;ids+=blockIdx.x*8;prob+=blockIdx.x*8;

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

__global__ void chunk_merge(float *x, const float *eo, const float *so, const float *sr,
                      const float *prob, float *hidden = nullptr) {
    int seq=blockIdx.y;x+=seq*2048;eo+=seq*16384;so+=seq*2048;sr+=seq;prob+=seq*8;if(hidden)hidden+=seq*2048;

    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < 2048) {
        float v = so[d] * sig(sr[0]);
        for (int i = 0; i < 8; i++)
            v += eo[i * 2048 + d] * prob[i];
        x[d] += v;
        if (hidden) hidden[d] = x[d];
    }
}
__global__ void chunk_embed(Matrix m,const int *ids,float *x) {
    int d=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y;
    if(d<2048)x[t*2048+d]=weight(m,ids[t],d);
}
__global__ void chunk_gemv(Matrix m,const float *x,float *y,const int *ids,int routedInput) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,e=blockIdx.y,t=blockIdx.z,n=gridDim.y;
    if(row>=m.rows)return;
    int expert=ids?ids[t*n+e]:0;size_t wr=size_t(expert)*m.rows+row;
    const float *input=x+(size_t(t)*(routedInput?n:1)+(routedInput?e:0))*m.cols;
    float val=0;
    for(int base=lane*8;base<m.cols;base+=256)for(int j=0;j<8;j++)val+=weight(m,wr,base+j)*input[base+j];
    val=sum32(val);if(lane==0)y[(size_t(t)*n+e)*m.rows+row]=val;
}
// cols must be divisible by256: eight bytes share one affine group and are aligned.
__global__ void chunk_gemv_packed(Matrix m,const float *x,float *y,const int *ids,int routedInput) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,e=blockIdx.y,t=blockIdx.z,n=gridDim.y;
    if(row>=m.rows)return;
    int expert=ids?ids[t*n+e]:0;size_t wr=size_t(expert)*m.rows+row;
    const float *input=x+(size_t(t)*(routedInput?n:1)+(routedInput?e:0))*m.cols;
    float val=0;
    for(int base=lane*8;base<m.cols;base+=256) {
        size_t index=wr*m.cols+base;
        uint2 bytes=*reinterpret_cast<const uint2*>(m.w+index);
        float scale=bf(m.s[index/64]),bias=bf(m.b[index/64]);
        for(int j=0;j<8;j++) {
            float w=float(((j<4?bytes.x:bytes.y)>>(8*(j&3)))&255)*scale+bias;
            val+=w*input[base+j];
        }
    }
    val=sum32(val);if(lane==0)y[(size_t(t)*n+e)*m.rows+row]=val;
}
// Gate and up projections use the same routed input. One launch and one input
// load stream serve both matrices while retaining their separate FP32 sums.
__global__ void chunk_expert_gate_up_packed(Matrix gate,Matrix up,const float *x,
                                            float *gateOut,float *upOut,const int *ids) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,e=blockIdx.y,t=blockIdx.z,n=gridDim.y;
    if(row>=gate.rows)return;
    int expert=ids[t*n+e];
    size_t gateRow=size_t(expert)*gate.rows+row,upRow=size_t(expert)*up.rows+row;
    const float *input=x+size_t(t)*gate.cols;
    float gateSum=0,upSum=0;
    for(int base=lane*8;base<gate.cols;base+=256) {
        size_t gi=gateRow*gate.cols+base,ui=upRow*up.cols+base;
        uint2 gw=*reinterpret_cast<const uint2*>(gate.w+gi);
        uint2 uw=*reinterpret_cast<const uint2*>(up.w+ui);
        float gs=bf(gate.s[gi/64]),gb=bf(gate.b[gi/64]);
        float us=bf(up.s[ui/64]),ub=bf(up.b[ui/64]);
        #pragma unroll
        for(int j=0;j<8;j++) {
            float activation=input[base+j];
            float g=float(((j<4?gw.x:gw.y)>>(8*(j&3)))&255)*gs+gb;
            float u=float(((j<4?uw.x:uw.y)>>(8*(j&3)))&255)*us+ub;
            gateSum+=g*activation;
            upSum+=u*activation;
        }
    }
    gateSum=sum32(gateSum);upSum=sum32(upSum);
    if(lane==0) {
        size_t out=(size_t(t)*n+e)*gate.rows+row;
        gateOut[out]=gateSum;upOut[out]=upSum;
    }
}
// Reuse each dequantized dense weight across independent token accumulators.
// Preserve the original per-lane accumulation and warp reduction order.
template<int Tile,bool Packed=false> __global__ void chunk_dense_reuse(Matrix m,const float *x,float *y,int count) {
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,first=blockIdx.y*Tile;
    if(row>=m.rows)return;
    float sums[Tile]{};
    for(int base=lane*8;base<m.cols;base+=256) {
      uint2 bytes{};float scale=0,bias=0;
      if constexpr(Packed) {
        size_t index=size_t(row)*m.cols+base;
        bytes=*reinterpret_cast<const uint2*>(m.w+index);
        scale=bf(m.s[index/64]);bias=bf(m.b[index/64]);
      }
      for(int j=0;j<8;j++) {
        float w;
        if constexpr(Packed)w=float(((j<4?bytes.x:bytes.y)>>(8*(j&3)))&255)*scale+bias;
        else w=weight(m,row,base+j);
        #pragma unroll
        for(int t=0;t<Tile;t++)if(first+t<count)
            sums[t]+=w*x[size_t(first+t)*m.cols+base+j];
      }
    }
    #pragma unroll
    for(int t=0;t<Tile;t++) {
        float v=sum32(sums[t]);
        if(lane==0&&first+t<count)y[size_t(first+t)*m.rows+row]=v;
    }
}
__device__ float chunk_previous(const float *raw,const float *history,int t,int back,int d) {
    int index=t-back;return index<0?history[(index+3)*8192+d]:raw[index*8192+d];
}
__global__ void chunk_linear(const float *raw,const float *history,const float *p,float *conv,float *a,float *b) {
    int t=threadIdx.x,h=blockIdx.x,seq=blockIdx.y,d=h*128+t;
    const float *w=p+4096;
    float v=sil(chunk_previous(raw,history,seq,3,d)*w[d*4]+chunk_previous(raw,history,seq,2,d)*w[d*4+1]+
                chunk_previous(raw,history,seq,1,d)*w[d*4+2]+raw[seq*8192+d]*w[d*4+3]);
    if(h<32)v*=rsqrtf(blocksum(v*v)+1e-6f)*(h<16?rsqrtf(128.f):1.f);
    conv[seq*8192+d]=v;
    if(h==0&&t<32){float z=a[seq*32+t]+p[36896+t];a[seq*32+t]=expf(-expf(p[36864+t])*(z>20?z:logf(1.f+expf(z))));b[seq*32+t]=sig(b[seq*32+t]);}
}
__global__ void chunk_history(const float *raw,float *history,int tokens) {
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<8192){float v0=chunk_previous(raw,history,tokens,3,d),v1=chunk_previous(raw,history,tokens,2,d),v2=chunk_previous(raw,history,tokens,1,d);history[d]=v0;history[8192+d]=v1;history[16384+d]=v2;}
}
