// Original SM89 prefill primitives. Experimental: not connected to the executor yet.
// BF16 operands, FP32 accumulation; affine8 weights remain packed in global memory.
#include <mma.h>
__device__ float prefill_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__global__ void prefill_gemv_bf16(Matrix m, const float *x, float *y, int tokens) {
    int lane=threadIdx.x&31, row=blockIdx.x*4+threadIdx.x/32, token=blockIdx.y;
    if(row>=m.rows || token>=tokens)return;
    float total=0;
    for(int base=lane*8;base<m.cols;base+=256)
        for(int j=0;j<8 && base+j<m.cols;j++)
            total+=prefill_bf16(weight(m,row,base+j))*prefill_bf16(x[token*m.cols+base+j]);
    total=sum32(total);if(lane==0)y[token*m.rows+row]=total;
}
// CTA output: 32 token rows x 64 output columns. Four warps each own two 16x16 tiles.
__global__ void prefill_gemm_bf16(Matrix m, const float *x, float *y, int tokens) {
    using namespace nvcuda;
    __shared__ __align__(32) __nv_bfloat16 sa[32*32], sb[64*32];
    __shared__ __align__(32) float result[4*512];
    int tid=threadIdx.x, warp=tid/32, lane=tid&31;
    int token0=blockIdx.y*32, row0=blockIdx.x*64;
    wmma::fragment<wmma::accumulator,16,16,16,float> c0,c1;
    wmma::fill_fragment(c0,0.f);wmma::fill_fragment(c1,0.f);
    for(int base=0;base<m.cols;base+=32) {
        for(int i=tid;i<32*32;i+=128) {
            int token=token0+i/32, col=base+i%32;
            sa[i]=__float2bfloat16_rn(token<tokens && col<m.cols ? x[token*m.cols+col] : 0.f);
        }
        for(int i=tid;i<64*32;i+=128) {
            int row=row0+i/32,col=base+i%32;
            sb[i]=__float2bfloat16_rn(row<m.rows && col<m.cols ? weight(m,row,col) : 0.f);
        }
        __syncthreads();
        for(int k=0;k<32;k+=16) {
            wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> av;
            wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b0,b1;
            wmma::load_matrix_sync(av,sa+(warp/2)*16*32+k,32);
            wmma::load_matrix_sync(b0,sb+(warp%2)*32*32+k,32);
            wmma::load_matrix_sync(b1,sb+((warp%2)*32+16)*32+k,32);
            wmma::mma_sync(c0,av,b0,c0);wmma::mma_sync(c1,av,b1,c1);
        }
        __syncthreads();
    }
    wmma::store_matrix_sync(result+warp*512,c0,16,wmma::mem_row_major);
    wmma::store_matrix_sync(result+warp*512+256,c1,16,wmma::mem_row_major);
    __syncwarp();
    for(int i=lane;i<256;i+=32) {
        int token=token0+(warp/2)*16+i/16, row=row0+(warp%2)*32+i%16;
        if(token<tokens && row<m.rows)y[token*m.rows+row]=result[warp*512+i];
        if(token<tokens && row+16<m.rows)y[token*m.rows+row+16]=result[warp*512+256+i];
    }
}
// Identical Delta recurrence/order to delta(), holding the state row in registers
// across the token sequence. Prepared conv/a/b for the entire chunk are required.
__global__ void prefill_delta_sequence(float *state,const float *conv,const float *a,
                                        const float *b,float *out,int tokens) {
    int lane=threadIdx.x&31,row=blockIdx.x*2+threadIdx.x/32,head=row/128,kh=head/2;
    float values[4];
    for(int j=0;j<4;j++)values[j]=state[row*128+lane+j*32];
    for(int t=0;t<tokens;t++) {
        const float *c=conv+t*8192;
        float decayed[4],pred=0;
        for(int j=0;j<4;j++) {
            int d=lane+j*32;decayed[j]=values[j]*a[t*32+head];
            pred+=decayed[j]*c[2048+kh*128+d];
        }
        pred=sum32(pred);
        float correction=(c[4096+row]-pred)*b[t*32+head],value=0;
        for(int j=0;j<4;j++) {
            int d=lane+j*32;
            values[j]=decayed[j]+c[2048+kh*128+d]*correction;
            value+=values[j]*c[kh*128+d];
        }
        value=sum32(value);if(lane==0)out[t*4096+row]=value;
    }
    for(int j=0;j<4;j++)state[row*128+lane+j*32]=values[j];
}
