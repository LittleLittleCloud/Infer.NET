#pragma once
// Partition FP32 attention over the key axis. Each warp computes one QK dot;
// each lane feature accumulates PV locally, then a second kernel combines LSE.
// KV can be FP32 or BF16; no operand quantization is added here.
template<class KV,int Tile>
__global__ void chunk_attention_parts(const float *q,const KV *keys,const KV *values,
                                     float *parts,int pos,int partitions) {
    int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    int head=blockIdx.x,seq=blockIdx.y,part=blockIdx.z;
    int start=part*Tile,length=min(Tile,pos+seq+1-start);
    size_t offset=((size_t(seq)*16+head)*partitions+part)*258;
    if(length<=0){if(tid==0){parts[offset+256]=-INFINITY;parts[offset+257]=0.f;}return;}
    __shared__ float scores[Tile],query[256];
    query[tid]=q[seq*4096+head*256+tid];__syncthreads();
    for(int i=warp;i<length;i+=8){
        float dot=0;
        for(int d=lane;d<256;d+=32)
            dot+=query[d]*float(keys[size_t(start+i)*512+(head/8)*256+d]);
        dot=sum32(dot);if(lane==0)scores[i]=dot/16.f;
    }
    __syncthreads();
    float best=-INFINITY;
    for(int i=tid;i<length;i+=256)best=fmaxf(best,scores[i]);
    for(int n=16;n;n/=2)best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,n));
    __shared__ float maxima[8];if(lane==0)maxima[warp]=best;__syncthreads();
    if(tid==0){float b=maxima[0];for(int i=1;i<8;i++)b=fmaxf(b,maxima[i]);maxima[0]=b;}
    __syncthreads();best=maxima[0];float total=0;
    for(int i=tid;i<length;i+=256){float v=expf(scores[i]-best);scores[i]=v;total+=v;}
    float den=blocksum(total);__syncthreads();
    float output=0;
    for(int i=0;i<length;i++)output+=scores[i]*float(values[size_t(start+i)*512+(head/8)*256+tid]);
    parts[offset+tid]=output;
    if(tid==0){parts[offset+256]=best;parts[offset+257]=den;}
}
__global__ void chunk_attention_combine(const float *parts,const float *raw,float *output,
                                         int pos,int partitions,int tile) {
    int tid=threadIdx.x,head=blockIdx.x,seq=blockIdx.y;
    int count=(pos+seq+tile)/tile;
    const float *p=parts+(size_t(seq)*16+head)*partitions*258;
    float best=-INFINITY;
    for(int i=0;i<count;i++)best=fmaxf(best,p[size_t(i)*258+256]);
    float total=0,value=0;
    for(int i=0;i<count;i++){
        float scale=expf(p[size_t(i)*258+256]-best);
        total+=scale*p[size_t(i)*258+257];value+=scale*p[size_t(i)*258+tid];
    }
    output[seq*4096+head*256+tid]=(value/total)*sig(raw[seq*8192+head*512+256+tid]);
}
