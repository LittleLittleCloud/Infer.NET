// Standalone CUDA operator check. Does not load model weights or alter the
// deployed prefix executor; compares against its retained FP32 attention path.
#include "split_prefix.cu"
#include <functional>

template<class KV> static void check_case(int length,int count) {
    Model g;
    const int pos=length-count;
    std::vector<float> q(size_t(count)*4096),raw(size_t(count)*8192);
    std::vector<KV> keys(size_t(length)*512),values(keys.size());
    unsigned seed=0x12345678u;
    auto sample=[&] {seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return (int(seed%20001)-10000)/10000.f;};
    for(auto &x:q)x=sample();
    for(auto &x:raw)x=sample();
    for(auto &x:keys)x=KV(sample());
    for(auto &x:values)x=KV(sample());
    const size_t outputCount=size_t(count)*4096+64;
    std::vector<float> baseline(outputCount,-123.f),candidate(baseline);
    float *dq=g.alloc<float>(q.size()),*dr=g.alloc<float>(raw.size());
    KV *dk=g.alloc<KV>(keys.size()),*dv=g.alloc<KV>(values.size());
    float *scores0=g.alloc<float>(size_t(count)*16*length),*scores1=g.alloc<float>(size_t(count)*16*length);
    float *y0=g.alloc<float>(outputCount),*y1=g.alloc<float>(outputCount);
    ck(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dr,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dk,keys.data(),keys.size()*sizeof(KV),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dv,values.data(),values.size()*sizeof(KV),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(y0,baseline.data(),outputCount*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(y1,candidate.data(),outputCount*4,cudaMemcpyHostToDevice));
    auto retained=[&] {
        chunk_scores<<<dim3((length+3)/4,16,count),128>>>(dq,dk,scores0,pos,length);
        chunk_attend<<<dim3(16,count),256>>>(scores0,dv,dr,y0,pos,length);
    };
    auto reuse=[&] {
        chunk_scores<<<dim3((length+3)/4,16,count),128>>>(dq,dk,scores1,pos,length);
        chunk_softmax<<<dim3(16,count),256>>>(scores1,pos,length);
        chunk_attend_reuse4<<<dim3(16,(count+3)/4),256>>>(scores1,dv,dr,y1,pos,length,count);
    };
    retained();reuse();ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(baseline.data(),y0,outputCount*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(candidate.data(),y1,outputCount*4,cudaMemcpyDeviceToHost));
    if(candidate!=baseline) {
        size_t first=0;while(first<outputCount && candidate[first]==baseline[first])first++;
        fprintf(stderr,"first mismatch length=%d count=%d kv=%zu index=%zu retained=%.9g reuse=%.9g\n",
                length,count,sizeof(KV),first,baseline[first],candidate[first]);
        throw std::runtime_error("Query reuse changed the retained attention result");
    }
    printf("{\"length\":%d,\"queries\":%d,\"kvBytes\":%zu,\"bitwiseEqual\":true}\n",length,count,sizeof(KV));
    fflush(stdout);
}
int main() {try {
    for(int length:{17,257,4096,32768})
        for(int count:{1,3,7,17,32}) {
            if(count>length)continue;
            check_case<float>(length,count);
            check_case<__nv_bfloat16>(length,count);
        }
    puts("{\"passed\":true,\"kind\":\"operator-only; not model acceptance\"}");
    return 0;
}catch(const std::exception &e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}
