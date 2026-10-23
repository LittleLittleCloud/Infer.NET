// Independent double oracle for short causal sequences; retained SIMT baseline
// for larger contexts. Operator GPU timings exclude CPU work and model traffic.
#include "split_prefix.cu"
#include "prefill_attention.cuh"
#include <functional>
static float sample(unsigned &state){state^=state<<13;state^=state>>17;state^=state<<5;return (int(state%20001)-10000)/10000.f;}
static double median_ms(const std::function<void()> &f){
    f();ck(cudaDeviceSynchronize());cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));std::vector<float> times;
    for(int i=0;i<5;i++){ck(cudaEventRecord(a));f();ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));times.push_back(ms);}
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));std::sort(times.begin(),times.end());return times[2];
}
template<class KV,int Tile>
static void check_case(int length,int tokens,bool oracle){
    Model g;int pos=length-tokens,partitions=(length+Tile-1)/Tile;unsigned seed=12345;
    std::vector<float> q(size_t(tokens)*4096),raw(size_t(tokens)*8192),out(size_t(tokens)*4096+64,-999.f),reference(out.size(),-999.f);
    std::vector<KV> keys(size_t(length)*512),values(keys.size());
    for(auto &v:q)v=sample(seed)*2.f;for(auto &v:raw)v=sample(seed);
    for(auto &v:keys)v=KV(sample(seed)*2.f);for(auto &v:values)v=KV(sample(seed));
    auto *dq=g.alloc<float>(q.size()),*dr=g.alloc<float>(raw.size()),*dy=g.alloc<float>(out.size()),*ref=g.alloc<float>(out.size());
    auto *dk=g.alloc<KV>(keys.size()),*dv=g.alloc<KV>(values.size());
    auto *scores=g.alloc<float>(size_t(tokens)*16*length),*parts=g.alloc<float>(size_t(tokens)*16*partitions*258);
    ck(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,raw.data(),raw.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dk,keys.data(),keys.size()*sizeof(KV),cudaMemcpyHostToDevice));ck(cudaMemcpy(dv,values.data(),values.size()*sizeof(KV),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dy,out.data(),out.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ref,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));
    auto legacy=[&]{chunk_scores<<<dim3((length+3)/4,16,tokens),128>>>(dq,dk,scores,pos,length);chunk_attend<<<dim3(16,tokens),256>>>(scores,dv,dr,ref,pos,length);};
    auto candidate=[&]{chunk_attention_parts<KV,Tile><<<dim3(16,tokens,partitions),256>>>(dq,dk,dv,parts,pos,partitions);chunk_attention_combine<<<dim3(16,tokens),256>>>(parts,dr,dy,pos,partitions,Tile);};
    legacy();candidate();ck(cudaGetLastError());ck(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(reference.data(),ref,reference.size()*4,cudaMemcpyDeviceToHost));
    std::vector<double> expected(size_t(tokens)*4096);
    if(oracle){
        for(int t=0;t<tokens;t++)for(int h=0;h<16;h++){
            int count=pos+t+1;std::vector<double> prob(count);double best=-INFINITY,total=0;
            for(int i=0;i<count;i++){double sum=0;for(int d=0;d<256;d++)sum+=double(q[t*4096+h*256+d])*float(keys[size_t(i)*512+(h/8)*256+d]);prob[i]=sum/16;best=std::max(best,prob[i]);}
            for(auto &v:prob){v=std::exp(v-best);total+=v;}
            for(int d=0;d<256;d++){double sum=0;for(int i=0;i<count;i++)sum+=prob[i]*float(values[size_t(i)*512+(h/8)*256+d]);expected[t*4096+h*256+d]=sum/total/(1+std::exp(-double(raw[t*8192+h*512+256+d])));}
        }
    }else for(size_t i=0;i<expected.size();i++)expected[i]=reference[i];
    double maxAbs=0,error2=0,ref2=0;
    for(size_t i=0;i<expected.size();i++){double e=out[i]-expected[i];if(!std::isfinite(out[i]))throw std::runtime_error("Nonfinite attention");maxAbs=std::max(maxAbs,std::abs(e));error2+=e*e;ref2+=expected[i]*expected[i];}
    for(size_t i=expected.size();i<out.size();i++)if(out[i]!=-999.f || reference[i]!=-999.f)throw std::runtime_error("Attention tail overwrite");
    double relative=std::sqrt(error2/std::max(ref2,1e-30));if(maxAbs>=.01||relative>=1e-4)throw std::runtime_error("Attention numerical gate failed");
    double baseline=median_ms(legacy),optimized=median_ms(candidate);
    printf("{\"length\":%d,\"tokens\":%d,\"kvBytes\":%d,\"tile\":%d,\"oracle\":\"%s\",\"maxAbs\":%.9g,\"relativeL2\":%.9g,\"legacyMedianMs\":%.6f,\"partitionMedianMs\":%.6f,\"partBytes\":%llu}\n",length,tokens,int(sizeof(KV)),Tile,oracle?"double":"legacy-fp32",maxAbs,relative,baseline,optimized,(unsigned long long)(size_t(tokens)*16*partitions*258*4));fflush(stdout);
}
int main(){try{
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));printf("{\"device\":\"%s\",\"kind\":\"operator-only\",\"warmups\":1,\"trials\":5}\n",p.name);
    for(auto c:std::vector<std::pair<int,int>>{{17,1},{257,17},{513,32}}){check_case<float,256>(c.first,c.second,true);check_case<__nv_bfloat16,256>(c.first,c.second,true);}
    for(int length:{2048,8192,32768,131072}){
        check_case<float,128>(length,32,false);check_case<float,256>(length,32,false);check_case<float,512>(length,32,false);
        check_case<__nv_bfloat16,256>(length,32,false);
    }
    puts("{\"passed\":true}");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}
