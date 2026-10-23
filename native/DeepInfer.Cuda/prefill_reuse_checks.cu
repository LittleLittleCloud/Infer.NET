// Standalone operator validation and timing, no model files or production backend swap.
#include "split_prefix.cu"
#include "prefill_batch.cuh"
#include <functional>
static float rounded(float x) {
    unsigned bits;memcpy(&bits,&x,4);bits+=0x7fff+((bits>>16)&1);bits&=0xffff0000;
    memcpy(&x,&bits,4);return x;
}
static float frombf(unsigned short v) {unsigned bits=unsigned(v)<<16;float f;memcpy(&f,&bits,4);return f;}
static std::vector<float> timings(const std::function<void()> &work,int repetitions) {
    work();ck(cudaDeviceSynchronize());std::vector<float> results;
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin));ck(cudaEventCreate(&end));
    for(int trial=0;trial<3;trial++) {
        ck(cudaEventRecord(begin));for(int i=0;i<repetitions;i++)work();ck(cudaEventRecord(end));
        ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,begin,end));results.push_back(ms/repetitions);
    }
    ck(cudaEventDestroy(begin));ck(cudaEventDestroy(end));return results;
}
static void print_times(const char *name,const std::vector<float>& a) {
    printf("\"%sMs\":[%.6f,%.6f,%.6f]",name,a[0],a[1],a[2]);
}
template<int Tile,bool Packed=false> static void reuse_case(int tokens,int rows,int cols,bool oracle) {
    Model g;Matrix m;m.experts=1;m.rows=rows;m.cols=cols;
    size_t size=size_t(rows)*cols;
    std::vector<unsigned char>w(size);std::vector<unsigned short>s(size/64),b(size/64);
    std::vector<float>x(size_t(tokens)*cols),actual(size_t(tokens)*rows+64,-123.f),reference(actual.size(),-123.f);
    for(size_t i=0;i<size;i++)w[i]=(i*17+3)%256;
    for(size_t i=0;i<s.size();i++){s[i]=0x3b00+(i%8)*16;b[i]=0xbe00+(i%4)*16;}
    for(size_t i=0;i<x.size();i++)x[i]=float(int((i*11)%101)-50)/211.f;
    m.w=g.alloc<unsigned char>(size);m.s=g.alloc<unsigned short>(s.size());m.b=g.alloc<unsigned short>(b.size());
    float *input=g.alloc<float>(x.size()),*y=g.alloc<float>(actual.size()),*ref=g.alloc<float>(actual.size());
    ck(cudaMemcpy(m.w,w.data(),size,cudaMemcpyHostToDevice));ck(cudaMemcpy(m.s,s.data(),s.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(m.b,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(input,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(y,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ref,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));
    auto gemm=[&]{chunk_dense_reuse<Tile,Packed><<<dim3((rows+3)/4,(tokens+Tile-1)/Tile),128>>>(m,input,y,tokens);};
    auto gemv=[&]{chunk_gemv<<<dim3((rows+3)/4,1,tokens),128>>>(m,input,ref,nullptr,0);};
    gemm();gemv();ck(cudaGetLastError());ck(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(reference.data(),ref,reference.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=reference)throw std::runtime_error("Reuse must match FP32 GEMV bitwise");
    double maxAbs=0,sumError=0,sumRef=0;
    for(int t=0;t<tokens;t++)for(int r=0;r<rows;r++) {
        double expected=reference[t*rows+r];
        if(oracle){expected=0;for(int k=0;k<cols;k++){size_t i=size_t(r)*cols+k;expected+=double(std::fma(float(w[i]),frombf(s[i/64]),frombf(b[i/64])))*x[t*cols+k];}}
        double error=actual[t*rows+r]-expected;maxAbs=std::max(maxAbs,fabs(error));sumError+=error*error;sumRef+=expected*expected;
    }
    for(size_t i=size_t(tokens)*rows;i<actual.size();i++)if(actual[i]!=-123.f||reference[i]!=-123.f)throw std::runtime_error("GEMM tail overwrite");
    double relative=sqrt(sumError/std::max(1e-30,sumRef));if(!std::isfinite(relative)||maxAbs>=.01||relative>=1e-4)throw std::runtime_error("GEMM FP32 oracle mismatch");
    auto v=timings(gemv,5),z=timings(gemm,5);
    printf("{\"op\":\"affine8-fp32-reuse\",\"tokens\":%d,\"rows\":%d,\"cols\":%d,\"oracle\":\"%s\",\"maxAbs\":%.9g,\"relativeL2\":%.9g,",tokens,rows,cols,oracle?"double-fp32":"simt-fp32",maxAbs,relative);
    printf("\"tile\":%d,\"packed\":%s,\"bitwiseEqual\":true,",Tile,Packed?"true":"false");print_times("simt",v);printf(",");print_times("reuse",z);puts("}");fflush(stdout);
}
int main(){try {
    for(auto v:std::vector<std::vector<int>>{{1,13,256},{3,33,512},{17,83,2048},{31,256,2048},{32,8192,2048},{32,2048,4096},{32,512,2048}}) {
        bool oracle=v[1]<100;
        reuse_case<2>(v[0],v[1],v[2],oracle);reuse_case<4>(v[0],v[1],v[2],oracle);reuse_case<8>(v[0],v[1],v[2],oracle);
        reuse_case<4,true>(v[0],v[1],v[2],oracle);reuse_case<8,true>(v[0],v[1],v[2],oracle);
    }
    puts("{\"passed\":true}");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}
