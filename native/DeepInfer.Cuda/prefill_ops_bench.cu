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
static void gemm_case(int tokens,int rows,int cols,bool oracle) {
    Model g;Matrix m;m.experts=1;m.rows=rows;m.cols=cols;
    size_t size=size_t(rows)*cols;
    std::vector<unsigned char>w(size);std::vector<unsigned short>s(size/64),b(size/64);
    std::vector<float>x(size_t(tokens)*cols),actual(size_t(tokens)*rows+64,-123.f),reference(actual.size(),-123.f),scalarActual(actual.size(),-123.f);
    for(size_t i=0;i<size;i++)w[i]=(i*17+3)%256;
    for(size_t i=0;i<s.size();i++){s[i]=0x3b00+(i%8)*16;b[i]=0xbe00+(i%4)*16;}
    for(size_t i=0;i<x.size();i++)x[i]=float(int((i*11)%101)-50)/211.f;
    m.w=g.alloc<unsigned char>(size);m.s=g.alloc<unsigned short>(s.size());m.b=g.alloc<unsigned short>(b.size());
    float *input=g.alloc<float>(x.size()),*y=g.alloc<float>(actual.size()),*ref=g.alloc<float>(actual.size()),*old=g.alloc<float>(actual.size());
    ck(cudaMemcpy(m.w,w.data(),size,cudaMemcpyHostToDevice));ck(cudaMemcpy(m.s,s.data(),s.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(m.b,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(input,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(y,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ref,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(old,scalarActual.data(),scalarActual.size()*4,cudaMemcpyHostToDevice));
    auto gemm=[&]{prefill_gemm_bf16<true><<<dim3((rows+63)/64,(tokens+31)/32),128>>>(m,input,y,tokens);};
    auto scalar=[&]{prefill_gemm_bf16<false><<<dim3((rows+63)/64,(tokens+31)/32),128>>>(m,input,old,tokens);};
    auto gemv=[&]{prefill_gemv_bf16<<<dim3((rows+3)/4,tokens),128>>>(m,input,ref,tokens);};
    gemm();scalar();gemv();ck(cudaGetLastError());ck(cudaMemcpy(actual.data(),y,actual.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(reference.data(),ref,reference.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(scalarActual.data(),old,scalarActual.size()*4,cudaMemcpyDeviceToHost));
    if(actual!=scalarActual)throw std::runtime_error("Packed BF16 GEMM differs bitwise from scalar-load Tensor Core GEMM");
    double maxAbs=0,sumError=0,sumRef=0;
    for(int t=0;t<tokens;t++)for(int r=0;r<rows;r++) {
        double expected=reference[t*rows+r];
        if(oracle){expected=0;for(int k=0;k<cols;k++){size_t i=size_t(r)*cols+k;expected+=double(rounded(float(w[i])*frombf(s[i/64])+frombf(b[i/64])))*rounded(x[t*cols+k]);}}
        double error=actual[t*rows+r]-expected;maxAbs=std::max(maxAbs,fabs(error));sumError+=error*error;sumRef+=expected*expected;
    }
    for(size_t i=size_t(tokens)*rows;i<actual.size();i++)if(actual[i]!=-123.f||reference[i]!=-123.f)throw std::runtime_error("GEMM tail overwrite");
    double relative=sqrt(sumError/std::max(1e-30,sumRef));if(!std::isfinite(relative)||maxAbs>=.01||relative>=1e-4)throw std::runtime_error("GEMM BF16 oracle mismatch");
    auto v=timings(gemv,5),o=timings(scalar,5),z=timings(gemm,5);
    printf("{\"op\":\"affine8-bf16-gemm\",\"tokens\":%d,\"rows\":%d,\"cols\":%d,\"oracle\":\"%s\",\"maxAbs\":%.9g,\"relativeL2\":%.9g,\"bitwiseScalarEqual\":true,",tokens,rows,cols,oracle?"double-bf16":"simt-bf16",maxAbs,relative);
    print_times("simt",v);printf(",");print_times("scalarTensorCore",o);printf(",");print_times("tensorCore",z);puts("}");fflush(stdout);
}
static void delta_case(int tokens) {
    Model g;size_t n=32*128*128;
    std::vector<float> state(n),conv(size_t(tokens)*8192),a(tokens*32),b(tokens*32),out(size_t(tokens)*4096),ref(out.size()),ss(n),rs(n);
    for(size_t i=0;i<n;i++)state[i]=float(int(i%19)-9)/100.f;
    for(size_t i=0;i<conv.size();i++)conv[i]=float(int((i*7)%41)-20)/100.f;
    for(size_t i=0;i<a.size();i++){a[i]=.9f+float(i%9)*.01f;b[i]=.1f+float(i%13)*.01f;}
    auto *dc=g.alloc<float>(conv.size()),*da=g.alloc<float>(a.size()),*db=g.alloc<float>(b.size());
    auto *ds=g.alloc<float>(n),*dr=g.alloc<float>(n),*dy=g.alloc<float>(out.size()),*ry=g.alloc<float>(out.size());
    ck(cudaMemcpy(dc,conv.data(),conv.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(ds,state.data(),n*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dr,state.data(),n*4,cudaMemcpyHostToDevice));
    auto serial=[&]{for(int t=0;t<tokens;t++)delta<<<2048,64>>>(dr,dc+t*8192,da+t*32,db+t*32,ry+t*4096);};
    auto sequence=[&]{prefill_delta_sequence<<<2048,64>>>(ds,dc,da,db,dy,tokens);};
    serial();sequence();ck(cudaGetLastError());
    ck(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(ref.data(),ry,ref.size()*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(ss.data(),ds,n*4,cudaMemcpyDeviceToHost));ck(cudaMemcpy(rs.data(),dr,n*4,cudaMemcpyDeviceToHost));
    double maximum=0;for(size_t i=0;i<out.size();i++)maximum=std::max(maximum,double(fabs(out[i]-ref[i])));for(size_t i=0;i<n;i++)maximum=std::max(maximum,double(fabs(ss[i]-rs[i])));
    bool exact=out==ref&&ss==rs;if(!std::isfinite(maximum)||maximum>1e-6)throw std::runtime_error("Sequence Delta differs");
    auto v=timings(serial,5),z=timings(sequence,5);
    printf("{\"op\":\"delta-sequence\",\"tokens\":%d,\"maxAbs\":%.9g,\"bitwiseEqual\":%s,",tokens,maximum,exact?"true":"false");print_times("serial",v);printf(",");print_times("sequence",z);puts("}");fflush(stdout);
}
int main(){try {
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0));if(p.major<8)throw std::runtime_error("BF16 Tensor Cores require SM80+");
    printf("{\"device\":\"%s\",\"sm\":%d,\"kind\":\"standalone-operators-not-model-throughput\"}\n",p.name,p.major*10+p.minor);
    for(auto v:std::vector<std::vector<int>>{{1,13,64},{17,33,192},{35,83,256}})gemm_case(v[0],v[1],v[2],true);
    for(int t:{32,128,512})gemm_case(t,8192,2048,false);
    for(int t:{1,3,17,32,128,512})delta_case(t);
    puts("{\"passed\":true}");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}
