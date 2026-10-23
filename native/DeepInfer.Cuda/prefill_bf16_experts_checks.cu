// Small standalone checks for the experimental BF16 routed-projection path.
// No model weights are loaded and production execution mode remains unchanged.
#include "split_prefix.cu"

static float rounded(float x) {
    unsigned bits; std::memcpy(&bits,&x,4);
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    std::memcpy(&x,&bits,4); return x;
}
static float frombf(unsigned short bits) {
    unsigned word=unsigned(bits)<<16; float x;
    std::memcpy(&x,&word,4); return x;
}
static void check(int tokens,int rows,int cols,bool routed) {
    constexpr int experts=16, selected=8;
    Model g; Matrix m; m.experts=experts; m.rows=rows; m.cols=cols;
    size_t weights=size_t(experts)*rows*cols, outputs=size_t(tokens)*selected*rows;
    std::vector<unsigned char> w(weights);
    std::vector<unsigned short> s(weights/64),b(weights/64);
    std::vector<float> x(size_t(tokens)*(routed?selected:1)*cols),actual(outputs+64,-123.f);
    std::vector<int> ids(size_t(tokens)*selected);
    for(size_t i=0;i<w.size();i++)w[i]=(i*17+3)%256;
    for(size_t i=0;i<s.size();i++){s[i]=0x3b00+(i%8)*16;b[i]=0xbe00+(i%4)*16;}
    for(size_t i=0;i<x.size();i++)x[i]=float(int((i*11)%101)-50)/211.f;
    for(size_t i=0;i<ids.size();i++)ids[i]=(i*7+i/selected)%experts;
    m.w=g.alloc<unsigned char>(w.size());m.s=g.alloc<unsigned short>(s.size());
    m.b=g.alloc<unsigned short>(b.size());
    auto *dx=g.alloc<float>(x.size()),*dy=g.alloc<float>(actual.size());
    auto *di=g.alloc<int>(ids.size());
    ck(cudaMemcpy(m.w,w.data(),w.size(),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(m.s,s.data(),s.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(m.b,b.data(),b.size()*2,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dy,actual.data(),actual.size()*4,cudaMemcpyHostToDevice));
    ck(cudaMemcpy(di,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
    chunk_gemv_bf16<<<dim3((rows+3)/4,selected,tokens),128>>>(m,dx,dy,di,routed);
    ck(cudaGetLastError());
    ck(cudaMemcpy(actual.data(),dy,actual.size()*4,cudaMemcpyDeviceToHost));
    double error2=0,reference2=0,maxAbs=0;
    for(int t=0;t<tokens;t++)for(int e=0;e<selected;e++)for(int r=0;r<rows;r++) {
        int expert=ids[t*selected+e];size_t weightRow=(size_t(expert)*rows+r)*cols;
        size_t inputRow=(size_t(t)*(routed?selected:1)+(routed?e:0))*cols;
        double expected=0;
        for(int c=0;c<cols;c++) {
            size_t i=weightRow+c;
            float dequant=std::fma(float(w[i]),frombf(s[i/64]),frombf(b[i/64]));
            expected+=double(rounded(dequant))*rounded(x[inputRow+c]);
        }
        double delta=actual[(size_t(t)*selected+e)*rows+r]-expected;
        error2+=delta*delta;reference2+=expected*expected;
        maxAbs=std::max(maxAbs,std::abs(delta));
    }
    for(size_t i=outputs;i<actual.size();i++)
        if(actual[i]!=-123.f)throw std::runtime_error("BF16 expert tail overwrite");
    double relative=std::sqrt(error2/std::max(1e-30,reference2));
    if(!std::isfinite(relative)||maxAbs>=.01||relative>=1e-4)
        throw std::runtime_error("BF16 expert oracle mismatch");
    printf("{\"tokens\":%d,\"rows\":%d,\"cols\":%d,\"routedInput\":%s,\"relativeL2\":%.9g,\"maxAbs\":%.9g}\n",
           tokens,rows,cols,routed?"true":"false",relative,maxAbs);
}
int main() {try {
    cudaDeviceProp device{};ck(cudaGetDeviceProperties(&device,0));
    if(device.major<8)throw std::runtime_error("BF16 requires SM80+");
    for(bool routed:{false,true}) {
        check(1,67,192,routed);
        check(17,129,256,routed);
        check(32,256,512,routed);
    }
    puts("{\"passed\":true}");return 0;
}catch(const std::exception &e){fprintf(stderr,"FAILED: %s\n",e.what());return 1;}}
