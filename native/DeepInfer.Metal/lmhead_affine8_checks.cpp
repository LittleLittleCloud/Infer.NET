#include "lmhead_affine8_layout.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool ok,const char *what) {
    if(!ok){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}
}
static float bf(uint16_t bits) {uint32_t u=uint32_t(bits)<<16;float f;std::memcpy(&f,&u,4);return f;}
static void roundtrip(uint32_t rows) {
    using namespace lmhead_affine8;
    Layout l(rows);
    std::vector<uint8_t> codes(size_t(rows)*2048),packed(l.bytes+32,0xa5);
    std::vector<uint16_t> scales(size_t(rows)*32),biases(scales.size());
    for(size_t i=0;i<codes.size();i++)codes[i]=uint8_t((i*73+(i/2048)*19+(i/64)*11)^(i>>9));
    for(size_t i=0;i<scales.size();i++){scales[i]=uint16_t(i*7919);biases[i]=uint16_t(~(i*3571));}
    require(pack(rows,codes.data(),codes.size(),scales.data(),scales.size()*2,biases.data(),biases.size()*2,
        packed.data()+16,l.bytes),"pack");
    for(int i=0;i<16;i++)require(packed[i]==0xa5&&packed[l.bytes+16+i]==0xa5,"pack sentinels");
    // Decode the serialized stream independently, without using Layout offsets.
    for(size_t i=0;i<l.codesBytes;i++) {
        size_t octet=i/8,r=octet%4,lane=(octet/4)%32,block=(octet/128)%8,tile=octet/1024;
        size_t row=tile*4+r,column=block*256+lane*8+i%8;
        require(packed[16+i]==(row<rows?codes[row*2048+column]:0),"all code bytes / padded rows");
    }
    for(size_t i=0;i<l.paddedRows*32;i++) {
        size_t row=(i/128)*4+i%4,g=(i/4)%32;uint16_t s,b;
        std::memcpy(&s,packed.data()+16+l.codesBytes+i*2,2);
        std::memcpy(&b,packed.data()+16+l.codesBytes+l.paddedRows*64+i*2,2);
        require(s==(row<rows?scales[row*32+g]:0)&&b==(row<rows?biases[row*32+g]:0),"all BF16 bits / padding");
    }
    auto digest=[&](){uint64_t h=14695981039346656037ull;for(uint8_t v:packed){h^=v;h*=1099511628211ull;}return h;};
    auto before=digest();
    require(!pack(rows,codes.data(),codes.size()-1,scales.data(),scales.size()*2,biases.data(),biases.size()*2,
        packed.data()+16,l.bytes),"reject code size");
    require(!pack(rows,codes.data(),codes.size(),scales.data(),scales.size()*2-1,biases.data(),biases.size()*2,
        packed.data()+16,l.bytes),"reject scale size");
    require(!pack(rows,codes.data(),codes.size(),scales.data(),scales.size()*2,biases.data(),biases.size()*2-1,
        packed.data()+16,l.bytes),"reject bias size");
    require(!pack(rows,codes.data(),codes.size(),scales.data(),scales.size()*2,biases.data(),biases.size()*2,
        packed.data()+16,l.bytes-1),"reject destination size");
    require(!pack(rows,nullptr,codes.size(),scales.data(),scales.size()*2,biases.data(),biases.size()*2,
        packed.data()+16,l.bytes),"reject null");
    require(before==digest(),"negative calls do not write");
    std::printf("PASS roundtrip rows=%u codesBytes=%zu scalesOffset=%zu biasesOffset=%zu totalBytes=%zu fnv64=%016llx\n",
        rows,l.codesBytes,l.scalesOffset,l.biasesOffset,l.bytes,(unsigned long long)before);
}
static void operator_check() {
    using namespace lmhead_affine8;
    constexpr uint32_t rows=17;Layout l(rows);
    std::vector<uint8_t> codes(rows*2048),packed(l.bytes);
    std::vector<uint16_t> s(rows*32),b(s.size());std::vector<float> x(2048);
    for(int pattern=0;pattern<5;pattern++) {
        for(size_t i=0;i<codes.size();i++)codes[i]=uint8_t(i*73+(i/2048)*17);
        for(size_t i=0;i<s.size();i++) {
            s[i]=pattern==0?0:pattern==3?((i%3)==0?0xbd80:0x3d80):((i%3)==0?0xbb80:0x3b80);
            b[i]=pattern==1?0:pattern==3?((i%2)?0xc180:0x4180):((i%2)?0xbf00:0x3f00);
        }
        for(int c=0;c<2048;c++)x[c]=pattern==0?0:pattern==4?((c%2)?-64.f:64.f):std::sin(c*.031f)*.1f;
        require(pack(rows,codes.data(),codes.size(),s.data(),s.size()*2,b.data(),b.size()*2,packed.data(),packed.size()),"operator pack");
        double error=0,norm=0,maxabs=0;
        for(uint32_t row=0;row<rows;row++) {
            float reference=0;double expected=0;
            // Row-major scalar oracle has no knowledge of the packed offsets or lane mapping.
            for(uint32_t c=0;c<2048;c++) {
                volatile float product=float(codes[row*2048+c])*bf(s[row*32+c/64]);
                volatile float weight=product+bf(b[row*32+c/64]);
                volatile float term=weight*x[c];reference+=term;expected+=term;
            }
            float lanes[32]{};
            for(uint32_t lane=0;lane<32;lane++)for(uint32_t block=0;block<8;block++)for(uint32_t j=0;j<8;j++) {
                uint32_t c=block*256+lane*8+j;uint16_t scale,bias;
                std::memcpy(&scale,packed.data()+l.scalesOffset+l.parameter(row,c/64)*2,2);
                std::memcpy(&bias,packed.data()+l.biasesOffset+l.parameter(row,c/64)*2,2);
                volatile float product=float(packed[l.code(row,c)])*bf(scale);
                volatile float weight=product+bf(bias);volatile float term=weight*x[c];lanes[lane]+=term;
            }
            float result=0;for(float v:lanes)result+=v;
            double diff=double(result)-reference;error+=diff*diff;norm+=double(reference)*reference;
            maxabs=std::max(maxabs,std::abs(diff));
            require(std::abs(result-expected)<.01,"CPU high-accuracy sum diagnostic");
            require(std::isfinite(reference)&&std::isfinite(result),"finite CPU operator");
        }
        double l2=norm?std::sqrt(error/norm):std::sqrt(error);
        require(l2<1e-4&&maxabs<.01,"independent CPU FP32 gate");
        std::printf("PASS CPU operator pattern=%d relL2=%.9g maxAbs=%.9g\n",pattern,l2,maxabs);
    }
}
int main() {
    using namespace lmhead_affine8;
    require(geometry(1,Rows,2048,8,64),"real geometry");
    require(!geometry(2,Rows,2048,8,64)&&!geometry(1,Rows-1,2048,8,64)&&!geometry(1,Rows,1024,8,64)&&
        !geometry(1,Rows,2048,5,64)&&!geometry(1,Rows,2048,8,32),"invalid geometry/precision");
    require(Layout(Rows).bytes==540344320&&Layout(Rows).scalesOffset==508559360&&
        Layout(Rows).biasesOffset==524451840,"fixed byte totals");
    require(!pack(0,nullptr,0,nullptr,0,nullptr,0,nullptr,0)&&
        !pack(Rows+1,nullptr,0,nullptr,0,nullptr,0,nullptr,0),"invalid row limits");
    for(uint32_t rows:{1u,3u,4u,5u,15u,16u,17u,Rows})roundtrip(rows);
    operator_check();
    std::puts("PASS: independent full-byte roundtrip, BF16 bit preservation, FP32 CPU operators and negative checks (no Metal execution)");
}
