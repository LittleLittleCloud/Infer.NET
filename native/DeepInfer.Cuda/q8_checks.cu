// Independent fixtures are generated before execution by q8-kv-fixtures.py.
#include "split_prefix.cu"
#include <iterator>
static std::vector<char> fixture(const char *path){
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("Missing fixture");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
__global__ void codec(const float *x,unsigned char *y,int *error){int i=blockIdx.x*blockDim.x+threadIdx.x;kv_write(y,i,x[i],error);}
__global__ void restore(const unsigned char *x,float *y){int i=blockIdx.x*blockDim.x+threadIdx.x;y[i]=kv_read(x,i);}
int main(int argc,char **argv){try{
    if(argc!=3)return 2;auto input=fixture(argv[1]),expected=fixture(argv[2]);
    if(input.size()%2048||expected.size()!=input.size()/2048*544)return 2;
    Model owner;auto *x=owner.alloc<float>(input.size()/4),*z=owner.alloc<float>(input.size()/4);
    auto *y=owner.alloc<unsigned char>(expected.size());auto *error=owner.alloc<int>(1);
    std::vector<unsigned char> actual(expected.size());std::vector<float> decoded(input.size()/4);
    for(int arm=0;arm<5;arm++){
        if(arm){float value=arm==1?NAN:arm==2?INFINITY:arm==3?1e30f:1e-30f;memcpy(input.data(),&value,4);for(int i=1;i<32;i++)memset(input.data()+i*4,0,4);}
        ck(cudaMemcpy(x,input.data(),input.size(),cudaMemcpyHostToDevice));ck(cudaMemset(error,0,4));
        codec<<<input.size()/4/256,256>>>(x,y,error);restore<<<input.size()/4/256,256>>>(y,z);
        int invalid=0;ck(cudaMemcpy(&invalid,error,4,cudaMemcpyDeviceToHost));
        if(arm){if(!invalid)throw std::runtime_error("Invalid Q8 accepted");continue;}
        ck(cudaMemcpy(actual.data(),y,actual.size(),cudaMemcpyDeviceToHost));
        if(invalid||memcmp(actual.data(),expected.data(),actual.size()))throw std::runtime_error("Q8 bytes differ from independent oracle");
        ck(cudaMemcpy(decoded.data(),z,input.size(),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<decoded.size();i++){
            const unsigned char *b=actual.data()+(i/32)*34;
            float value=__half2float(*reinterpret_cast<const __half*>(b))*float(static_cast<signed char>(b[2+i%32]));
            if(decoded[i]!=value)throw std::runtime_error("Q8 load mismatch");
        }
    }
    puts("PASS: CUDA Q8 bytes, loads, zero/ties and invalid input/scale");return 0;
}catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}}
