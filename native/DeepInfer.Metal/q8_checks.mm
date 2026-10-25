// Model-free GPU codec checks against pre-frozen independent Python bytes.
#include "bridge.mm"
#include <fstream>
#include <iterator>
#include <vector>
static std::vector<char> read_file(const char *path) {
    std::ifstream f(path,std::ios::binary);
    if(!f)throw std::runtime_error("Missing fixture");
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
int main(int argc,char **argv){@autoreleasepool {
    if(argc!=3)return 2;
    auto input=read_file(argv[1]),expected=read_file(argv[2]);
    if(input.size()%2048||expected.size()!=input.size()/2048*544)return 2;
    GModel g{};g.kvQ8=true;g.compute.device=MTLCreateSystemDefaultDevice();g.compute.queue=[g.compute.device newCommandQueue];
    NSString *extra=@R"metal(
kernel void codec(device const float *x [[buffer(0)]],device DIKV *out [[buffer(1)]],device atomic_uint *error [[buffer(2)]],uint i [[thread_position_in_grid]]){kv_write(out,i,x[i],error);}
kernel void restore(device const DIKV *x [[buffer(0)]],device float *out [[buffer(1)]],uint i [[thread_position_in_grid]]){out[i]=kv_read(x,i);}
)metal";
    MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    NSError *failure=nil;
    auto lib=[g.compute.device newLibraryWithSource:[g_kv_source(&g) stringByAppendingString:extra] options:options error:&failure];
    if(!lib){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);return 1;}
    auto encode=[g.compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"codec"] error:&failure];
    auto decode=[g.compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"restore"] error:&failure];
    auto x=[g.compute.device newBufferWithBytes:input.data() length:input.size() options:MTLResourceStorageModeShared];
    auto y=[g.compute.device newBufferWithLength:expected.size() options:MTLResourceStorageModeShared];
    auto z=[g.compute.device newBufferWithLength:input.size() options:MTLResourceStorageModeShared];
    auto error=g_buffer(&g,1);
    for(int arm=0;arm<5;arm++){
        memset(error.contents,0,4);
        if(arm)*static_cast<float*>(x.contents)=arm==1?NAN:arm==2?INFINITY:arm==3?1e30f:1e-30f;
        if(arm)for(int j=1;j<32;j++)static_cast<float*>(x.contents)[j]=0;
        auto command=[g.compute.queue commandBuffer];auto e=[command computeCommandEncoder];
        [e setComputePipelineState:encode];g_bind(e,x,0);g_bind(e,y,1);g_bind(e,error,2);
        [e dispatchThreads:MTLSizeMake(input.size()/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [e setComputePipelineState:decode];g_bind(e,y,0);g_bind(e,z,1);
        [e dispatchThreads:MTLSizeMake(input.size()/4,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [e endEncoding];char message[2048]{};double ms;
        if(!q_complete(command,&ms,message,sizeof(message))){fprintf(stderr,"%s\n",message);return 1;}
        if(arm){if(!*static_cast<uint32_t*>(error.contents))return 1;continue;}
        if(*static_cast<uint32_t*>(error.contents)||memcmp(y.contents,expected.data(),expected.size())){
            auto *actual=static_cast<unsigned char*>(y.contents);for(size_t i=0;i<expected.size();i++)if(actual[i]!=(unsigned char)expected[i]){fprintf(stderr,"Q8 byte %zu: %u != %u\n",i,actual[i],(unsigned char)expected[i]);break;}return 1;
        }
        if(!di_q8_valid(y.contents,expected.size()))return 1;
        for(size_t i=0;i<input.size()/4;i++){
            const auto *b=reinterpret_cast<const unsigned char*>(expected.data())+(i/32)*34;
            __fp16 h;memcpy(&h,b,2);float ref=float(h)*float(static_cast<signed char>(b[2+i%32]));
            if(static_cast<float*>(z.contents)[i]!=ref)return 1;
        }
    }
    // Compile and exercise the production prefill/decode attention programs too.
    g.kvError=error;
    char message[2048]{};
    if(di_graph_prepare_prefill_scratch(&g,32,message,sizeof(message))){fprintf(stderr,"%s\n",message);return 1;}
    printf("PASS: Metal Q8 bytes, loads, zero/ties, invalid input/scale and production shader compilation\n");
    return 0;
}}
