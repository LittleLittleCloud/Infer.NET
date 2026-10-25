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
    for(auto pair:{std::make_pair(GScores,@"graph_attention_scores"),std::make_pair(GAttention,@"graph_attention_output"),std::make_pair(GAttentionCombine,@"graph_attention_combine")}){
        g.ps[pair.first]=[g.compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:pair.second] error:&failure];
        if(!g.ps[pair.first])return 1;
    }
    // Restore valid fixtures after invalid-append arms, then attend to their decoded values.
    memcpy(y.contents,expected.data(),expected.size());
    auto q=g_buffer(&g,3*4096),raw=g_buffer(&g,3*8192),out=g_buffer(&g,3*4096),scores=g_buffer(&g,3*16*g.capacity);
    g.attentionParts=g_buffer(&g,3*16*258);memset(raw.contents,0,raw.length);
    for(int i=0;i<3*4096;i++)static_cast<float*>(q.contents)[i]=float(int(i%29)-14)*.003f;
    auto decoded=[&](int i){auto *b=reinterpret_cast<const unsigned char*>(expected.data())+(i/32)*34;__fp16 h;memcpy(&h,b,2);return float(h)*float(static_cast<signed char>(b[2+i%32]));};
    for(int mode=0;mode<3;mode++){
        g.position=mode==2?2:4;const int count=mode==2?3:1;
        auto cb=[g.compute.queue commandBuffer];auto e=[cb computeCommandEncoder];
        if(mode==0){GStep shape{g.position,0,0,1e-6f,1e7f,g.capacity};
            g_bind(e,q,0);g_bind(e,y,1);g_bind(e,scores,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];[e setComputePipelineState:g.ps[GScores]];[e dispatchThreadgroups:MTLSizeMake(2,16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g_barrier(e);
            g_bind(e,scores,0);g_bind(e,y,1);g_bind(e,raw,2);g_bind(e,out,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];g_run(&g,e,GAttention,16,256);
        }else if(mode==1)p_decode_attention(&g,e,q,y,y,raw,out);
        else p_fused_attention(&g,e,q,y,y,raw,out,count);
        [e endEncoding];if(!q_complete(cb,nullptr,message,sizeof(message)))return 1;
        double squared=0,norm=0,worst=0;
        for(int t=0;t<count;t++)for(int h=0;h<16;h++){
            std::vector<double> weights(g.position+t+1);double maximum=-INFINITY;
            for(int k=0;k<int(weights.size());k++){double dot=0;for(int d=0;d<256;d++)dot+=double(static_cast<float*>(q.contents)[t*4096+h*256+d])*decoded(k*512+(h/8)*256+d);weights[k]=dot/16;maximum=std::max(maximum,weights[k]);}
            double total=0;for(auto &v:weights){v=exp(v-maximum);total+=v;}
            for(int d=0;d<256;d++){double ref=0;for(int k=0;k<int(weights.size());k++)ref+=weights[k]*decoded(k*512+(h/8)*256+d);ref=ref/total*.5;
                double diff=static_cast<float*>(out.contents)[t*4096+h*256+d]-ref;if(!std::isfinite(diff))return 1;squared+=diff*diff;norm+=ref*ref;worst=std::max(worst,std::abs(diff));}
        }
        if(worst>=.01||sqrt(squared/std::max(norm,1e-30))>=1e-4){fprintf(stderr,"Q8 attention oracle failed mode=%d max=%g relative=%g\n",mode,worst,sqrt(squared/norm));return 1;}
    }
    // Import exact bytes, reject invalid blocks and preserve history across growth.
    g.position=5;g.layers[3].loaded=true;g.layers[3].keys=g_kv_buffer(&g,2048*512);g.layers[3].values=g_kv_buffer(&g,2048*512);
    g.splitActive=true;g.splitMissing=1ull<<6;
    if(di_graph_prefix_kv_range(&g,3,0,0,5,5,expected.data(),int(expected.size())))return 1;
    if(memcmp(g.layers[3].keys.contents,expected.data(),expected.size()))return 1;
    g.splitMissing=1ull<<7;auto corrupt=expected;corrupt[0]=0;corrupt[1]=char(0x7c);
    if(!di_graph_prefix_kv_range(&g,3,1,0,5,5,corrupt.data(),int(corrupt.size())))return 1;
    if(di_graph_prefix_kv_range(&g,3,1,0,5,5,expected.data(),int(expected.size())))return 1;
    if(di_graph_grow(&g,4097,message,sizeof(message))||g.capacity!=6144||memcmp(g.layers[3].keys.contents,expected.data(),expected.size()))return 1;
    int cap=0;uint64_t kv=0,allocated=0;if(di_graph_stats(&g,&cap,&kv,&allocated)||kv!=uint64_t(6144)*10880)return 1;
    printf("PASS: Metal Q8 bytes, loads, invalid inputs, three attention paths, import rejection and growth\n");
    return 0;
}}
