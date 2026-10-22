// Hardware checks for graph primitives with independent scalar oracles.
#include "bridge.mm"
#include <numeric>
#include <vector>
int main(){@autoreleasepool {
    setenv("DEEPINFER_METAL_PREFILL_CHUNK","128",1);
    char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));
    if(!c){fprintf(stderr,"%s\n",error);return 1;}
    NSError *failure=nil;MTLCompileOptions *options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    auto library=[c->device newLibraryWithSource:graph_source options:options error:&failure];
    if(!library){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);delete c;return 1;}
    auto top=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_top8"] error:&failure];
    auto norm=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_norm"] error:&failure];
    auto argBlocks=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_argmax_blocks"] error:&failure];
    auto argFinal=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_argmax_final"] error:&failure];
    if(!top||!norm||!argBlocks||!argFinal){delete c;return 1;}
    auto buffer=[&](int n){return [c->device newBufferWithLength:n*4 options:MTLResourceStorageModeShared];};
    auto input=buffer(2048),weight=buffer(2048),output=buffer(2048),ids=buffer(8),prob=buffer(8);
    for(int test=0;test<5;test++) {
        auto *values=static_cast<float*>(input.contents);
        for(int i=0;i<256;i++)values[i]=test==0?0:test==1?float(i)*.01f:test==2?(i%32==0||i==255?3:-1):
            test==3?float(i-128)*100.0f:float((i*7919)%97-48)*.07f;
        std::array<int,256> expected;std::iota(expected.begin(),expected.end(),0);
        std::sort(expected.begin(),expected.end(),[&](int a,int b){return values[a]>values[b]||(values[a]==values[b]&&a<b);});
        auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:top];
        [e setBuffer:input offset:0 atIndex:0];[e setBuffer:ids offset:0 atIndex:1];[e setBuffer:prob offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
        if(!q_complete(command,nullptr,error,sizeof(error))){delete c;return 1;}
        auto *actual=static_cast<const uint32_t*>(ids.contents);auto *p=static_cast<const float*>(prob.contents);double total=0;
        for(int j=0;j<8;j++)total+=std::exp(double(values[expected[j]])-values[expected[0]]);
        for(int j=0;j<8;j++)if(actual[j]!=(uint32_t)expected[j]||
            std::abs(p[j]-std::exp(double(values[expected[j]])-values[expected[0]])/total)>1e-6){fprintf(stderr,"Top8 oracle failed, case=%d slot=%d\n",test,j);delete c;return 1;}
    }
    for(int test=0;test<2;test++) {
        auto *x=static_cast<float*>(input.contents),*w=static_cast<float*>(weight.contents);double sum=0;
        for(int i=0;i<2048;i++){x[i]=test?std::sin(float(i)*.071f):0;w[i]=1+std::cos(float(i)*.013f)*.1f;sum+=double(x[i])*x[i];}
        GStep step{0,0,0,1e-6f,1e7f};auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
        [e setComputePipelineState:norm];[e setBuffer:input offset:0 atIndex:0];[e setBuffer:weight offset:0 atIndex:1];
        [e setBuffer:output offset:0 atIndex:2];[e setBytes:&step length:sizeof(step) atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
        if(!q_complete(command,nullptr,error,sizeof(error))){delete c;return 1;}
        auto *y=static_cast<const float*>(output.contents);double scale=1/std::sqrt(sum/2048+1e-6f);
        for(int i=0;i<2048;i++)if(std::abs(y[i]-double(x[i])*scale*w[i])>2e-5){fprintf(stderr,"RMS norm oracle failed\n");delete c;return 1;}
    }
    auto vocabulary=buffer(248320),argValues=buffer(970),argIds=buffer(970),next=buffer(1);
    for(int test=0;test<3;test++) {
        auto *x=static_cast<float*>(vocabulary.contents);for(int i=0;i<248320;i++)x[i]=float(i)*.00001f;
        uint32_t expected=248319;
        if(test==1){x[100]=x[200]=10;expected=100;}
        if(test==2){x[0]=std::nanf("");expected=0xffffffffu;}
        auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
        [e setComputePipelineState:argBlocks];[e setBuffer:vocabulary offset:0 atIndex:0];
        [e setBuffer:argValues offset:0 atIndex:1];[e setBuffer:argIds offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(970,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [e memoryBarrierWithScope:MTLBarrierScopeBuffers];[e setComputePipelineState:argFinal];
        [e setBuffer:argValues offset:0 atIndex:0];[e setBuffer:argIds offset:0 atIndex:1];[e setBuffer:next offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[e endEncoding];
        if(!q_complete(command,nullptr,error,sizeof(error))||static_cast<uint32_t*>(next.contents)[0]!=expected){fprintf(stderr,"Argmax oracle failed\n");delete c;return 1;}
    }
    {
        GModel g{};g.compute=*c;
        if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"Prefill pipeline: %s\n",error);delete c;return 1;}
        constexpr int rows=129,columns=256;
        auto bx=buffer(128*columns),by=buffer(128*rows);
        auto *x=static_cast<float*>(bx.contents);
        for(int i=0;i<128*columns;i++)x[i]=std::sin(i*.017f)*.1f;
        for(bool matrix:{false,true}) {g.prefill->matrix=matrix;
        for(int bits:{5,8}) {
            auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,1,rows,columns,bits,64,error,sizeof(error)));
            if(!m){delete c;return 1;}
            auto *packed=static_cast<uint32_t*>(m->packed.contents);memset(packed,0,m->packed.length);
            auto *scales=static_cast<uint16_t*>(m->scales.contents),*biases=static_cast<uint16_t*>(m->biases.contents);
            for(int r=0;r<rows;r++) {
                for(int a=0;a<4;a++){float scale=(a+1)/512.0f,bias=(r+a-3)/32.0f;uint32_t b;
                    memcpy(&b,&scale,4);scales[r*4+a]=b>>16;memcpy(&b,&bias,4);biases[r*4+a]=b>>16;}
                for(int col=0;col<columns;col++){uint32_t q=(r*7+col*3)&((1<<bits)-1);int bit=col*bits,at=r*(columns*bits/32)+bit/32,shift=bit%32;
                    packed[at]|=q<<shift;if(shift+bits>32)packed[at+1]|=q>>(32-shift);}
            }
            for(int tokens:{1,3,4,5,9,16,17,24,32,65,128}) {
                PShape shape{0,0,(uint32_t)tokens,0,0,0,1e-6f};
                auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
                p_gemv(&g,e,*m,bx,by,shape);[e endEncoding];
                if(!q_complete(command,nullptr,error,sizeof(error))){delete m;delete c;return 1;}
                auto *y=static_cast<const float*>(by.contents);
                for(int t=0;t<tokens;t++)for(int r=0;r<rows;r++){double expected=0;
                    for(int col=0;col<columns;col++){int a=col/64,q=(r*7+col*3)&((1<<bits)-1);
                        expected+=(double(q)*(a+1)/512.0+(r+a-3)/32.0)*x[t*columns+col];}
                    if(std::abs(expected-y[t*rows+r])>.0003){fprintf(stderr,"Batched projection oracle failed bits=%d tokens=%d row=%d token=%d\n",bits,tokens,r,t);delete m;delete c;return 1;}
                }
            }
            delete m;
        }
        }
        auto state=buffer(32*128*128),history=buffer(3*8192),raw=buffer(128*8192);
        auto *conv=static_cast<float*>(g.prefill->conv.contents),*a=static_cast<float*>(g.prefill->a.contents),*b=static_cast<float*>(g.prefill->b.contents);
        auto *rawValues=static_cast<float*>(raw.contents);
        for(int i=0;i<128*8192;i++){rawValues[i]=std::sin(i*.007f);conv[i]=std::cos(i*.013f)*.03f;}
        for(int i=0;i<128*32;i++){a[i]=.9f+(i%7)*.01f;b[i]=.4f+(i%5)*.02f;}
        for(int tokens:{1,2,3,32,64,128}) {
            auto *stored=static_cast<float*>(state.contents),*h=static_cast<float*>(history.contents);
            std::vector<double> oracle(32*128*128);
            for(int i=0;i<32*128*128;i++)oracle[i]=stored[i]=std::sin(i*.003f)*.001f;
            for(int i=0;i<3*8192;i++)h[i]=float(i)*.0001f;
            PShape shape{0,0,(uint32_t)tokens,0,0,0,1e-6f};
            auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
            g_bind(e,state,0);g_bind(e,g.prefill->conv,1);g_bind(e,g.prefill->a,2);g_bind(e,g.prefill->b,3);g_bind(e,g.prefill->attention,4);g_bind(e,history,5);g_bind(e,raw,6);
            [e setBytes:&shape length:sizeof(shape) atIndex:7];p_run(&g,e,PDelta,MTLSizeMake(2048,1,1),64);[e endEncoding];
            if(!q_complete(command,nullptr,error,sizeof(error))){delete c;return 1;}
            auto *actual=static_cast<float*>(g.prefill->attention.contents);
            for(int t=0;t<tokens;t++)for(int row=0;row<4096;row++) {
                int head=row/128,kh=head/2;double prediction=0,result=0;
                for(int d=0;d<128;d++){oracle[row*128+d]*=a[t*32+head];prediction+=oracle[row*128+d]*conv[t*8192+2048+kh*128+d];}
                double correction=(conv[t*8192+4096+row]-prediction)*b[t*32+head];
                for(int d=0;d<128;d++){oracle[row*128+d]+=conv[t*8192+2048+kh*128+d]*correction;result+=oracle[row*128+d]*conv[t*8192+kh*128+d];}
                if(std::abs(result-actual[t*4096+row])>1e-6){fprintf(stderr,"Batched recurrence output oracle failed\n");delete c;return 1;}
            }
            for(int i=0;i<32*128*128;i++)if(std::abs(oracle[i]-stored[i])>1e-6){fprintf(stderr,"Batched recurrence state oracle failed\n");delete c;return 1;}
            for(int j=0;j<3;j++)for(int d=0;d<8192;d++){int at=j+tokens;float expected=at<3?float(at*8192+d)*.0001f:rawValues[(at-3)*8192+d];
                if(h[j*8192+d]!=expected){fprintf(stderr,"Convolution history tail oracle failed\n");delete c;return 1;}}
        }
    }
    delete c;printf("PASS: 60 hardware checks (10 graph primitives, 44 batched projections including 65/128-token tiles, 6 recurrence/state/history cases)\n");return 0;
}}
