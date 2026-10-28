#include "bridge.mm"

static void check(bool ok,const char *what) {
    if(!ok){fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}
}
static float from_bf(uint16_t bits) {uint32_t u=uint32_t(bits)<<16;float f;memcpy(&f,&u,4);return f;}
static void populate(QMatrix &m,int pattern) {
    auto *w=static_cast<uint8_t*>(m.packed.contents);
    auto *s=static_cast<uint16_t*>(m.scales.contents),*b=static_cast<uint16_t*>(m.biases.contents);
    for(size_t i=0;i<m.packed.length;i++)w[i]=uint8_t(i*73+(i/2048)*17+(i/64)*11);
    for(size_t i=0;i<m.scales.length/2;i++) {
        s[i]=pattern==1?0:pattern==6?0x8000:((i%3)==0?0xbb80:0x3b80);
        b[i]=pattern==2?0:pattern==6?0x8000:((i%2)?0xbf00:0x3f00);
    }
}
static void input_values(float *x,int pattern) {
    for(int c=0;c<2048;c++)x[c]=pattern==0?0:pattern==4?((c%2)?-64.f:64.f):
        pattern==5?(c%64==0||c%64==63?((c%2)?-.25f:.25f):0):std::sin(c*.031f)*.1f;
}
static std::vector<double> oracle(const QMatrix &m,const float *x) {
    auto *w=static_cast<const uint8_t*>(m.packed.contents);
    auto *s=static_cast<const uint16_t*>(m.scales.contents),*b=static_cast<const uint16_t*>(m.biases.contents);
    std::vector<double> result(m.rows);
    for(uint32_t row=0;row<m.rows;row++) {
        float total=0;
        for(uint32_t c=0;c<2048;c++) {
            volatile float product=float(w[size_t(row)*2048+c])*from_bf(s[size_t(row)*32+c/64]);
            volatile float weight=product+from_bf(b[size_t(row)*32+c/64]);
            volatile float term=weight*x[c];total+=term;
        }
        result[row]=total;
    }
    return result;
}
static uint32_t argmax(const float *y,uint32_t rows) {
    uint32_t selected=0;
    for(uint32_t r=0;r<rows;r++){check(std::isfinite(y[r]),"finite logits");if(y[r]>y[selected])selected=r;}
    return selected;
}
static void compare(const float *y,const std::vector<double> &reference,const char *label) {
    double error=0,norm=0,maxabs=0;
    for(size_t r=0;r<reference.size();r++) {
        check(std::isfinite(y[r]),"finite output");double d=y[r]-reference[r];
        error+=d*d;norm+=reference[r]*reference[r];maxabs=std::max(maxabs,std::abs(d));
    }
    double l2=norm?std::sqrt(error/norm):std::sqrt(error);
    printf("CORRECTNESS mode=%s rows=%zu relL2=%.9g maxAbs=%.9g\n",label,reference.size(),l2,maxabs);
    check(l2<1e-4&&maxabs<.01,"independent FP32 relL2 AND maxAbs gate");
}
static double execute(QContext *c,QMatrix &m,const LmHead &head,id<MTLBuffer> x,id<MTLBuffer> y,bool candidate) {
    char error[2048]{};uint32_t zero=0;double ms=0;
    auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
    check(command&&e,"encoder");
    if(candidate)lmhead_encode(head,e,x,y,m.rows);
    else q_encode(c,e,&m,x,&zero,1,false);
    [e endEncoding];check(q_complete(command,&ms,error,sizeof(error)),error);return ms;
}
static void small_operators(QContext *c) {
    char error[2048]{};
    for(uint32_t rows:{1u,3u,4u,5u,15u,16u,17u,33u,65u}) {
        auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,1,rows,2048,8,64,error,sizeof(error)));
        check(m,error);lmhead_affine8::Layout layout(rows);
        LmHead head;head.pipeline=lmhead_pipeline(c->device,error,sizeof(error));check(head.pipeline,error);
        head.packed=[c->device newBufferWithLength:layout.bytes options:MTLResourceStorageModeShared];
        auto y=[c->device newBufferWithLength:(rows+32)*4 options:MTLResourceStorageModeShared];
        check(head.packed&&y,"small operator buffers");
        for(int pattern=0;pattern<7;pattern++) {
            populate(*m,pattern);auto *x=static_cast<float*>(m->input.contents);input_values(x,pattern);
            check(lmhead_affine8::pack(rows,m->packed.contents,m->packed.length,m->scales.contents,m->scales.length,
                m->biases.contents,m->biases.length,head.packed.contents,head.packed.length),"small pack");
            auto expected=oracle(*m,x);std::vector<float> prior;
            auto *out=static_cast<float*>(y.contents);
            for(int repeat=0;repeat<2;repeat++) {
                std::fill(out,out+rows+32,NAN);execute(c,*m,head,m->input,y,true);
                compare(out,expected,"tile4");
                for(uint32_t r=rows;r<rows+32;r++)check(std::isnan(out[r]),"output tail sentinel");
                if(repeat==0)prior.assign(out,out+rows);
                else check(!memcmp(prior.data(),out,rows*4),"deterministic repeated output");
            }
            execute(c,*m,head,m->input,y,false);
            auto *original=static_cast<float*>(m->output.contents);
            compare(original,expected,"original");
            std::vector<double> originalReference(original,original+rows);compare(out,originalReference,"tile4-v-original");
            check(argmax(out,rows)==argmax(original,rows),"vocabulary order / greedy tie rule");
            printf("PASS GPU operator rows=%u pattern=%d bitIdentical=%d\n",rows,pattern,!memcmp(out,original,rows*4));
        }
        delete m;
    }
}
static void argmax_checks(QContext *c) {
    char error[2048]{};NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
    auto library=[c->device newLibraryWithSource:graph_source options:options error:&failure];check(library,failure.localizedDescription.UTF8String);
    auto blocks=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_argmax_blocks"] error:&failure];
    auto final=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_argmax_final"] error:&failure];check(blocks&&final,"argmax pipelines");
    GModel g{};g.compute=*c;g.ps[GArgBlocks]=blocks;g.ps[GArgFinal]=final;
    g.logits=g_buffer(&g,248320);g.argValues=g_buffer(&g,970);g.argIds=g_buffer(&g,970);g.nextToken=g_buffer(&g,1);
    check(g.logits&&g.argValues&&g.argIds&&g.nextToken,"argmax buffers");
    for(int pattern=0;pattern<5;pattern++) {
        auto *y=static_cast<float*>(g.logits.contents);std::fill(y,y+248320,-1.f);
        y[255]=y[256]=y[248319]=2;
        if(pattern==1)y[0]=2;
        if(pattern==2)y[12345]=NAN;
        if(pattern==3)y[12345]=INFINITY;
        if(pattern==4)y[12345]=-INFINITY;
        auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
        g_bind(e,g.logits,0);g_bind(e,g.argValues,1);g_bind(e,g.argIds,2);g_run(&g,e,GArgBlocks,970,32);g_barrier(e);
        g_bind(e,g.argValues,0);g_bind(e,g.argIds,1);g_bind(e,g.nextToken,2);g_run(&g,e,GArgFinal,1,32);[e endEncoding];
        check(q_complete(command,nullptr,error,sizeof(error)),error);
        check(*static_cast<uint32_t*>(g.nextToken.contents)==(pattern>=2?UINT32_MAX:pattern==1?0u:255u),"unchanged argmax ties/nonfinite rejection");
    }
    puts("PASS GPU argmax tie, vocabulary boundary, NaN and +/-Inf controls");
}
static void lifecycle_and_real(QContext *c,bool benchmark,int warmups,int trials,int pressureMiB) {
    char error[2048]{};
    auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,1,248320,2048,8,64,error,sizeof(error)));check(m,error);
    populate(*m,3);
    GModel g{};g.compute=*c;g.embedding=*m;g.classifier=*m;
    uint64_t before=c->device.currentAllocatedSize,info[7]{};info[6]=12345;
    check(di_graph_lmhead_info(nullptr,info,6)!=0&&di_graph_lmhead_info(&g,nullptr,6)!=0&&
        di_graph_lmhead_info(&g,info,5)!=0&&di_graph_lmhead_info(&g,info,7)!=0,"metadata negatives");
    check(di_graph_lmhead_configure(&g,-1,error,sizeof(error))!=0&&di_graph_lmhead_configure(&g,2,error,sizeof(error))!=0,"invalid native modes");
    check(di_graph_lmhead_configure(&g,0,error,sizeof(error))==0,error);
    check(c->device.currentAllocatedSize==before&&!g.lmHead.packed&&!g.lmHead.pipeline,"off has zero extra GPU allocation or pipeline");
    check(di_graph_lmhead_info(&g,info,6)==0&&info[0]==0&&info[1]==0&&info[2]==0&&info[3]==0&&info[4]==0&&info[5]==1&&info[6]==12345,"off metadata and alias");
    check(di_graph_lmhead_configure(&g,1,error,sizeof(error))!=0,"reject reconfiguration");
    g.lmHead=LmHead{};
    for(int invalid=0;invalid<5;invalid++) {
        auto saved=g.classifier;
        if(invalid==0)g.classifier.bits=5;
        if(invalid==1)g.classifier.columns=1024;
        if(invalid==2)g.classifier.rows--;
        if(invalid==3)g.classifier.experts=2;
        if(invalid==4)g.classifier.group=32;
        check(di_graph_lmhead_configure(&g,1,error,sizeof(error))!=0&&!g.lmHead.packed,"invalid geometry never falls back");
        g.classifier=saved;
    }
    g.layers[39].loaded=true;
    check(di_graph_lmhead_configure(&g,1,error,sizeof(error))!=0,"late configuration rejected");g.layers[39].loaded=false;
    uint64_t residentBefore=c->residency?c->residency->allocated():0;
    check(di_graph_lmhead_configure(&g,1,error,sizeof(error))==0,error);
    check(di_graph_lmhead_info(&g,info,6)==0&&info[0]==1&&info[1]==540344320&&info[2]>=info[1]&&
        info[3]==info[2]&&info[4]==0&&info[5]==1&&info[6]==12345,"candidate exact metadata");
    auto packed=g.lmHead.packed;
    for(auto &layer:g.layers)layer.loaded=true;
    g.position=71;check(di_graph_reset(&g)==0&&g.position==0&&g.lmHead.packed==packed,"reset retains immutable head");
    g.norm=g_buffer(&g,2048);g.logits=g_buffer(&g,248320+32);check(g.norm&&g.logits,"real operator buffers");
    input_values(static_cast<float*>(g.norm.contents),3);
    auto expected=oracle(*m,static_cast<float*>(g.norm.contents));
    execute(c,*m,g.lmHead,g.norm,g.logits,false);compare(static_cast<float*>(m->output.contents),expected,"original-real");
    auto *out=static_cast<float*>(g.logits.contents);std::fill(out,out+248352,NAN);
    auto command=[c->queue commandBuffer];auto encoder=[command computeCommandEncoder];g_classifier(&g,encoder);[encoder endEncoding];
    check(q_complete(command,nullptr,error,sizeof(error)),error);compare(out,expected,"tile4-real-dispatch");
    for(int r=248320;r<248352;r++)check(std::isnan(out[r]),"real output sentinel");
    check(argmax(out,248320)==argmax(static_cast<float*>(m->output.contents),248320),"real greedy semantics");
    printf("ALLOCATION sourcePayloadBytes=540344320 originalInputBytes=%lu originalOutputBytes=%lu candidatePayloadBytes=%llu candidateAllocatedBytes=%llu transientExtraPeakBytes=%llu candidateScratchBytes=0\n",
        (unsigned long)m->input.length,(unsigned long)m->output.length,(unsigned long long)info[1],
        (unsigned long long)info[2],(unsigned long long)info[3]);
    if(c->residency)check(c->residency->allocated()==residentBefore+info[2],"candidate residency bytes");
    if(benchmark) {
        static NSString *pressureSource=@R"metal(
#include <metal_stdlib>
using namespace metal;
kernel void pressure(device const uint *data [[buffer(0)]],device uint *result [[buffer(1)]],
 constant uint &count [[buffer(2)]],uint tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint sum=0;for(uint i=tid;i<count;i+=32768)sum+=data[i];
 sum=simd_sum(sum);if(lane==0)result[tid/32]=sum;
}
)metal";
        NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
        auto library=[c->device newLibraryWithSource:pressureSource options:options error:&failure];
        auto pressure=library?[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"pressure"] error:&failure]:nil;
        auto data=[c->device newBufferWithLength:size_t(pressureMiB)*1024*1024 options:MTLResourceStorageModeShared];
        auto result=[c->device newBufferWithLength:4096 options:MTLResourceStorageModeShared];check(pressure&&data&&result,"cache-pressure resources");
        uint32_t count=uint32_t(data.length/4);auto *words=static_cast<uint32_t*>(data.contents);
        for(uint32_t i=0;i<count;i++)words[i]=i*2654435761u;
        printf("BENCH geometry=N248320-K2048-affine8-group64 fixture=synthetic-v1 baseline=vector-row4 candidate=tile4-block256-lane32-octet8-v1 warmups=%d trials=%d pressureBytes=%lu pressureScratchBytes=%lu currentMetalBytes=%llu clock=command-GPUStartTime-GPUEndTime dispatchesPerSample=1\n",
            warmups,trials,(unsigned long)data.length,(unsigned long)result.length,(unsigned long long)c->device.currentAllocatedSize);
        for(int cache=0;cache<2;cache++)for(int round=-warmups;round<trials;round++)for(int arm=0;arm<2;arm++) {
            bool candidate=((round+warmups)%2)^arm;
            double pressureMs=0;
            if(cache) {
                auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:pressure];
                g_bind(e,data,0);g_bind(e,result,1);[e setBytes:&count length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
                check(q_complete(command,&pressureMs,error,sizeof(error)),error);
            }
            double ms=execute(c,*m,g.lmHead,g.norm,g.logits,candidate);
            check(ms>0&&std::isfinite(ms),"valid GPU clock");
            printf("SAMPLE cache=%s phase=%s round=%d order=%d mode=%s gpuMs=%.9f pressureGpuMs=%.9f\n",
                cache?"pressure":"warm",round<0?"warmup":"formal",round,arm,candidate?"tile4":"original",ms,pressureMs);
            compare(static_cast<float*>((candidate?g.logits:m->output).contents),expected,candidate?"tile4-sample":"original-sample");
        }
    }
    auto *retained=new GModel{};retained->lmHead=g.lmHead;
    g.lmHead=LmHead{};packed=nil;
    if(c->residency)check(c->residency->allocated()==residentBefore+info[2],"copied graph retains candidate lease");
    di_graph_destroy(retained);
    check(!c->residency||c->residency->allocated()==residentBefore,"destroy releases candidate residency lease");
    delete m;
    puts("PASS lifecycle, off allocation, explicit selector, real-dimension oracle and resource checks");
}
int main(int argc,char **argv){@autoreleasepool {
    check(argc>=2,"usage: --compile-only | --gpu | --bench [warmups=2 trials=5 pressureMiB=128]");
    if(!strcmp(argv[1],"--compile-only")) {
        check(argc==2,"compile arguments");char error[2048]{};auto device=MTLCreateSystemDefaultDevice();check(device,"Metal device");
        check(lmhead_pipeline(device,error,sizeof(error)),error);
        puts("PASS Metal tile4 library/pipeline compilation; zero GPU submissions");return 0;
    }
    bool benchmark=!strcmp(argv[1],"--bench");
    check(benchmark||!strcmp(argv[1],"--gpu"),"unknown mode");
    check((benchmark&&(argc==2||argc==5))||(!benchmark&&argc==2),"driver arguments");
    int warmups=2,trials=5,pressureMiB=128;
    if(argc==5) {
        auto number=[](const char *text){char *end=nullptr;long v=strtol(text,&end,10);check(*text&&end&&!*end&&v>=0&&v<=256,"integer argument");return int(v);};
        warmups=number(argv[2]);trials=number(argv[3]);pressureMiB=number(argv[4]);
    }
    check(warmups>=1&&warmups<=5&&trials>=1&&trials<=10&&pressureMiB>=64&&pressureMiB<=256,"bounded benchmark arguments");
    setenv("DEEPINFER_METAL_VECTOR_QUANT","1",1);setenv("DEEPINFER_METAL_ROW_TILE","4",1);setenv("DEEPINFER_METAL_GENERIC","0",1);
    char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create_configured(benchmark?0:1,0,error,sizeof(error)));check(c,error);
    if(!benchmark)di_quant_residency_prepare(c);
    printf("DEVICE name=%s baselineVector=1 baselineRowTile=4 residency=%s\n",c->device.name.UTF8String,benchmark?"off":"on");
    if(!benchmark){small_operators(c);argmax_checks(c);}
    lifecycle_and_real(c,benchmark,warmups,trials,pressureMiB);
    delete c;return 0;
}}
