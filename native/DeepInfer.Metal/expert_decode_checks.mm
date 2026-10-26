// Independent affine8 scalar oracle and bounded decode-kernel timing for #92.
#include "bridge.mm"
#include <vector>
#include <random>
#include <numeric>
#include <cstdio>

static void require(bool value,const char *message){if(!value){fprintf(stderr,"FAIL %s\n",message);exit(1);}}
static float bf(uint16_t bits){uint32_t u=uint32_t(bits)<<16;float f;memcpy(&f,&u,4);return f;}
static void fill(QMatrix *m,uint32_t seed){
    std::mt19937 random(seed);
    auto *w=static_cast<uint32_t*>(m->packed.contents);
    for(size_t i=0;i<m->packed.length/4;i++)w[i]=random();
    auto *s=static_cast<uint16_t*>(m->scales.contents),*b=static_cast<uint16_t*>(m->biases.contents);
    for(size_t i=0;i<m->scales.length/2;i++){
        // Signed, exactly representable BF16 coefficients; no shared decoder.
        s[i]=uint16_t((i%3==0?0xbb80:0x3b80)+(i%7));b[i]=uint16_t((i%2?0xbe80:0x3e80)+(i%5));
    }
}
static std::vector<float> reference(QMatrix *m,const std::vector<float>& x,const uint32_t *ids,bool batched){
    auto *w=static_cast<const uint8_t*>(m->packed.contents);
    auto *s=static_cast<const uint16_t*>(m->scales.contents),*b=static_cast<const uint16_t*>(m->biases.contents);
    std::vector<float> out(8*m->rows);
    for(uint32_t slot=0;slot<8;slot++)for(uint32_t row=0;row<m->rows;row++){
        float sum=0;
        size_t base=(size_t(ids[slot])*m->rows+row)*m->columns;
        for(uint32_t k=0;k<m->columns;k++){
            size_t p=(base+k)/64;
            float weight=float(w[base+k])*bf(s[p])+bf(b[p]);
            sum+=weight*x[(batched?slot*m->columns:0)+k];
        }
        out[slot*m->rows+row]=sum;
    }
    return out;
}
static void compare(const std::vector<float>& want,const float *got,const char *label,int lanes){
    double diff=0,norm=0;float maximum=0;
    for(size_t i=0;i<want.size();i++){
        require(std::isfinite(got[i]),"nonfinite output");
        double d=double(got[i])-want[i];diff+=d*d;norm+=double(want[i])*want[i];maximum=std::max(maximum,float(fabs(d)));
    }
    double relative=sqrt(diff/std::max(norm,1e-30));
    printf("{\"kind\":\"oracle\",\"label\":\"%s\",\"lanes\":%d,\"relativeL2\":%.12g,\"maxAbs\":%.12g}\n",label,lanes,relative,maximum);
    require(relative<1e-4&&maximum<.01,"oracle tolerance");
}
int main(int argc,char **argv){@autoreleasepool {
    bool timing=argc>1&&!strcmp(argv[1],"--timing");
    char error[4096]{};
    if(!timing){
        setenv("DEEPINFER_METAL_EXPERT_LANES","7",1);
        require(!di_quant_create(error,sizeof(error)),"invalid selector accepted");
        setenv("DEEPINFER_METAL_EXPERT_LANES","4",1);setenv("DEEPINFER_METAL_GENERIC","1",1);
        require(!di_quant_create(error,sizeof(error)),"generic conflict accepted");unsetenv("DEEPINFER_METAL_GENERIC");
    }
    std::vector<int> laneOrder=argc>2&&!strcmp(argv[2],"--reverse")?std::vector<int>{16,8,4,0}:std::vector<int>{0,4,8,16};
    for(int lanes:laneOrder){
        std::string flag=std::to_string(lanes);setenv("DEEPINFER_METAL_EXPERT_LANES",flag.c_str(),1);
        auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c){fprintf(stderr,"%s\n",error);return 1;}
        require(di_quant_expert_lanes(c)==lanes,"native selector mismatch");
        printf("{\"kind\":\"device\",\"lanes\":%d,\"name\":\"%s\"}\n",lanes,c->device.name.UTF8String);
        for(int mode=0;mode<(timing?2:6);mode++){
            uint32_t rows=mode%2?2048:512,columns=mode%2?512:2048;
            if(!timing&&mode<4){rows=mode%2?137:33;columns=mode<2?64:256;}
            uint32_t expertCount=timing?256:17;
            auto matrix=[&](uint32_t seed){auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,expertCount,rows,columns,8,64,error,sizeof(error)));require(m,"matrix allocation");fill(m,seed);return m;};
            auto *gate=matrix(100+mode),*up=matrix(200+mode);
            uint32_t ids[8]={expertCount-1,0,7,3,12,1,8,2};
            auto idBuffer=[c->device newBufferWithBytes:ids length:sizeof(ids) options:MTLResourceStorageModeShared];
            std::vector<float> x(columns*8);
            auto input=[c->device newBufferWithLength:x.size()*4 options:MTLResourceStorageModeShared];
            auto output=[c->device newBufferWithLength:(8*rows+32)*4 options:MTLResourceStorageModeShared];
            for(int fused=0;fused<2;fused++){
                // down uses distinct per-slot inputs; gate/up shares one vector.
                if((timing||mode>=4)&&fused==(mode%2))continue;
                for(int pattern=0;pattern<(timing?1:4);pattern++){
                    for(size_t i=0;i<x.size();i++)x[i]=pattern==3?std::sin(float(i)*.017f)*.125f:pattern==1?0:pattern==2?(i%2?.25f:-.25f):float(int((i*37+pattern*19)%101)-50)/4096;
                    memcpy(input.contents,x.data(),x.size()*4);
                    auto *guard=static_cast<float*>(output.contents);std::fill(guard,guard+8*rows+32,123456.0f);
                    auto encode=[&](id<MTLComputeCommandEncoder> e){
                        struct {uint32_t experts,rows,columns,bits,group,count,batched;} shape{expertCount,rows,columns,8,64,8,fused?0u:1u};
                        if(fused){
                            g_bind(e,gate->packed,0);g_bind(e,gate->scales,1);g_bind(e,gate->biases,2);g_bind(e,up->packed,3);g_bind(e,up->scales,4);g_bind(e,up->biases,5);g_bind(e,input,6);g_bind(e,output,7);[e setBytes:&shape length:sizeof(shape) atIndex:8];[e setBytes:ids length:sizeof(ids) atIndex:9];
                            [e setComputePipelineState:lanes?c->expertGateSmall[lanes==4?0:lanes==8?1:2]:c->gateUp8];
                        }else{
                            g_bind(e,gate->packed,0);g_bind(e,gate->scales,1);g_bind(e,gate->biases,2);g_bind(e,input,3);g_bind(e,output,4);[e setBytes:&shape length:sizeof(shape) atIndex:5];[e setBytes:ids length:sizeof(ids) atIndex:6];
                            [e setComputePipelineState:lanes?c->expertDownSmall[lanes==4?0:lanes==8?1:2]:c->tiled8];
                        }
                        uint32_t tile=lanes?128/lanes:4;
                        [e dispatchThreadgroups:MTLSizeMake((rows+tile-1)/tile,8,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];
                    };
                    auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];encode(e);[e endEncoding];require(q_complete(command,nullptr,error,sizeof(error)),error);
                    if(!timing){
                        auto want=reference(gate,x,ids,!fused);
                        if(fused){auto other=reference(up,x,ids,false);for(size_t i=0;i<want.size();i++)want[i]=(want[i]/(1+expf(-want[i])))*other[i];}
                        compare(want,guard,fused?"gate-up":"down",lanes);
                        for(size_t i=8*rows;i<8*rows+32;i++)require(guard[i]==123456.0f,"output sentinel changed");
                        // Exercise the graph dispatch helpers, including shared norm binding and per-slot inputs.
                        GModel graph{};graph.compute=*c;graph.ids=idBuffer;graph.norm=input;
                        std::fill(guard,guard+8*rows+32,123456.0f);
                        auto check=[c->queue commandBuffer];auto ce=[check computeCommandEncoder];
                        if(fused)g_gate_up(&graph,ce,*gate,*up,output,true);else g_gemv(&graph,ce,*gate,input,output,true);
                        [ce endEncoding];require(q_complete(check,nullptr,error,sizeof(error)),error);
                        compare(want,guard,fused?"graph-gate-up":"graph-down",lanes);
                        for(size_t i=8*rows;i<8*rows+32;i++)require(guard[i]==123456.0f,"graph output sentinel changed");
                    }else{
                        for(int rotate=0;rotate<2;rotate++)for(int trial=-2;trial<7;trial++){
                            auto cb=[c->queue commandBuffer];auto enc=[cb computeCommandEncoder];
                            for(int iteration=0;iteration<64;iteration++){
                                if(rotate){uint32_t selected[8];for(int slot=0;slot<8;slot++)selected[slot]=(iteration*8+slot)%expertCount;
                                    // setBytes is snapshotted by Metal per dispatch; do not mutate a live shared ID buffer.
                                    memcpy(ids,selected,sizeof(ids));}
                                else {uint32_t selected[8]={255,0,7,3,12,1,8,2};memcpy(ids,selected,sizeof(ids));}
                                encode(enc);
                                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            }
                            [enc endEncoding];double ms=0;require(q_complete(cb,&ms,error,sizeof(error)),error);
                            printf("{\"kind\":\"timing\",\"lanes\":%d,\"projection\":\"%s\",\"rotating\":%s,\"trial\":%d,\"iterations\":64,\"gpuMs\":%.9g}\n",lanes,fused?"gate-up":"down",rotate?"true":"false",trial,ms);
                        }
                    }
                }
            }
            di_quant_matrix_destroy(gate);di_quant_matrix_destroy(up);
        }
        di_quant_destroy(c);
    }
    return 0;
}}
