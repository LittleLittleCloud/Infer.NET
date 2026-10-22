// Full gate/up -> BF16 -> down composition: direct layout vs original pack.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_GEMM","bf16-nax",1);setenv("DEEPINFER_METAL_PREFILL_GEMM_IMPL","hybrid",1);
 setenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL","packed",1);setenv("DEEPINFER_METAL_PREFILL_EXPERT_FLOW","direct",1);
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 std::vector<QMatrix*> m;
 for(auto shape:{std::pair<int,int>{128,256},{128,256},{129,128}}){
  auto *q=static_cast<QMatrix*>(di_quant_matrix_create(c,256,shape.first,shape.second,5,64,error,sizeof(error)));if(!q)return 1;m.push_back(q);
  auto *w=static_cast<uint32_t*>(q->packed.contents);for(NSUInteger i=0;i<q->packed.length/4;i++)w[i]=uint32_t(i*2654435761u+m.size()*7919u);
  auto *sc=static_cast<uint16_t*>(q->scales.contents),*bi=static_cast<uint16_t*>(q->biases.contents);
  for(NSUInteger i=0;i<q->scales.length/2;i++){sc[i]=0x3b00;bi[i]=0xbd00;}
 }
 auto output=g_buffer(&g,2048*8*129+32);auto *x=static_cast<float*>(g.prefill->norm.contents);
 for(int i=0;i<2048*256;i++)x[i]=std::sin(i*.021f)*.2f;
 auto bf=[](float v){uint32_t u;memcpy(&u,&v,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);};
 int checks=0;uint64_t hash=14695981039346656037ull;
 for(int count:{32,64,65,127,129,513,2048})for(int pattern:{0,1,2}){
  auto *ids=static_cast<uint32_t*>(g.prefill->ids.contents);
  for(int t=0;t<count;t++)for(int slot=0;slot<8;slot++)ids[t*8+slot]=pattern==0?slot:pattern==1?(t*7+slot*13)%256:(t/5*11+slot*19)%256;
  std::vector<float> reference,activation;int referenceDispatch=0;
  for(bool direct:{false,true}){
   g.prefill->directExperts=direct;g.dispatches=0;static_cast<uint32_t*>(g.prefill->groupedTileCount.contents)[0]=0;
   auto *y=static_cast<float*>(output.contents);std::fill(y,y+output.length/4,NAN);
   PShape s{128,256,(uint32_t)count,0,0,0,1e-6f,1e7f,2048};
   auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
   if(count>=64)p_build_groups(&g,e,s,128,129);
   p_gate(&g,e,*m[0],*m[1],g.prefill->expertAct,s,true);g_barrier(e);
   [e endEncoding];if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
   if(!direct)activation.assign(static_cast<float*>(g.prefill->expertAct.contents),static_cast<float*>(g.prefill->expertAct.contents)+count*8*128);
   if(direct&&count>=64){
    auto *tiles=static_cast<uint32_t*>(g.prefill->groupedTiles.contents),*counts=static_cast<uint32_t*>(g.prefill->groupedCounts.contents),*routes=static_cast<uint32_t*>(g.prefill->groupedRoutes.contents);
    auto *packed=static_cast<uint16_t*>(g.prefill->expertAct.contents);uint32_t tileCount=*static_cast<uint32_t*>(g.prefill->groupedTileCount.contents);
    for(uint32_t tile=0;tile<tileCount;tile++)for(uint32_t t=0;t<std::min(64u,counts[tiles[tile*2]]-tiles[tile*2+1]);t++)for(int r=0;r<128;r++){
     auto route=routes[tiles[tile*2]*count+tiles[tile*2+1]+t];
     if(packed[(tile*64+t)*128+r]!=bf(activation[route*128+r])){fprintf(stderr,"BF16 layout mismatch n=%d tile=%u t=%u row=%d\n",count,tile,t,r);return 1;}
    }
   }
   command=[c->queue commandBuffer];e=[command computeCommandEncoder];
   p_gemv(&g,e,*m[2],g.prefill->expertAct,output,s,true);[e endEncoding];
   if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
   for(int i=0;i<count*8*129;i++)if(!std::isfinite(y[i]))return 1;
   for(int i=count*8*129;i<count*8*129+32;i++)if(!std::isnan(y[i]))return 1;
   if(!direct){reference.assign(y,y+count*8*129);referenceDispatch=g.dispatches;}
   else if(memcmp(reference.data(),y,reference.size()*4)||g.dispatches!=referenceDispatch-(count>=64?1:0)){
    fprintf(stderr,"Composition/count mismatch n=%d pattern=%d dispatch=%d reference=%d\n",count,pattern,g.dispatches,referenceDispatch);return 1;}
  }
  for(float v:reference){uint32_t bits;memcpy(&bits,&v,4);for(int b=0;b<4;b++){hash^=(bits>>(8*b))&255;hash*=1099511628211ull;}}
  checks++;
 }
 for(auto *q:m)delete q;
 delete c;printf("Output FNV64: %016llx\n",(unsigned long long)hash);printf("PASS: %d expert-flow checks; BF16 layout and final FP32 outputs bit-identical, hot/scattered/skewed routes, 32-token fallback, tails and dispatch reduction\n",checks);
}}
