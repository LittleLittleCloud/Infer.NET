// Complete expert gate/pack/down chain, synthetic routing; not model throughput.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_GEMM","bf16-nax",1);setenv("DEEPINFER_METAL_PREFILL_GEMM_IMPL","hybrid",1);
 setenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL","packed",1);setenv("DEEPINFER_METAL_PREFILL_EXPERT_FLOW","direct",1);
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 std::vector<QMatrix*> m;
 for(auto shape:{std::pair<int,int>{512,2048},{512,2048},{2048,512}}){
  auto *q=static_cast<QMatrix*>(di_quant_matrix_create(c,256,shape.first,shape.second,5,64,error,sizeof(error)));if(!q)return 1;m.push_back(q);
  auto *w=static_cast<uint32_t*>(q->packed.contents);for(NSUInteger i=0;i<q->packed.length/4;i++)w[i]=uint32_t(i*2654435761u+m.size()*7919u);
  auto *sc=static_cast<uint16_t*>(q->scales.contents),*bi=static_cast<uint16_t*>(q->biases.contents);
  for(NSUInteger i=0;i<q->scales.length/2;i++){sc[i]=0x3b00;bi[i]=0xbd00;}
 }
 auto *x=static_cast<float*>(g.prefill->norm.contents);for(int i=0;i<2048*2048;i++)x[i]=std::sin(i*.021f)*.02f;
 for(int pattern:{0,1,2}){
  auto *ids=static_cast<uint32_t*>(g.prefill->ids.contents);for(int t=0;t<2048;t++)for(int slot=0;slot<8;slot++)ids[t*8+slot]=pattern==0?slot:pattern==1?(t*7+slot*13)%256:(t/5*11+slot*19)%256;
  for(bool direct:{false,true}){
   g.prefill->directExperts=direct;double sum=0;
   for(int trial=0;trial<8;trial++){
    static_cast<uint32_t*>(g.prefill->groupedTileCount.contents)[0]=0;g.dispatches=0;
    PShape s{512,2048,2048,0,0,0,1e-6f,1e7f,2048};auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
    p_build_groups(&g,e,s,512,2048);p_gate(&g,e,*m[0],*m[1],g.prefill->expertAct,s,true);g_barrier(e);
    p_gemv(&g,e,*m[2],g.prefill->expertAct,g.prefill->expertOut,s,true);[e endEncoding];double ms=0;
    if(!q_complete(command,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}if(trial>=3)sum+=ms/5;
   }
   printf("{\"pattern\":%d,\"direct\":%s,\"pipeline\":%s,\"meanGpuMs\":%.6f,\"dispatches\":%d}\n",pattern,direct?"true":"false",g.prefill->pipelineExperts?"true":"false",sum,g.dispatches);fflush(stdout);
  }
 }
 for(auto *q:m)delete q;delete c;
}}
