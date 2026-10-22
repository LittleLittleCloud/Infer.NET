// Analytic BF16 attention through the last native context address; no model.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION_IMPL","bf16-nax",1);setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);
 const char *batch=std::getenv("DEEPINFER_METAL_PREFILL_ATTENTION_BATCH");uint32_t queries=batch?std::max(128,std::atoi(batch)):128;
 char error[4096]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;g.capacity=262144;g.kvBf16=true;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 NSError *failure=nil;auto lib=[c->device newLibraryWithSource:g_kv_source(&g) options:nil error:&failure];
 g.ps[GAttentionCombine]=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"graph_attention_combine"] error:&failure];if(!g.ps[GAttentionCombine])return 1;
 auto keys=g_kv_buffer(&g,NSUInteger(262144)*512),values=g_kv_buffer(&g,NSUInteger(262144)*512),q=g_buffer(&g,queries*4096),raw=g_buffer(&g,queries*8192),out=g_buffer(&g,queries*4096);
 if(!keys||!values||!q||!raw||!out)return 1;memset(keys.contents,0,keys.length);memset(q.contents,0,q.length);memset(raw.contents,0,raw.length);
 auto *v=static_cast<uint16_t*>(values.contents);for(NSUInteger i=0;i<NSUInteger(262144)*512;i++)v[i]=(i%512)<256?0x3f80:0xc000;
 int checks=0;
 std::vector<uint32_t> counts{1,3,128};if(queries>128){counts.push_back(queries-1);counts.push_back(queries);}
 for(uint32_t end:{2048u,32768u,131072u,262144u})for(uint32_t count:counts){
  g.position=end-count;auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];p_fused_attention(&g,e,q,keys,values,raw,out,count);[e endEncoding];
  if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  auto *actual=static_cast<float*>(out.contents);for(uint32_t t=0;t<count;t++)for(uint32_t h=0;h<16;h++)for(uint32_t d=0;d<256;d++){
   float expected=h<8?.5f:-1.f,value=actual[t*4096+h*256+d];if(!std::isfinite(value)||std::abs(value-expected)>3e-5){fprintf(stderr,"Long BF16 mismatch end=%u t=%u h=%u d=%u value=%g\n",end,t,h,d,value);return 1;}
  }checks++;
 }
 printf("PASS: %d BF16 attention analytic checks through position 262143, partial query tiles and partitions\n",checks);delete c;
}}
