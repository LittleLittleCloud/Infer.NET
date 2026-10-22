// Metamorphic consistency against the double-oracle-tested 128-query path.
// Exercises real packing, larger partial strides, offsets, tails and reserve.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION_IMPL","bf16-nax",1);
 setenv("DEEPINFER_METAL_PREFILL_KEY_TILE","128",1);
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION_BATCH","2048",1);
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);
 char error[4096]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;g.kvBf16=true;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 if(g.prefill->packedQueries.length!=NSUInteger(2048)*4096*2)return 1;
 if(di_graph_reserve(&g,8192,error,sizeof(error))||g.capacity!=8192||g.attentionParts.length!=NSUInteger(2048)*16*8*258*4)return 1;
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto lib=[c->device newLibraryWithSource:g_kv_source(&g) options:options error:&failure];
 g.ps[GAttentionCombine]=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"graph_attention_combine"] error:&failure];if(!g.ps[GAttentionCombine])return 1;
 constexpr uint storage=2064,offset=7;
 auto q=g_buffer(&g,storage*4096),keys=g_kv_buffer(&g,8192*512),values=g_kv_buffer(&g,8192*512),raw=g_buffer(&g,storage*8192),out=g_buffer(&g,storage*4096);
 uint32_t seed=0x71f23c5;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%10001)-5000)/12500.0f;};
 for(auto b:{q,raw})for(NSUInteger i=0;i<b.length/4;i++)static_cast<float*>(b.contents)[i]=random();
 for(auto b:{keys,values})for(NSUInteger i=0;i<b.length/2;i++){float v=random();uint32_t bits;memcpy(&bits,&v,4);static_cast<uint16_t*>(b.contents)[i]=(bits+0x7fff+((bits>>16)&1))>>16;}
 auto *actual=static_cast<float*>(out.contents);int checks=1;double worst=0,worstRelative=0;
 for(uint base:{0u,1022u,4097u})for(uint count:{3u,129u,257u,513u,1024u,2048u}){
  g.position=base;std::vector<float> reference(count*4096);
  for(bool large:{false,true}){
   std::fill(actual,actual+storage*4096,NAN);
   auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
   for(uint start=0;start<count;start+=large?count:128)p_fused_attention(&g,e,q,keys,values,raw,out,std::min(large?count:128u,count-start),offset+start);
   [e endEncoding];if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
   for(uint i=0;i<offset*4096;i++)if(!std::isnan(actual[i]))return 1;
   for(uint i=(offset+count)*4096;i<storage*4096;i++)if(!std::isnan(actual[i]))return 1;
   if(!large){memcpy(reference.data(),actual+offset*4096,count*4096*4);continue;}
   double error2=0,norm2=0;
   for(uint i=0;i<count*4096;i++){double diff=actual[offset*4096+i]-reference[i];
    if(!std::isfinite(diff)||std::abs(diff)>3e-5){fprintf(stderr,"Attention batch mismatch base=%u count=%u i=%u reference=%g actual=%g\n",base,count,i,reference[i],actual[offset*4096+i]);return 1;}
    worst=std::max(worst,std::abs(diff));error2+=diff*diff;norm2+=double(reference[i])*reference[i];
   }
   double relative=std::sqrt(error2/std::max(norm2,1e-30));if(relative>1e-4)return 1;worstRelative=std::max(worstRelative,relative);
  }checks++;
 }
 for(const char *bad:{"0","1023","4096","invalid"}){
  setenv("DEEPINFER_METAL_PREFILL_ATTENTION_BATCH",bad,1);GModel invalid{};invalid.compute=*c;invalid.kvBf16=true;
  if(p_create(&invalid,error,sizeof(error)))return 1;checks++;
 }
 printf("PASS: %d attention batch checks; 128-query composition vs 3..2048 queries, random causal inputs, offsets, output guards, expanded reserve and invalid options; max absolute %.9g, relative L2 %.9g\n",checks,worst,worstRelative);
 delete c;return 0;
}}
