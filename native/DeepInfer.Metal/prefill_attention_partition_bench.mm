// Geometry sweep of bounded BF16 attention; not whole-model throughput.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION_IMPL","bf16-nax",1);setenv("DEEPINFER_METAL_KV","bf16",1);
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","128",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;g.capacity=32768;g.kvBf16=true;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto lib=[c->device newLibraryWithSource:g_kv_source(&g) options:options error:&failure];
 g.ps[GAttentionCombine]=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"graph_attention_combine"] error:&failure];if(!g.ps[GAttentionCombine])return 1;
 auto q=g_buffer(&g,128*4096),keys=g_kv_buffer(&g,32768*512),values=g_kv_buffer(&g,32768*512),raw=g_buffer(&g,128*8192),out=g_buffer(&g,128*4096);
 uint32_t seed=0x853ce1;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%10001)-5000)/12500.0f;};
 for(auto b:{q,raw})for(NSUInteger i=0;i<b.length/4;i++)static_cast<float*>(b.contents)[i]=random();
 for(auto b:{keys,values})for(NSUInteger i=0;i<b.length/2;i++){float v=random();uint32_t bits;memcpy(&bits,&v,4);static_cast<uint16_t*>(b.contents)[i]=(bits+0x7fff+((bits>>16)&1))>>16;}
 printf("{\"purpose\":\"isolated random BF16 attention, 128 queries, not whole-model throughput\",\"warmupExcluded\":4,\"samples\":5,\"rows\":[\n");int index=0;
 for(int partition:{1024,2048,4096,8192,16384,32768})for(int math:{0})for(int groups:{16})for(int queries:{4})for(int keyTile:{128}){
  if(queries*8*keyTile*6+queries*8*12>32768)continue;
  options=[MTLCompileOptions new];options.mathMode=math?MTLMathModeRelaxed:MTLMathModeSafe;
  if(@available(macOS 26.0,*)){options.languageVersion=MTLLanguageVersion4_0;}else return 1;
  auto source=[[NSString stringWithFormat:@"#define BF16_ATTENTION_PARTITION %d\n#define BF16_QUERY_TOKENS %d\n#define BF16_KEY_TOKENS %d\n#define BF16_ATTENTION_SG %d\n",partition,queries,keyTile,groups] stringByAppendingString:[g_kv_source(&g) stringByAppendingString:prefill_attention_bf16_source]];
  lib=[c->device newLibraryWithSource:source options:options error:&failure];if(!lib){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);return 1;}
  auto ps=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"prefill_attention_bf16"] error:&failure];if(!ps)return 1;
  if(ps.maxTotalThreadsPerThreadgroup<NSUInteger(groups*32)){fprintf(stderr,"Unsupported %d threads\n",groups*32);continue;}
  g.prefill->attentionPartitionTokens=partition;g.prefill->attentionWidePs=nil;g.prefill->attentionThreads=groups*32;g.prefill->attentionBf16Ps=ps;g.prefill->attentionQueryTokens=queries;g.prefill->attentionKeyTokens=keyTile;
  for(int length:{2048,8192,32768}){
   g.position=length-128;double total=0;
   for(int trial=0;trial<9;trial++){
    auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];p_fused_attention(&g,e,q,keys,values,raw,out,128);[e endEncoding];
    double ms=0;if(!q_complete(command,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}if(trial>=4)total+=ms/5;
   }
   printf("%s{\"partitionTokens\":%d,\"relaxedMath\":%d,\"simdgroups\":%d,\"queryTile\":%d,\"keyTile\":%d,\"contextTokens\":%d,\"gpuMs\":%.6f}",index++?",\n":"",partition,math,groups,queries,keyTile,length,total);fflush(stdout);
  }
 }
 printf("\n]}\n");delete c;return 0;
}}
