// Isolated full-attention timing; not an end-to-end model throughput claim.
#include "bridge.mm"
int main(){@autoreleasepool {
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c){fprintf(stderr,"%s\n",error);return 1;}
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION","legacy",1);
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","32",1);
 GModel g{};g.compute=*c;g.capacity=262144;NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto library=[c->device newLibraryWithSource:graph_source options:options error:&failure];
 if(!library||!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s %s\n",error,failure.localizedDescription.UTF8String);return 1;}
 for(auto entry:{std::pair<int,NSString*>{GAttentionSplit,@"graph_attention_split"},{GAttentionCombine,@"graph_attention_combine"}}){
  g.ps[entry.first]=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:entry.second] error:&failure];if(!g.ps[entry.first])return 1;
 }
 auto q=g_buffer(&g,32*4096),keys=g_buffer(&g,NSUInteger(262144)*512),values=g_buffer(&g,NSUInteger(262144)*512),raw=g_buffer(&g,32*8192),out=g_buffer(&g,32*4096);
 g.attentionParts=g_buffer(&g,NSUInteger(32)*16*256*258);
 for(auto buffer:{q,keys,values,raw}){auto *x=static_cast<float*>(buffer.contents);for(NSUInteger i=0;i<buffer.length/4;i++)x[i]=float(int(i%97)-48)*.002f;}
 printf("{\"purpose\":\"isolated FP32 attention, 32 queries, not whole-model throughput\",\"warmupExcluded\":1,\"trials\":3,\"rows\":[\n");
 int n=0;
 for(int length:{2048,8192,32768,131072,262144}){
  g.position=length-32;double mean[2]{};
  for(int mode=0;mode<2;mode++)for(int trial=0;trial<4;trial++){
   auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
   if(mode){p_fused_attention(&g,e,q,keys,values,raw,out,32);}
   else {PShape shape{0,0,32,g.position,0,0,1e-6f,1e7f,g.capacity};
    g_bind(e,q,0);g_bind(e,keys,1);g_bind(e,g.prefill->scores,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];
    p_run(&g,e,PScores,MTLSizeMake((length+3)/4,16,32),128);g_barrier(e);
    g_long_attention(&g,e,g.prefill->scores,values,raw,out,32);
   }
   [e endEncoding];double ms=0;if(!q_complete(command,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
   if(trial)mean[mode]+=ms/3;
  }
  printf("%s{\"contextTokens\":%d,\"legacyMs\":%.6f,\"fusedMs\":%.6f,\"speedup\":%.4f}",n++?",\n":"",length,mean[0],mean[1],mean[0]/mean[1]);fflush(stdout);
 }
 printf("\n]}\n");delete c;return 0;
}}
