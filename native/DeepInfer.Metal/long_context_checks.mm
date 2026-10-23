// Independent analytic attention oracle, including long strides and causal tails.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));
 if(!c){fprintf(stderr,"%s\n",error);return 1;}
 GModel g{};g.compute=*c;NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto library=[c->device newLibraryWithSource:graph_source options:options error:&failure];
 if(!library||!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s %s\n",error,failure.localizedDescription.UTF8String);return 1;}
 for(auto entry:{std::pair<int,NSString*>{GScores,@"graph_attention_scores"},{GAttentionSplit,@"graph_attention_split"},{GAttentionCombine,@"graph_attention_combine"}}){
  g.ps[entry.first]=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:entry.second] error:&failure];
  if(!g.ps[entry.first])return 1;
 }
 int checks=0;
 for(int base:{0,1022,1024,2047,2048,4098,262141})for(int tokens:{1,3}){
  g.position=base;g.capacity=base>=4098?262144:8192;
  int length=base+tokens;auto buffer=[&](NSUInteger n){return g_buffer(&g,n);};
  auto q=buffer(tokens*4096),keys=buffer(length*512),values=buffer(length*512),scores=buffer(NSUInteger(tokens)*16*g.capacity),raw=buffer(tokens*8192),out=buffer(tokens*4096);
  g.attentionParts=buffer(NSUInteger(tokens)*16*((length+1023)/1024)*258);
  auto *qx=static_cast<float*>(q.contents),*kx=static_cast<float*>(keys.contents),*vx=static_cast<float*>(values.contents),*rx=static_cast<float*>(raw.contents);
  for(int t=0;t<tokens;t++)for(int h=0;h<16;h++)for(int d=0;d<256;d++){
   qx[t*4096+h*256+d]=.1f+h*.01f+t*.03f;
   rx[t*8192+h*512+256+d]=(h-8)*.2f+t*.03f;
  }
  for(int t=0;t<length;t++)for(int h=0;h<2;h++)for(int d=0;d<256;d++){
   kx[t*512+h*256+d]=(t%37-18)*.03f+h*.01f;
   vx[t*512+h*256+d]=(t%41-20)*.02f+(d%17)*.003f+h*.1f;
  }
  auto *sx=static_cast<float*>(scores.contents);for(NSUInteger i=0;i<scores.length/4;i++)sx[i]=NAN;
  for(bool fused:{false,true}) {
  auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
  if(!fused){
  g_bind(e,q,0);g_bind(e,keys,1);g_bind(e,scores,2);
  if(tokens==1){GStep shape{(uint32_t)base,0,0,1e-6f,1e7f,g.capacity};[e setBytes:&shape length:sizeof(shape) atIndex:3];
   [e setComputePipelineState:g.ps[GScores]];[e dispatchThreadgroups:MTLSizeMake((length+3)/4,16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];}
  else {PShape shape{0,0,(uint32_t)tokens,(uint32_t)base,0,0,1e-6f,1e7f,g.capacity};[e setBytes:&shape length:sizeof(shape) atIndex:3];
   p_run(&g,e,PScores,MTLSizeMake((length+3)/4,16,tokens),128);}
  g_barrier(e);g_long_attention(&g,e,scores,values,raw,out,tokens);
  } else p_fused_attention(&g,e,q,keys,values,raw,out,tokens);
  [e endEncoding];
  if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  auto *actual=static_cast<const float*>(out.contents);
  for(int t=0;t<tokens;t++)for(int h=0;h<16;h++){
   double maximum=-INFINITY;std::vector<double> expected(base+t+1);
   for(int k=0;k<=base+t;k++){
    double score=double(qx[t*4096+h*256])*kx[k*512+(h/8)*256]*16;
    if(std::abs(sx[(NSUInteger(t)*16+h)*g.capacity+k]-score)>1e-5){fprintf(stderr,"Score oracle failed base=%d token=%d head=%d\n",base,t,h);return 1;}
    expected[k]=score;maximum=std::max(maximum,score);
   }
   double total=0,weighted=0;for(int k=0;k<=base+t;k++){double p=std::exp(expected[k]-maximum);total+=p;weighted+=p*vx[k*512+(h/8)*256];}
   double gate=1/(1+std::exp(-double(rx[t*8192+h*512+256])));
   for(int d=0;d<256;d++){
    double answer=(weighted/total+(d%17)*.003)*gate;
    if(!std::isfinite(actual[t*4096+h*256+d])||std::abs(actual[t*4096+h*256+d]-answer)>2e-5){
     fprintf(stderr,"Attention oracle failed base=%d tokens=%d head=%d d=%d expected=%.9g actual=%.9g\n",base,tokens,h,d,answer,actual[t*4096+h*256+d]);return 1;}
   }
  }
  checks++;
  }
 }
 {
  GModel reserve{};reserve.compute=*c;
  for(int bad:{0,-1,262145}){if(di_graph_reserve(&reserve,bad,error,sizeof(error))==0)return 1;checks++;}
  reserve.position=1;if(di_graph_reserve(&reserve,4096,error,sizeof(error))==0)return 1;checks++;reserve.position=0;
  // Real KV allocation for one attention layer; stats describe the full fixed geometry.
  reserve.layers[3].loaded=true;
  if(di_graph_reserve(&reserve,4097,error,sizeof(error))!=0||reserve.capacity!=6144||reserve.layers[3].keys.length!=NSUInteger(6144)*512*4)return 1;
  int capacity=0;uint64_t kv=0,allocated=0;
  if(di_graph_stats(&reserve,&capacity,&kv,&allocated)!=0||capacity!=6144||kv!=uint64_t(6144)*40960||!allocated)return 1;
  checks++;
  // Grow during decode and preserve already-written KV. Linear state is not reallocated.
  reserve.position=7;auto oldKeys=reserve.layers[3].keys,oldValues=reserve.layers[3].values;
  for(int i=0;i<7*512;i++){((float*)oldKeys.contents)[i]=float(i%17)/17;((float*)oldValues.contents)[i]=float(i%23)/23;}
  if(di_graph_grow(&reserve,8192,error,sizeof(error))||reserve.capacity!=8192||reserve.position!=7)return 1;
  if(memcmp(oldKeys.contents,reserve.layers[3].keys.contents,7*512*4)||memcmp(oldValues.contents,reserve.layers[3].values.contents,7*512*4))return 1;
  reserve.position=0;checks++;
  for(int requested:{131072,131073,135168,262144}){
   uint32_t expected=(uint32_t(requested)+2047)/2048*2048;
   if(di_graph_reserve(&reserve,requested,error,sizeof(error))||reserve.capacity!=expected||reserve.layers[3].keys.length!=NSUInteger(expected)*512*4)return 1;
   if(di_graph_reserve(&reserve,requested-1,error,sizeof(error))||reserve.capacity!=expected)return 1;
   checks++;
  }
 }
 delete c;printf("PASS: %d long-context hardware checks (28 analytic legacy/fused attention cases, including final context position 262143; 10 reservation/statistics/grow guards)\n",checks);return 0;
}}
