// Compare existing FP32 decode attention paths with a double oracle and GPU timings.
// No model weights required. Timings are hot-cache operator diagnostics, not inference.
#include "bridge.mm"
#include <vector>
#include <algorithm>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","32",1);
 setenv("DEEPINFER_METAL_PREFILL_GEMM","fp32",1);
 setenv("DEEPINFER_METAL_PREFILL_ATTENTION_IMPL","fp32",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c){fprintf(stderr,"%s\n",error);return 1;}
 GModel g{};g.compute=*c;g.capacity=32768;g.kvBf16=false;
 if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto lib=[c->device newLibraryWithSource:g_kv_source(&g) options:options error:&failure];if(!lib)return 1;
 for(auto pair:{std::make_pair(GScores,@"graph_attention_scores"),std::make_pair(GAttention,@"graph_attention_output"),std::make_pair(GAttentionSplit,@"graph_attention_split"),std::make_pair(GAttentionCombine,@"graph_attention_combine")}){
  g.ps[pair.first]=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:pair.second] error:&failure];if(!g.ps[pair.first])return 1;
 }
 auto q=g_buffer(&g,4096),keys=g_kv_buffer(&g,size_t(g.capacity)*512),values=g_kv_buffer(&g,size_t(g.capacity)*512),raw=g_buffer(&g,8192),out=g_buffer(&g,4096),scores=g_buffer(&g,16*g.capacity);
 auto *qx=static_cast<float*>(q.contents),*kx=static_cast<float*>(keys.contents),*vx=static_cast<float*>(values.contents),*rx=static_cast<float*>(raw.contents);
 uint32_t seed=0x71f23c5;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%10001)-5000)/12500.0f;};
 for(auto b:{q,keys,values,raw})for(NSUInteger i=0;i<b.length/4;i++)static_cast<float*>(b.contents)[i]=random();
 auto encode=[&](id<MTLComputeCommandEncoder> e,int mode){
  if(mode==2){p_decode_attention(&g,e,q,keys,values,raw,out);return;}
  if(mode==1){p_fused_attention(&g,e,q,keys,values,raw,out,1);return;}
  GStep shape{g.position,0,0,1e-6f,1e7f,g.capacity};
  g_bind(e,q,0);g_bind(e,keys,1);g_bind(e,scores,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];
  [e setComputePipelineState:g.ps[GScores]];[e dispatchThreadgroups:MTLSizeMake((g.position+4)/4,16,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];g_barrier(e);
  if(g.position>=2048)g_long_attention(&g,e,scores,values,raw,out,1);
  else {
   g_bind(e,scores,0);g_bind(e,values,1);g_bind(e,raw,2);g_bind(e,out,3);[e setBytes:&shape length:sizeof(shape) atIndex:4];
   [e setComputePipelineState:g.ps[GAttention]];[e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g_barrier(e);
  }
 };
 for(int length:{17,2048,8192,32768}){
  g.position=length-1;std::vector<double> reference(4096),logits(length);
  for(int h=0;h<16;h++){
   double maximum=-INFINITY;
   for(int t=0;t<length;t++){double dot=0;for(int d=0;d<256;d++)dot+=double(qx[h*256+d])*kx[t*512+(h/8)*256+d];logits[t]=dot/16;maximum=std::max(maximum,logits[t]);}
   double total=0;for(auto &x:logits){x=std::exp(x-maximum);total+=x;}
   for(int d=0;d<256;d++){double sum=0;for(int t=0;t<length;t++)sum+=logits[t]*vx[t*512+(h/8)*256+d];reference[h*256+d]=sum/total/(1+std::exp(-double(rx[h*512+256+d])));}
  }
  for(int mode:{0,1,2}){
   std::vector<double> timings;double worst=0,relative=0;
   for(int trial=0;trial<11;trial++){
    auto cb=[c->queue commandBuffer];auto e=[cb computeCommandEncoder];
    for(int repeat=0;repeat<8;repeat++)encode(e,mode);[e endEncoding];double ms=0;
    if(!q_complete(cb,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
    if(trial)timings.push_back(ms/8);
    auto *actual=static_cast<float*>(out.contents);double squared=0,norm=0;
    for(int i=0;i<4096;i++){double diff=actual[i]-reference[i];if(!std::isfinite(diff))return 1;worst=std::max(worst,std::abs(diff));squared+=diff*diff;norm+=reference[i]*reference[i];}
    relative=std::max(relative,std::sqrt(squared/std::max(norm,1e-30)));
   }
   std::sort(timings.begin(),timings.end());double median=(timings[4]+timings[5])/2;
   printf("{\"length\":%d,\"mode\":%d,\"medianGpuMs\":%.9g,\"maxAbs\":%.9g,\"relativeL2\":%.9g,\"allFinite\":true}\n",length,mode,median,worst,relative);fflush(stdout);
   if(worst>=1e-5||relative>=1e-4){fprintf(stderr,"Decode attention oracle failed\n");return 1;}
  }
 }
 delete c;return 0;
}}
