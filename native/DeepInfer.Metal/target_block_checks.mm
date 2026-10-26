// Independent FP32 recurrence and history oracles for the target block path.
#include "bridge.mm"
#include <vector>
#include <fstream>
static bool close_values(const char *name,const float *actual,const std::vector<float> &expected){
 double error=0,norm=0;float worst=0;for(size_t i=0;i<expected.size();i++){
  if(!std::isfinite(actual[i]))return false;double d=double(actual[i])-expected[i];error+=d*d;norm+=double(expected[i])*expected[i];worst=std::max(worst,float(std::abs(d)));}
 double rel=std::sqrt(error/std::max(norm,1e-30));printf("%s relativeL2=%.9g maxAbs=%.9g\n",name,rel,worst);return rel<1e-4&&worst<0.01f;
}
int main(int argc,char **argv){@autoreleasepool{
 auto derived=block_graph_source();if(!derived){fprintf(stderr,"Source derivation failed\n");return 1;}
 if(argc==3&&strcmp(argv[1],"--dump")==0){NSString *source=[[graph_source stringByAppendingString:derived] stringByAppendingString:block_state_source];
  NSError *failure=nil;if(![source writeToFile:[NSString stringWithUTF8String:argv[2]] atomically:YES encoding:NSUTF8StringEncoding error:&failure])return 1;return 0;}
 auto device=MTLCreateSystemDefaultDevice();if(!device)return 1;auto queue=[device newCommandQueue];
 auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;NSError *failure=nil;
 auto lib=[device newLibraryWithSource:[[graph_source stringByAppendingString:derived] stringByAppendingString:block_state_source] options:options error:&failure];
 if(!lib){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);return 1;}
 auto delta=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"block_delta"] error:&failure];
 auto commit=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"block_commit_history"] error:&failure];
 auto conv=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"block_graph_linear_prepare"] error:&failure];
 if(!delta||!commit||!conv)return 1;
 auto buffer=[&](const std::vector<float> &v){return [device newBufferWithBytes:v.data() length:v.size()*4 options:MTLResourceStorageModeShared];};
 for(int seed:{17,71})for(uint32_t tokens:{1u,4u,8u}){
  std::vector<float> state(blockStateElements),raw(tokens*8192),decay(tokens*32),beta(tokens*32),history(3*8192),weights(4*8192),alog(32,-2),dt(32,0.1f);
  for(size_t i=0;i<state.size();i++)state[i]=std::sin(float(i+seed)*0.013f)*0.03f;
  for(size_t i=0;i<raw.size();i++)raw[i]=std::sin(float(i+seed)*0.021f)*0.1f;
  for(size_t i=0;i<history.size();i++)history[i]=std::cos(float(i+seed)*0.017f)*0.2f;
  for(size_t i=0;i<weights.size();i++)weights[i]=std::sin(float(i+seed)*0.031f)*0.15f;
  for(size_t i=0;i<decay.size();i++){decay[i]=0.8f+0.1f*std::sin(float(i+seed));beta[i]=0.4f+0.1f*std::cos(float(i+seed));}
  auto st=buffer(state),rw=buffer(raw),de=buffer(decay),be=buffer(beta),hi=buffer(history),we=buffer(weights),al=buffer(alog),db=buffer(dt);
  auto prefix=[device newBufferWithLength:size_t(tokens)*blockStateElements*4+64 options:MTLResourceStorageModeShared];
  auto out=[device newBufferWithLength:tokens*4096*4+64 options:MTLResourceStorageModeShared];
  memset(prefix.contents,0xa5,prefix.length);memset(out.contents,0xa5,out.length);
  std::vector<float> expect(size_t(tokens)*blockStateElements),expectOut(tokens*4096);
  // CPU independently implements the recurrence in key-dimension order.
  for(uint32_t t=0;t<tokens;t++)for(uint32_t r=0;r<4096;r++){
   uint32_t head=r/128,kh=head/2;float prediction=0;
   for(uint32_t d=0;d<128;d++){state[r*128+d]*=decay[t*32+head];prediction+=state[r*128+d]*raw[t*8192+2048+kh*128+d];}
   float correction=(raw[t*8192+4096+r]-prediction)*beta[t*32+head],value=0;
   for(uint32_t d=0;d<128;d++){state[r*128+d]+=raw[t*8192+2048+kh*128+d]*correction;expect[size_t(t)*blockStateElements+r*128+d]=state[r*128+d];value+=state[r*128+d]*raw[t*8192+kh*128+d];}
   expectOut[t*4096+r]=value;
  }
  auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:delta];
  id<MTLBuffer> binds[]={st,rw,de,be,out,prefix};for(int i=0;i<6;i++)g_bind(e,binds[i],i);[e setBytes:&tokens length:4 atIndex:6];
  [e dispatchThreadgroups:MTLSizeMake(2048,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[e endEncoding];char error[2048]{};
  if(!q_complete(command,nullptr,error,sizeof(error))||!close_values("prefix",(const float*)prefix.contents,expect)||!close_values("delta-output",(const float*)out.contents,expectOut))return 1;
  for(size_t i=size_t(tokens)*blockStateElements*4;i<prefix.length;i++)if(((const unsigned char*)prefix.contents)[i]!=0xa5)return 1;
  for(uint32_t accepted=0;accepted<=tokens;accepted++){
   memcpy(hi.contents,history.data(),history.size()*4);std::vector<float> expectedHistory(history.size());
   for(uint32_t j=0;j<3;j++)for(uint32_t d=0;d<8192;d++){uint32_t at=j+accepted;expectedHistory[j*8192+d]=at<3?history[at*8192+d]:raw[(at-3)*8192+d];}
   command=[queue commandBuffer];e=[command computeCommandEncoder];[e setComputePipelineState:commit];g_bind(e,hi,0);g_bind(e,rw,1);[e setBytes:&accepted length:4 atIndex:2];
   [e dispatchThreadgroups:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];
   if(!q_complete(command,nullptr,error,sizeof(error))||memcmp(hi.contents,expectedHistory.data(),history.size()*4))return 1;
  }
  memcpy(hi.contents,history.data(),history.size()*4);
  // Independent causal convolution, per-head normalization, decay and beta.
  std::vector<float> expectedConv(tokens*8192),expectedA(tokens*32),expectedB(tokens*32),initialA(tokens*32,0.2f),initialB(tokens*32,-0.3f);
  for(uint32_t t=0;t<tokens;t++)for(uint32_t h=0;h<64;h++){
   float squared=0;
   for(uint32_t d=0;d<128;d++){uint32_t at=h*128+d;float value=0;
    for(uint32_t j=0;j<4;j++){int p=int(t)+int(j)-3;float x=p<0?history[(p+3)*8192+at]:raw[p*8192+at];value+=x*weights[at*4+j];}
    value=value/(1+std::exp(-value));expectedConv[t*8192+at]=value;squared+=value*value;}
   if(h<32){float scale=1/std::sqrt(squared+1e-6f)*(h<16?1/std::sqrt(128.0f):1);
    for(uint32_t d=0;d<128;d++)expectedConv[t*8192+h*128+d]*=scale;}
  }
  for(size_t i=0;i<expectedA.size();i++){float z=initialA[i]+dt[i%32];expectedA[i]=std::exp(-std::exp(alog[i%32])*std::log(1+std::exp(z)));expectedB[i]=1/(1+std::exp(-initialB[i]));}
  auto a=buffer(initialA),b=buffer(initialB);auto prepared=[device newBufferWithLength:raw.size()*4 options:MTLResourceStorageModeShared];
  command=[queue commandBuffer];e=[command computeCommandEncoder];[e setComputePipelineState:conv];id<MTLBuffer> cs[]={rw,hi,we,prepared,a,b,al,db};for(int i=0;i<8;i++)g_bind(e,cs[i],i);
  [e dispatchThreadgroups:MTLSizeMake(64,tokens,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
  if(!q_complete(command,nullptr,error,sizeof(error))||!close_values("convolution",(const float*)prepared.contents,expectedConv)||!close_values("decay",(const float*)a.contents,expectedA)||!close_values("beta",(const float*)b.contents,expectedB)||memcmp(hi.contents,history.data(),history.size()*4))return 1;
  printf("PASS seed=%d rows=%u history-prefixes=%u\n",seed,tokens,tokens+1);
 }
 // Verify absolute-position KV writes across a 2K boundary. Distinct row values
 // and poisoned neighboring rows catch all rows incorrectly sharing one slot.
 {
  constexpr uint32_t rows=8,start=2045,capacity=4096;
  std::vector<float> raw(rows*8192),k(rows*512),v(rows*512),norm(256,1),empty(capacity*512,std::numeric_limits<float>::quiet_NaN());
  for(size_t i=0;i<raw.size();i++)raw[i]=std::sin(float(i+23)*0.019f);
  for(size_t i=0;i<k.size();i++){k[i]=std::cos(float(i+19)*0.031f);v[i]=float(i+1)*0.001f;}
  auto rw=buffer(raw),ky=buffer(k),vl=buffer(v),n=buffer(norm),keys=buffer(empty),values=buffer(empty);
  auto q=[device newBufferWithLength:rows*4096*4 options:MTLResourceStorageModeShared];
  auto head=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"block_graph_head_prepare"] error:&failure];
  GStep shape{start,3,0,1e-6f,10000000.0f,capacity};auto command=[queue commandBuffer];auto e=[command computeCommandEncoder];[e setComputePipelineState:head];
  id<MTLBuffer> bs[]={rw,ky,vl,n,n,q,keys,values};for(int i=0;i<8;i++)g_bind(e,bs[i],i);[e setBytes:&shape length:sizeof(shape) atIndex:8];
  [e dispatchThreadgroups:MTLSizeMake(18,rows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[e endEncoding];char error[2048]{};
  if(!q_complete(command,nullptr,error,sizeof(error)))return 1;
  const auto *actual=(const float*)values.contents;const auto *actualK=(const float*)keys.contents;
  for(uint32_t row=0;row<capacity;row++)for(uint32_t d=0;d<512;d++){
   if(row>=start&&row<start+rows){if(actual[row*512+d]!=v[(row-start)*512+d]||!std::isfinite(actualK[row*512+d]))return 1;}
   else if(!std::isnan(actual[row*512+d])||!std::isnan(actualK[row*512+d]))return 1;
  }
  printf("PASS absolute-position KV rows and poisoned neighbors\n");
 }
 printf("PASS target-block independent state oracles\n");return 0;
}}
