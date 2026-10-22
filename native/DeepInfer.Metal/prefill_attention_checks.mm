// Independent double oracle for fused attention and the real bounded short scratch.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","128",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;g.capacity=8192;g.kvBf16=std::getenv("DEEPINFER_METAL_KV")&&strcmp(std::getenv("DEEPINFER_METAL_KV"),"bf16")==0;NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
 auto library=[c->device newLibraryWithSource:g_kv_source(&g) options:options error:&failure];
 if(!library||!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s %s\n",error,failure.localizedDescription.UTF8String);return 1;}
 if(g.prefill->maxTokens!=128||g.prefill->scores.length<NSUInteger(128)*16*2048*4){fprintf(stderr,"Undersized prefill score allocation\n");return 1;}
 g.ps[GAttentionCombine]=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_attention_combine"] error:&failure];if(!g.ps[GAttentionCombine])return 1;
 g.ps[GAttentionSplit]=[c->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"graph_attention_split"] error:&failure];if(!g.ps[GAttentionSplit])return 1;
 auto q=g_buffer(&g,256*4096),keys=g_kv_buffer(&g,8192*512),values=g_kv_buffer(&g,8192*512),raw=g_buffer(&g,256*8192),out=g_buffer(&g,256*4096);
 g.attentionParts=g_buffer(&g,NSUInteger(128)*16*8*258);
 std::vector<float> keyReference(8192*512),valueReference(8192*512);
 auto *qx=static_cast<float*>(q.contents),*kx=keyReference.data(),*vx=valueReference.data(),*rx=static_cast<float*>(raw.contents);
 uint32_t seed=0x6a51c9e7u;
 auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return float(int(seed%10001)-5000)/12500.0f;};
 for(int i=0;i<256*4096;i++)qx[i]=random();
 for(int i=0;i<8192*512;i++){
  for(auto pair:{std::make_pair(keys,kx),std::make_pair(values,vx)}){float value=random();
   if(g.kvBf16){uint32_t bits;memcpy(&bits,&value,4);uint16_t b=(bits+0x7fff+((bits>>16)&1))>>16;
    static_cast<uint16_t*>(pair.first.contents)[i]=b;bits=uint32_t(b)<<16;memcpy(&value,&bits,4);
   }else static_cast<float*>(pair.first.contents)[i]=value;pair.second[i]=value;}
 }
 for(int i=0;i<256*8192;i++)rx[i]=random()*5;
 auto longScores=g_buffer(&g,NSUInteger(128)*16*g.capacity);
 auto roundbf=[](float value){uint32_t bits;memcpy(&bits,&value,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;memcpy(&value,&bits,4);return value;};
 int checks=0;double worst=0,worstRelative=0;
 for(int base:{0,31,1022,2047,4098})for(int count:{1,3,5,17,32,65,128}) {
  for(int offset:{0,128}) {
  if(offset&&(!(base==31||base==2047||base==4098)||!(count==3||count==128)))continue;
  for(int mode:{0,1,2}) {
  bool shortLegacy=mode==1;
  if(shortLegacy&&base+offset+count>2048)continue;
  if(mode==2&&!(offset&&base>=2047))continue;
  g.position=base;auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
  if(mode==0)p_fused_attention(&g,e,q,keys,values,raw,out,count,offset);
  else {
   PShape shape{0,0,(uint32_t)count,(uint32_t)(base+offset),0,0,1e-6f,1e7f,shortLegacy?g.prefill->scoreCapacity:g.capacity};
   auto scores=shortLegacy?g.prefill->scores:longScores;
   g_bind(e,q,0,NSUInteger(offset)*4096*4);g_bind(e,keys,1);g_bind(e,scores,2);[e setBytes:&shape length:sizeof(shape) atIndex:3];
   p_run(&g,e,PScores,MTLSizeMake((base+offset+count+3)/4,16,count),128);g_barrier(e);
   g_bind(e,scores,0);g_bind(e,values,1);g_bind(e,raw,2,NSUInteger(offset)*8192*4);g_bind(e,out,3,NSUInteger(offset)*4096*4);[e setBytes:&shape length:sizeof(shape) atIndex:4];
   if(shortLegacy){p_run(&g,e,PAttention,MTLSizeMake(16,count,1),256);g_barrier(e);}
   else g_long_attention(&g,e,scores,values,raw,out,count,offset);
  }
  [e endEncoding];
  if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  auto *actual=static_cast<const float*>(out.contents)+offset*4096;double error2=0,norm2=0;
  for(int t=0;t<count;t++)for(int h=0;h<16;h++) {
   std::vector<double> scores(base+offset+t+1);double maximum=-INFINITY;
   for(int key=0;key<=base+offset+t;key++){double score=0;for(int d=0;d<256;d++)score+=(mode==0&&g.prefill->attentionBf16?double(roundbf(qx[(offset+t)*4096+h*256+d]))*roundbf(kx[key*512+(h/8)*256+d]):double(qx[(offset+t)*4096+h*256+d])*kx[key*512+(h/8)*256+d]);
    scores[key]=score/16;maximum=std::max(maximum,scores[key]);}
   double total=0;std::array<double,256> weighted{};
   if(mode==0&&g.prefill->attentionBf16){
    // Independent double online reference with BF16 Q/K/V operands and BF16
    // tile probabilities, FP32-like unrounded normalization statistics.
    int partition=count>=32?int(g.prefill->attentionPartitionTokens):1024;
    for(int start=0;start<int(scores.size());start+=partition){double pm=-INFINITY,psum=0;std::array<double,256> numerator{};
     for(int block=start;block<std::min(start+partition,int(scores.size()));block+=g.prefill->attentionKeyTokens){int end=std::min(block+int(g.prefill->attentionKeyTokens),int(scores.size()));double next=pm;for(int key=block;key<end;key++)next=std::max(next,scores[key]);double alpha=std::isfinite(pm)?std::exp(pm-next):0;
      psum*=alpha;for(auto &value:numerator)value*=alpha;
      for(int key=block;key<end;key++){double prob=std::exp(scores[key]-next);psum+=prob;
       for(int d=0;d<256;d++)numerator[d]+=roundbf(float(prob))*roundbf(vx[key*512+(h/8)*256+d]);}pm=next;
     }
     double scale=std::exp(pm-maximum);total+=scale*psum;for(int d=0;d<256;d++)weighted[d]+=scale*numerator[d];
    }
   }else for(int key=0;key<=base+offset+t;key++){double p=std::exp(scores[key]-maximum);total+=p;
    for(int d=0;d<256;d++)weighted[d]+=p*vx[key*512+(h/8)*256+d];}
   for(int d=0;d<256;d++){double expected=weighted[d]/total/(1+std::exp(-double(rx[(offset+t)*8192+h*512+256+d])));
    double difference=actual[t*4096+h*256+d]-expected;
    if(!std::isfinite(difference)||std::abs(difference)>3e-5){fprintf(stderr,"Fused random oracle mismatch base=%d count=%d t=%d h=%d d=%d expected=%.9g got=%.9g\n",base,count,t,h,d,expected,actual[t*4096+h*256+d]);return 1;}
    worst=std::max(worst,std::abs(difference));error2+=difference*difference;norm2+=expected*expected;
   }
  }
  double relative=std::sqrt(error2/std::max(norm2,1e-30));if(relative>1e-4)return 1;worstRelative=std::max(worstRelative,relative);checks++;
  }
  }
 }
 printf("PASS: %d random fused/short mixed attention checks, double oracle, query offsets and bounded 128-token scratch; worst absolute %.9g, relative L2 %.9g\n",checks,worst,worstRelative);delete c;return 0;
}}
