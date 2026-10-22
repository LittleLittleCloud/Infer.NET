// Independent double recurrence oracle; tests the production chunkwise kernels.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);setenv("DEEPINFER_METAL_PREFILL_DELTA","chunked",1);
 char error[65536]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 auto *p=g.prefill;auto state=g_buffer(&g,32*128*128),history=g_buffer(&g,3*8192);
 auto *conv=static_cast<float*>(p->conv.contents),*a=static_cast<float*>(p->a.contents),*b=static_cast<float*>(p->b.contents),*raw=static_cast<float*>(p->qkv.contents);
 for(int t=0;t<2048;t++)for(int h=0;h<16;h++){
  double norm=0;for(int d=0;d<128;d++){float v=std::sin((t*19+h*7+d)*.071f);conv[t*8192+2048+h*128+d]=v;norm+=double(v)*v;}
  for(int d=0;d<128;d++){conv[t*8192+2048+h*128+d]/=std::sqrt(norm);conv[t*8192+h*128+d]=std::cos((t*7+h*13+d)*.031f)*.02f;}
 }
 for(int t=0;t<2048;t++)for(int h=0;h<32;h++)for(int d=0;d<128;d++)conv[t*8192+4096+h*128+d]=std::sin((t*11+h*17+d)*.013f)*.7f;
 for(int i=0;i<2048*8192;i++)raw[i]=std::sin(i*.007f);
 int checks=0;double worstOut=0,worstState=0,worstRelative=0;double times[2]{};
 for(int mode=0;mode<4;mode++)for(int tokens:{1,3,31,32,63,64,65,127,128,129,256,512,513,1024,2048}) {
  if(mode==3)for(int t=0;t<tokens;t++)for(int h=0;h<16;h++){
   double n=0;for(int d=0;d<128;d++){float v=std::sin((h*7+d)*.071f);conv[t*8192+2048+h*128+d]=v;n+=double(v)*v;}
   for(int d=0;d<128;d++)conv[t*8192+2048+h*128+d]/=std::sqrt(n);
  }
  for(int t=0;t<tokens;t++)for(int h=0;h<32;h++){
   a[t*32+h]=mode==3?.9999f:mode==0?.9f+((t+h)%10)*.009f:mode==1?((t+h)%17==0?0.0f:(t+h)%13==0?1.0f:.7f):((t+h)%7==0?1e-15f:.999f);
   b[t*32+h]=mode==3?.9999f:mode==1?((t+h)%11==0?0.0f:1.0f):.3f+((t+h)%7)*.09f;
  }
  std::vector<double> oracle(32*128*128);auto *stored=static_cast<float*>(state.contents),*hist=static_cast<float*>(history.contents);
  for(int i=0;i<32*128*128;i++)oracle[i]=stored[i]=std::sin(i*.003f)*.01f;
  for(int i=0;i<3*8192;i++)hist[i]=float(i)*.0001f;
  PShape shape{0,0,(uint32_t)tokens,0,0,0,1e-6f};auto command=[c->queue commandBuffer];auto encoder=[command computeCommandEncoder];
  p_delta(&g,encoder,state,history,shape);[encoder endEncoding];double ms=0;
  if(!q_complete(command,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  auto *actual=static_cast<float*>(p->attention.contents);double out2=0,norm2=0;
  for(int t=0;t<tokens;t++)for(int row=0;row<4096;row++){
   int head=row/128,kh=head/2;double prediction=0,result=0;
   for(int d=0;d<128;d++){oracle[row*128+d]*=a[t*32+head];prediction+=oracle[row*128+d]*conv[t*8192+2048+kh*128+d];}
   double correction=(conv[t*8192+4096+row]-prediction)*b[t*32+head];
   for(int d=0;d<128;d++){oracle[row*128+d]+=conv[t*8192+2048+kh*128+d]*correction;result+=oracle[row*128+d]*conv[t*8192+kh*128+d];}
   double diff=actual[t*4096+row]-result;worstOut=std::max(worstOut,std::abs(diff));out2+=diff*diff;norm2+=result*result;
   if(!std::isfinite(diff)||std::abs(diff)>1e-5){fprintf(stderr,"Delta output mismatch mode=%d T=%d t=%d row=%d expected=%.9g actual=%.9g\n",mode,tokens,t,row,result,actual[t*4096+row]);return 1;}
  }
  double rel=std::sqrt(out2/std::max(norm2,1e-30));worstRelative=std::max(worstRelative,rel);if(rel>3e-5)return 1;
  for(int i=0;i<32*128*128;i++){double diff=std::abs(oracle[i]-stored[i]);worstState=std::max(worstState,diff);
   if(!std::isfinite(diff)||diff>2e-5){fprintf(stderr,"Delta state mismatch mode=%d T=%d i=%d expected=%.9g actual=%.9g\n",mode,tokens,i,oracle[i],stored[i]);return 1;}}
  for(int j=0;j<3;j++)for(int d=0;d<8192;d++){int at=j+tokens;float expected=at<3?float(at*8192+d)*.0001f:raw[(at-3)*8192+d];if(hist[j*8192+d]!=expected)return 1;}
  checks++;printf("PASS delta mode=%d tokens=%d ms=%.4f\n",mode,tokens,ms);fflush(stdout);
 }
 printf("PASS: %d chunkwise delta checks; max output %.9g, max state %.9g, relative L2 %.9g\n",checks,worstOut,worstState,worstRelative);delete c;return 0;
}}
