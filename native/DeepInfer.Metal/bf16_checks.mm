// Independent CPU BF16 RNE + double products; exercises real low-precision MPP.
#include "bridge.mm"
#include <vector>
static float bf_round(float value){uint32_t b;memcpy(&b,&value,4);b=(b+0x7fff+((b>>16)&1))&0xffff0000;memcpy(&value,&b,4);return value;}
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_GEMM","bf16-nax",1);setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);
 char error[4096]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;g.kvBf16=true;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 int checks=0;double worstRelative=0;
 constexpr int cols=256;
 auto x=g_buffer(&g,2048*cols),y=g_buffer(&g,2048*129);auto *input=static_cast<float*>(x.contents);
 for(int rows:{32,129})for(int bits:{5,8}){
  auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,1,rows,cols,bits,64,error,sizeof(error)));if(!m)return 1;
  auto *w=static_cast<uint32_t*>(m->packed.contents);memset(w,0,m->packed.length);
  auto *sc=static_cast<uint16_t*>(m->scales.contents),*bi=static_cast<uint16_t*>(m->biases.contents);
  std::vector<float> reference(rows*cols);
  for(int r=0;r<rows;r++){
   for(int group=0;group<4;group++){float scale=bf_round(.00017f*(1+(r+group)%13)),bias=bf_round(-.0271f*(1+(r+group)%3));uint32_t b;memcpy(&b,&scale,4);sc[r*4+group]=b>>16;memcpy(&b,&bias,4);bi[r*4+group]=b>>16;}
   for(int col=0;col<cols;col++){uint32_t q=(r*37+col*19+7)&((1<<bits)-1);int bit=col*bits,at=r*(cols*bits/32)+bit/32,shift=bit%32;
    w[at]|=q<<shift;if(shift+bits>32)w[at+1]|=q>>(32-shift);
    uint32_t sb=uint32_t(sc[r*4+col/64])<<16,bb=uint32_t(bi[r*4+col/64])<<16;float scale,bias;memcpy(&scale,&sb,4);memcpy(&bias,&bb,4);
    reference[r*cols+col]=bf_round(float(q)*scale+bias);
   }
  }
  for(int n:{64,65,128,2048})for(float amplitude:{.2f,1e10f}){
   for(int i=0;i<n*cols;i++)input[i]=std::sin(i*.019f)*amplitude;
   auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];PShape s{uint32_t(rows),cols,(uint32_t)n,0,0,0,1e-6f,1e7f};p_gemv(&g,e,*m,x,y,s);[e endEncoding];
   if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
   auto *out=static_cast<float*>(y.contents);double diff2=0,norm2=0;
   for(int t=0;t<n;t++)for(int r=0;r<rows;r++){double expected=0;for(int col=0;col<cols;col++)expected+=double(reference[r*cols+col])*bf_round(input[t*cols+col]);
    double diff=out[t*rows+r]-expected;if(!std::isfinite(diff))return 1;diff2+=diff*diff;norm2+=expected*expected;
   }
   double relative=std::sqrt(diff2/std::max(norm2,1e-30));if(relative>2e-5){fprintf(stderr,"BF16 dense failed bits=%d n=%d amplitude=%g relative=%g\n",bits,n,amplitude,relative);return 1;}
   worstRelative=std::max(worstRelative,relative);checks++;
  }
  delete m;
 }
 // Both prefill and decode must write actual 16-bit cache elements, not a float
 // buffer that happens to contain rounded values. Guard elements catch overruns.
 auto options=[MTLCompileOptions new];NSError *failure=nil;
 auto lib=[c->device newLibraryWithSource:g_kv_source(&g) options:options error:&failure];if(!lib){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);return 1;}
 auto ps=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"graph_head_prepare"] error:&failure];if(!ps)return 1;
 auto raw=g_buffer(&g,8192),k=g_buffer(&g,512),v=g_buffer(&g,512),norm=g_buffer(&g,256),q=g_buffer(&g,4096);
 auto keys=g_kv_buffer(&g,2048*512),values=g_kv_buffer(&g,2048*512);
 for(int i=0;i<8192;i++)static_cast<float*>(raw.contents)[i]=std::sin(i*.13f);
 for(int i=0;i<512;i++){static_cast<float*>(k.contents)[i]=std::cos(i*.11f);static_cast<float*>(v.contents)[i]=std::sin(i*.07f)*1e8f;}
 std::fill_n(static_cast<float*>(norm.contents),256,1.0f);
 for(bool prefill:{false,true}){
  memset(keys.contents,0xa5,keys.length);memset(values.contents,0xa5,values.length);
  auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
  g_bind(e,raw,0);g_bind(e,k,1);g_bind(e,v,2);g_bind(e,norm,3);g_bind(e,norm,4);g_bind(e,q,5);g_bind(e,keys,6);g_bind(e,values,7);
  if(prefill){PShape s{0,0,1,31,0,0,1e-6f,1e7f};[e setBytes:&s length:sizeof(s) atIndex:8];p_run(&g,e,PHead,MTLSizeMake(18,1,1),256);}
  else{GStep s{31,0,0,1e-6f,1e7f};[e setBytes:&s length:sizeof(s) atIndex:8];[e setComputePipelineState:ps];[e dispatchThreadgroups:MTLSizeMake(18,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
  [e endEncoding];if(!q_complete(command,nullptr,error,sizeof(error)))return 1;
  auto *keyBits=static_cast<uint16_t*>(keys.contents);
  for(int h=0;h<2;h++){
   double sum=0;for(int d=0;d<256;d++){double value=static_cast<float*>(k.contents)[h*256+d];sum+=value*value;}
   double scale=1/std::sqrt(sum/256+1e-6);
   for(int d=0;d<256;d++){double value=static_cast<float*>(k.contents)[h*256+d]*scale;
    if(d<64){int frequency=d%32,partner=d<32?d+32:d-32;double angle=31/std::pow(1e7,double(frequency)/32);
     double other=static_cast<float*>(k.contents)[h*256+partner]*scale;value=value*std::cos(angle)+(d<32?-other:other)*std::sin(angle);}
    uint32_t b=uint32_t(keyBits[31*512+h*256+d])<<16;float actual;memcpy(&actual,&b,4);
    if(!std::isfinite(actual)||std::abs(actual-value)>.005*std::max(1.0,std::abs(value)))return 1;
   }
  }
  auto *actual=static_cast<uint16_t*>(values.contents);for(int i=0;i<2048*512;i++){
   if(i>=31*512&&i<32*512){float value=bf_round(static_cast<float*>(v.contents)[i-31*512]);uint32_t b;memcpy(&b,&value,4);if(actual[i]!=(b>>16))return 1;}
   else if(actual[i]!=0xa5a5||keyBits[i]!=0xa5a5)return 1;
  }
  checks++;
 }
 for(int i=3;i<40;i+=4){g.layers[i].loaded=true;g.layers[i].linear=false;}
 if(di_graph_reserve(&g,8192,error,sizeof(error)))return 1;
 int capacity;uint64_t kvBytes,allocated;if(di_graph_stats(&g,&capacity,&kvBytes,&allocated)||capacity!=8192||kvBytes!=uint64_t(8192)*20480)return 1;
 for(int i=3;i<40;i+=4)if(g.layers[i].keys.length!=NSUInteger(8192)*512*2||g.layers[i].values.length!=NSUInteger(8192)*512*2)return 1;
 g.position=1;if(di_graph_reserve(&g,16384,error,sizeof(error))==0)return 1;g.position=0;
 if(di_graph_reserve(&g,262145,error,sizeof(error))==0)return 1;checks++;
 printf("PASS: %d BF16 checks (5/8-bit dense GEMM, tails, 1e10 range, FP32 accumulation, prefill/decode cache writes, 16-bit KV allocations/bounds); worst relative L2 %.9g\n",checks,worstRelative);delete c;
}}
