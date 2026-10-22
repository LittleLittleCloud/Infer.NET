// Grouped route gather/scatter and affine GEMM against an independent double oracle.
#include "bridge.mm"
#include <vector>
int main(int argc,char **argv){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_CHUNK","2048",1);setenv("DEEPINFER_METAL_PREFILL_EXPERTS","grouped",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 const int bits=argc>1?std::atoi(argv[1]):5;if(bits!=5&&bits!=8)return 1;
 constexpr int rows=129,columns=256;std::vector<QMatrix*> matrices;
 for(int kind=0;kind<3;kind++){
  auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,256,rows,columns,bits,64,error,sizeof(error)));if(!m)return 1;matrices.push_back(m);
  auto *w=static_cast<uint32_t*>(m->packed.contents);memset(w,0,m->packed.length);
  auto *scales=static_cast<uint16_t*>(m->scales.contents),*biases=static_cast<uint16_t*>(m->biases.contents);
  for(int e=0;e<256;e++)for(int r=0;r<rows;r++){
   for(int a=0;a<4;a++){float scale=(a+1)/4096.0f,bias=((e+r+kind)%7-3)/1024.0f;uint32_t b;memcpy(&b,&scale,4);scales[(e*rows+r)*4+a]=b>>16;memcpy(&b,&bias,4);biases[(e*rows+r)*4+a]=b>>16;}
   for(int col=0;col<columns;col++){uint32_t q=(e*3+r*7+col*5+kind*11)&((1<<bits)-1);int bit=col*bits,at=(e*rows+r)*(columns*bits/32)+bit/32,shift=bit%32;
    w[at]|=q<<shift;if(shift+bits>32)w[at+1]|=q>>(32-shift);}
  }
 }
 auto x=g_buffer(&g,2048*8*columns),y=g_buffer(&g,2048*8*rows);auto *input=static_cast<float*>(x.contents);
 for(int i=0;i<2048*8*columns;i++)input[i]=std::sin(i*.021f)*.02f;
 auto value=[&](int kind,int e,int r,int col){return double((e*3+r*7+col*5+kind*11)&((1<<bits)-1))*(col/64+1)/4096.0+((e+r+kind)%7-3)/1024.0;};
 bool bf16=g.prefill->bf16Gemm;
 auto roundbf=[](float value){uint32_t bits;memcpy(&bits,&value,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;memcpy(&value,&bits,4);return value;};
 double worst=0;int checks=0;
 for(int count:{64,65,127,128,129,512,513,2048})for(int pattern:{0,1,2})for(bool down:{false,true}){
  auto *ids=static_cast<uint32_t*>(g.prefill->ids.contents);for(int t=0;t<count;t++)for(int slot=0;slot<8;slot++)ids[t*8+slot]=pattern==0?slot:pattern==1?(t*7+slot*13)%256:(t/5*11+slot*19)%256;
  std::vector<bool> seen(count*8);std::vector<uint32_t> expectedIds(ids,ids+count*8);
  auto *router=static_cast<float*>(g.prefill->router.contents);std::fill(router,router+count*256,-1000.0f);
  for(int t=0;t<count;t++)for(int slot=0;slot<8;slot++)router[t*256+expectedIds[t*8+slot]]=100.0f-slot;
  static_cast<uint32_t*>(g.prefill->groupedTileCount.contents)[0]=999;
  PShape shape{rows,columns,(uint32_t)count,0,0,0,1e-6f,1e7f,2048};
  auto command=[c->queue commandBuffer];auto encoder=[command computeCommandEncoder];
  g_bind(encoder,g.prefill->router,0);g_bind(encoder,g.prefill->ids,1);g_bind(encoder,g.prefill->prob,2);g_bind(encoder,g.prefill->groupedTileCount,3);
  p_run(&g,encoder,PTop,MTLSizeMake(count,1,1),32);g_barrier(encoder);
  p_build_groups(&g,encoder,shape,rows,rows);
  if(down)p_gemv(&g,encoder,*matrices[2],x,y,shape,true);
  else {
   // p_gate's layer input is the model norm scratch.
   memcpy(g.prefill->norm.contents,x.contents,count*columns*4);
   p_gate(&g,encoder,*matrices[0],*matrices[1],y,shape,true);
  }
  [encoder endEncoding];if(!q_complete(command,nullptr,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
  if(!std::equal(expectedIds.begin(),expectedIds.end(),ids))return 1;
  auto *routes=static_cast<uint32_t*>(g.prefill->groupedRoutes.contents),*counts=static_cast<uint32_t*>(g.prefill->groupedCounts.contents);
  for(int expert=0;expert<256;expert++)for(uint32_t k=0;k<counts[expert];k++){
   auto route=routes[expert*count+k];if(route>=uint32_t(count*8)||ids[route]!=uint32_t(expert)||seen[route])return 1;seen[route]=true;
  }
  if(std::find(seen.begin(),seen.end(),false)!=seen.end())return 1;
  uint32_t tiles=0;for(int e=0;e<256;e++)tiles+=(counts[e]+(bf16?63:15))/(bf16?64:16);
  auto *args=static_cast<uint32_t*>(g.prefill->groupedDispatch.contents);
  if(args[0]!=(g.prefill->packedExperts?5u:3u)||args[1]!=tiles||args[2]!=1||args[4]!=(g.prefill->packedExperts?5u:3u)||args[5]!=tiles||args[6]!=1)return 1;
  auto *actual=static_cast<float*>(y.contents);
  for(int route=0;route<count*8;route++)for(int row=0;row<rows;row++){
   double a=0,b=0;for(int col=0;col<columns;col++){float v=input[(down?route:route/8)*columns+col];double w=value(down?2:0,ids[route],row,col),uw=value(1,ids[route],row,col);
    if(bf16){v=roundbf(v);w=roundbf(float(w));uw=roundbf(float(uw));}a+=w*v;if(!down)b+=uw*v;}
   double expected=down?a:a/(1+std::exp(-a))*b,diff=std::abs(expected-actual[route*rows+row]);worst=std::max(worst,diff);
   if(!std::isfinite(diff)||diff>2e-6){fprintf(stderr,"Grouped oracle failed n=%d pattern=%d down=%d route=%d row=%d expected=%.9g actual=%.9g\n",count,pattern,down,route,row,expected,actual[route*rows+row]);return 1;}
  }
  checks++;
 }
 if(g.prefill->expandedGemm&&bf16)for(int count:{64,65,127,128,129,512,513,2048}){
  memcpy(g.prefill->norm.contents,x.contents,count*columns*4);
  auto command=[c->queue commandBuffer];auto encoder=[command computeCommandEncoder];PShape shape{rows,columns,(uint32_t)count,0,0,0,1e-6f,1e7f,2048};
  p_gate(&g,encoder,*matrices[0],*matrices[1],y,shape,false);[encoder endEncoding];if(!q_complete(command,nullptr,error,sizeof(error)))return 1;
  auto *actual=static_cast<float*>(y.contents);
  for(int t=0;t<count;t++)for(int r=0;r<rows;r++){double gate=0,up=0;
   for(int col=0;col<columns;col++){double v=roundbf(input[t*columns+col]);gate+=roundbf(float(value(0,0,r,col)))*v;up+=roundbf(float(value(1,0,r,col)))*v;}
   double expected=gate/(1+std::exp(-gate))*up,diff=std::abs(expected-actual[t*rows+r]);if(!std::isfinite(diff)||diff>2e-6)return 1;worst=std::max(worst,diff);
  }checks++;
 }
 for(auto *m:matrices)delete m;delete c;printf("PASS: %d-bit, %d grouped expert checks (64..2048 tokens, hot/scattered/skewed routes, tails, gate/up/down, indirect grids); %s operands, max absolute %.9g\n",bits,checks,bf16?"BF16/FP32 accumulator":"FP32",worst);return 0;
}}
