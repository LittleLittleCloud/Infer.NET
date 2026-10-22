// Synthetic expert geometry sweep. Isolated kernel times are not model throughput.
#include "bridge.mm"
#include <vector>
int main(){@autoreleasepool {
 setenv("DEEPINFER_METAL_PREFILL_GEMM","bf16-nax",1);setenv("DEEPINFER_METAL_PREFILL_GEMM_IMPL","hybrid",1);setenv("DEEPINFER_METAL_PREFILL_EXPERT_IMPL","packed",1);
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c)return 1;
 GModel g{};g.compute=*c;if(!p_create(&g,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
 std::vector<QMatrix*> matrices;
 for(auto shape:{std::pair<int,int>{512,2048},{512,2048},{2048,512}}){
  auto *m=static_cast<QMatrix*>(di_quant_matrix_create(c,256,shape.first,shape.second,5,64,error,sizeof(error)));if(!m)return 1;matrices.push_back(m);
  auto *w=static_cast<uint32_t*>(m->packed.contents);for(NSUInteger i=0;i<m->packed.length/4;i++)w[i]=uint32_t(i*2654435761u);
  auto *sc=static_cast<uint16_t*>(m->scales.contents),*bi=static_cast<uint16_t*>(m->biases.contents);
  for(NSUInteger i=0;i<m->scales.length/2;i++){sc[i]=0x3b00;bi[i]=0xbd00;}
 }
 auto x=g_buffer(&g,2048*8*2048),y=g_buffer(&g,2048*8*2048);
 auto *input=static_cast<float*>(x.contents);for(NSUInteger i=0;i<x.length/4;i++)input[i]=std::sin(i*.021f)*.02f;
 memcpy(g.prefill->norm.contents,x.contents,2048*2048*4);
 printf("{\"purpose\":\"isolated synthetic 5-bit expert geometry sweep, not model throughput\",\"warmupExcluded\":1,\"samples\":5,\"rows\":[\n");int index=0;
 for(int math:{0})for(int rows:{32})for(int inner:{64})for(int groups:{2,4,8}){
  if(rows*inner*4>32768)continue;
  NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=math?MTLMathModeRelaxed:MTLMathModeSafe;if(@available(macOS 26.0,*)){options.languageVersion=MTLLanguageVersion4_0;}else return 1;
  auto source=[[NSString stringWithFormat:@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS 64\n#define PACKED_EXPERT_ROWS %d\n#define PACKED_EXPERT_K %d\n#define PACKED_EXPERT_SG %d\n",rows,inner,groups] stringByAppendingString:[[quant_source stringByAppendingString:prefill_source] stringByAppendingString:[prefill_expert_packed_source stringByReplacingOccurrencesOfString:@"execution_simdgroups<2>> op" withString:@"execution_simdgroups<PACKED_EXPERT_SG>> op"]]];
  auto lib=[c->device newLibraryWithSource:source options:options error:&failure];if(!lib){fprintf(stderr,"%s\n",failure.localizedDescription.UTF8String);return 1;}
  id<MTLComputePipelineState> ps[2];for(int i=0;i<2;i++){ps[i]=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:i?@"packed_expert_down":@"packed_expert_gate_fused"] error:&failure];if(!ps[i])return 1;}
  for(int pattern:{0,1,2}){
   auto *ids=static_cast<uint32_t*>(g.prefill->ids.contents);for(int t=0;t<2048;t++)for(int slot=0;slot<8;slot++)ids[t*8+slot]=pattern==0?slot:pattern==1?(t*7+slot*13)%256:(t/5*11+slot*19)%256;
   for(int down=0;down<1;down++){
    auto &m=*matrices[down?2:0];PShape shape{m.rows,m.columns,2048,0,0,0,1e-6f,1e7f,2048};
    static_cast<uint32_t*>(g.prefill->groupedTileCount.contents)[0]=0;
    auto command=[c->queue commandBuffer];auto e=[command computeCommandEncoder];
    p_build_groups(&g,e,shape,512,2048);p_pack_expert(&g,e,shape,!down,down?x:g.prefill->norm);
    [e endEncoding];if(!q_complete(command,nullptr,error,sizeof(error)))return 1;
    auto *counts=static_cast<uint32_t*>(g.prefill->groupedCounts.contents);uint32_t expected=0;for(int i=0;i<256;i++)expected+=(counts[i]+63)/64;
    if(expected>512||static_cast<uint32_t*>(g.prefill->groupedTileCount.contents)[0]!=expected)return 1;
    double total=0;
    for(int trial=0;trial<6;trial++){
     command=[c->queue commandBuffer];e=[command computeCommandEncoder];
     g_bind(e,m.packed,0);g_bind(e,m.scales,1);g_bind(e,m.biases,2);int at=3;
     if(!down){auto &up=*matrices[1];g_bind(e,up.packed,at++);g_bind(e,up.scales,at++);g_bind(e,up.biases,at++);}
     g_bind(e,g.prefill->expandedInput,at++);g_bind(e,y,at++);[e setBytes:&shape length:sizeof(shape) atIndex:at++];g_bind(e,g.prefill->groupedRoutes,at++);g_bind(e,g.prefill->groupedCounts,at++);g_bind(e,g.prefill->groupedTiles,at++);
     [e setComputePipelineState:ps[down]];[e dispatchThreadgroups:MTLSizeMake((m.rows+rows-1)/rows,expected,1) threadsPerThreadgroup:MTLSizeMake(groups*32,1,1)];
     [e endEncoding];double ms=0;if(!q_complete(command,&ms,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}if(trial)total+=ms/5;
    }
    printf("%s{\"relaxedMath\":%d,\"tileRows\":%d,\"inner\":%d,\"simdgroups\":%d,\"pattern\":%d,\"down\":%s,\"gpuMs\":%.6f}",index++?",\n":"",math,rows,inner,groups,pattern,down?"true":"false",total);fflush(stdout);
   }
  }
 }
 printf("\n]}\n");for(auto *m:matrices)delete m;delete c;return 0;
}}
