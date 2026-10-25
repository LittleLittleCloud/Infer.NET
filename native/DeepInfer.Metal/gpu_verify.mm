// Exact-decode multi-row verification support. Compiled lazily by di_graph_prepare_verify;
// a graph that never enables MTP allocates none of this.
//
// The decode path computes the SwiGLU activation as gate*(1/(1+exp(-gate)))*up, while the
// prefill kernels compute (gate/(1+exp(-gate)))*up. These differ in the last ulp, so the
// verification kernels are derived from the prefill kernels with exactly that expression
// swapped (a textual transform that must match, or preparation fails).
static NSString *verify_transform(NSString *source,int *replacements) {
 NSString *out=source;int n=0;
 NSArray<NSArray<NSString*>*> *edits=@[
  @[@"(gate[i]/(1.0f+exp(-gate[i])))*up[i]",@"(gate[i]*(1.0f/(1.0f+exp(-gate[i]))))*up[i]"],
  @[@"(gate/(1.0f+exp(-gate)))*up",@"(gate*(1.0f/(1.0f+exp(-gate))))*up"],
  @[@"kernel void prefill_shared(",@"kernel void exact_shared("],
  @[@"PREFILL_EXPERTS(prefill_experts5,5)",@"PREFILL_EXPERTS(exact_experts5,5)"],
  @[@"PREFILL_EXPERTS(prefill_experts8,8)",@"PREFILL_EXPERTS(exact_experts8,8)"]];
 for(NSArray<NSString*> *edit in edits){
  if([out rangeOfString:edit[0]].location==NSNotFound){*replacements=-1;return nil;}
  out=[out stringByReplacingOccurrencesOfString:edit[0] withString:edit[1]];n++;
 }
 *replacements=n;return out;
}
extern "C" int di_graph_prepare_verify(void *graph,char *error,int capacity) {
 @autoreleasepool {
  if(!graph){error_text(error,capacity,@"Invalid graph");return -1;}
  auto *g=static_cast<GModel*>(graph);
  if(!g->prefill&&!p_create(g,error,capacity))return -1;
  auto *p=g->prefill;if(p->exactShared)return 0;
  int n=0;NSString *transformed=verify_transform(prefill_source,&n);
  if(!transformed){error_text(error,capacity,@"Prefill kernel source no longer matches the exact-decode transform");return -1;}
  NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  NSString *source=[[NSString stringWithFormat:@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS %u\n#define PREFILL_ROUTE_TILE %u\n",p->matrixRows,16u] stringByAppendingString:[quant_source stringByAppendingString:transformed]];
  auto lib=[g->compute.device newLibraryWithSource:source options:options error:&failure];
  if(!lib){error_text(error,capacity,failure.localizedDescription);return -1;}
  id<MTLComputePipelineState> built[3]={nil,nil,nil};NSString *names[3]={@"exact_shared",@"exact_experts5",@"exact_experts8"};
  for(int i=0;i<3;i++){
   built[i]=[g->compute.device newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&failure];
   if(!built[i]||built[i].threadExecutionWidth!=32||built[i].maxTotalThreadsPerThreadgroup<128){error_text(error,capacity,failure.localizedDescription?:@"Unsupported exact verification pipeline");return -1;}
  }
  p->exactShared=built[0];p->exactExperts5=built[1];p->exactExperts8=built[2];
  return 0;
 }
}
