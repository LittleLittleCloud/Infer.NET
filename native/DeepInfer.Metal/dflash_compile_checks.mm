// Compile-only check for the Issue #80 DFlash2 and exact-verification Metal sources. It creates shader libraries and
// pipeline states but allocates no buffers, encodes no commands and executes nothing on the GPU.
#include "bridge.mm"
int main(){@autoreleasepool {
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c){fprintf(stderr,"context: %s\n",error);return 1;}
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;int failures=0;
 auto lib=[c->device newLibraryWithSource:dflash_source options:options error:&failure];
 if(!lib){fprintf(stderr,"dflash library: %s\n",failure.localizedDescription.UTF8String);return 1;}
 struct Need {NSString *name;NSUInteger threads;};
 Need needs[]={{@"df_gemm",128},{@"df_rms",256},{@"df_norm_rope",128},{@"df_ring_copy",1},{@"df_dynconv",1},{@"df_silu_mul",1},{@"df_copy",1},{@"df_tap",256},
  {@"df_attn_part",128},{@"df_attn_combine",128},{@"df_topk1",256},{@"df_topk2",256},{@"df_selector",512}};
 for(auto &n:needs){
  auto ps=[c->device newComputePipelineStateWithFunction:[lib newFunctionWithName:n.name] error:&failure];
  if(!ps||ps.threadExecutionWidth!=32||ps.maxTotalThreadsPerThreadgroup<n.threads){fprintf(stderr,"FAIL pipeline %s: %s\n",n.name.UTF8String,failure.localizedDescription.UTF8String);failures++;}
  else printf("ok pipeline %s maxThreads=%lu\n",n.name.UTF8String,(unsigned long)ps.maxTotalThreadsPerThreadgroup);
 }
 int replacements=0;NSString *transformed=verify_transform(prefill_source,&replacements);
 if(!transformed){fprintf(stderr,"FAIL verify transform no longer matches prefill source\n");return 1;}
 NSString *source=[@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS 64\n#define PREFILL_ROUTE_TILE 16\n" stringByAppendingString:[quant_source stringByAppendingString:transformed]];
 auto exact=[c->device newLibraryWithSource:source options:options error:&failure];
 if(!exact){fprintf(stderr,"exact library: %s\n",failure.localizedDescription.UTF8String);return 1;}
 for(NSString *name in @[@"exact_shared",@"exact_experts5",@"exact_experts8"]){
  auto ps=[c->device newComputePipelineStateWithFunction:[exact newFunctionWithName:name] error:&failure];
  if(!ps||ps.threadExecutionWidth!=32||ps.maxTotalThreadsPerThreadgroup<128){fprintf(stderr,"FAIL pipeline %s: %s\n",name.UTF8String,failure.localizedDescription.UTF8String);failures++;}
  else printf("ok pipeline %s\n",name.UTF8String);
 }
 printf(failures?"dflash compile checks FAILED\n":"dflash compile checks passed (%d transforms)\n",replacements);return failures?1:0;
}}
