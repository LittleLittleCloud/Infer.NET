// Compile-only check for the Issue #76 MTP/verification Metal sources. It creates shader libraries and
// pipeline states but allocates no buffers, encodes no commands and executes nothing on the GPU.
#include "bridge.mm"
int main(){@autoreleasepool {
 char error[2048]{};auto *c=static_cast<QContext*>(di_quant_create(error,sizeof(error)));if(!c){fprintf(stderr,"context: %s\n",error);return 1;}
 NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;int failures=0;
 auto mtp=[c->device newLibraryWithSource:mtp_source options:options error:&failure];
 if(!mtp){fprintf(stderr,"mtp library: %s\n",failure.localizedDescription.UTF8String);return 1;}
 for(NSString *name in @[@"mtp_gemv",@"mtp_gateup",@"mtp_down",@"mtp_copy"]){
  auto ps=[c->device newComputePipelineStateWithFunction:[mtp newFunctionWithName:name] error:&failure];
  if(!ps||ps.threadExecutionWidth!=32){fprintf(stderr,"FAIL pipeline %s: %s\n",name.UTF8String,failure.localizedDescription.UTF8String);failures++;}
  else printf("ok pipeline %s maxThreads=%lu\n",name.UTF8String,(unsigned long)ps.maxTotalThreadsPerThreadgroup);
 }
 int replacements=0;NSString *transformed=verify_transform(prefill_source,&replacements);
 if(!transformed){fprintf(stderr,"FAIL verify transform no longer matches prefill source\n");return 1;}
 if([transformed rangeOfString:@"gate[i]*(1.0f/(1.0f+exp(-gate[i])))"].location==NSNotFound){fprintf(stderr,"FAIL transform did not change the shared activation\n");return 1;}
 NSString *source=[@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS 64\n#define PREFILL_ROUTE_TILE 16\n" stringByAppendingString:[quant_source stringByAppendingString:transformed]];
 auto exact=[c->device newLibraryWithSource:source options:options error:&failure];
 if(!exact){fprintf(stderr,"exact library: %s\n",failure.localizedDescription.UTF8String);return 1;}
 for(NSString *name in @[@"exact_shared",@"exact_experts5",@"exact_experts8"]){
  auto ps=[c->device newComputePipelineStateWithFunction:[exact newFunctionWithName:name] error:&failure];
  if(!ps||ps.threadExecutionWidth!=32||ps.maxTotalThreadsPerThreadgroup<128){fprintf(stderr,"FAIL pipeline %s: %s\n",name.UTF8String,failure.localizedDescription.UTF8String);failures++;}
  else printf("ok pipeline %s\n",name.UTF8String);
 }
 printf(failures?"mtp compile checks FAILED\n":"mtp compile checks passed (%d transforms)\n",replacements);return failures?1:0;
}}
