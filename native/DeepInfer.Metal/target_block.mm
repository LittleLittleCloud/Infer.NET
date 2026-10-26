// Opt-in target-only block verification. Live recurrent state is immutable until
// commit; each recurrent SIMD lane writes its intermediate values directly to
// the prefix slots. KV suffix writes remain invisible behind graph.position.
// The exact SwiGLU source transform follows our unadopted #80 implementation
// (c706064); no draft network or state snapshot/replay code is imported.
struct GTargetBlock {
 std::array<id<MTLComputePipelineState>,8> ps{};
 id<MTLComputePipelineState> history,copy;
 std::array<id<MTLBuffer>,40> states{},raw{};
 id<MTLBuffer> logits,next;
 uint32_t maxRows=0,rows=0,position=0;
 uint64_t bytes=0;
};
static void g_block_destroy(GTargetBlock *b){delete b;}
static constexpr uint32_t blockStateElements=32*128*128;
static constexpr uint64_t blockAllocationLimit=uint64_t(38.5*1024*1024*1024);
// Preserve the scalar expressions and reduction trees while adding a row axis
// to stateless kernels. Every rewrite is checked; source drift fails prepare.
static NSString *block_replace(NSString *s,NSString *from,NSString *to){
 if(!s||[s rangeOfString:from].location==NSNotFound)return nil;
 return [s stringByReplacingOccurrencesOfString:from withString:to];
}
static NSString *block_once(NSString *s,NSString *from,NSString *to){
 if(!s)return nil;NSRange r=[s rangeOfString:from];if(r.location==NSNotFound)return nil;
 return [s stringByReplacingCharactersInRange:r withString:to];
}
static NSString *block_kernel(NSString *name,NSString *group,NSString *setup){
 NSString *start=[@"kernel void " stringByAppendingString:name];NSRange r=[graph_source rangeOfString:start];
 if(r.location==NSNotFound)return nil;
 NSRange tail=NSMakeRange(r.location+r.length,graph_source.length-r.location-r.length);
 NSRange end=[graph_source rangeOfString:@"\nkernel void " options:0 range:tail];
 if(end.location==NSNotFound)return nil;
 NSString *s=[graph_source substringWithRange:NSMakeRange(r.location,end.location-r.location)];
 s=block_replace(s,start,[@"kernel void block_" stringByAppendingString:name]);
 if(group)s=block_replace(s,group,@"uint3 block [[threadgroup_position_in_grid]]");
 else s=block_once(s,@") {",@", uint3 block [[threadgroup_position_in_grid]]) {");
 s=block_once(s,@") {",[@") {\n uint token=block.y;\n " stringByAppendingString:setup]);
 return s;
}
static NSString *block_graph_source(){
 NSArray *kernels=@[
  block_kernel(@"graph_norm",nil,@"x+=token*2048;out+=token*2048;" )?:@"",
  block_kernel(@"graph_add_norm",nil,@"x+=token*2048;add+=token*2048;out+=token*2048;" )?:@"",
  block_kernel(@"graph_merge_norm",nil,@"x+=token*2048;expert+=token*16384;shared+=token*2048;route+=token;prob+=token*8;out+=token*2048;trace+=token*40*2048;" )?:@"",
  block_kernel(@"graph_delta_norm",@"uint head [[threadgroup_position_in_grid]]",@"uint head=block.x;x+=token*4096;z+=token*4096;" )?:@"",
  block_kernel(@"graph_head_prepare",@"uint head [[threadgroup_position_in_grid]]",@"uint head=block.x;raw+=token*8192;k+=token*512;v+=token*512;q+=token*4096;" )?:@"",
  block_kernel(@"graph_top8",nil,@"logits+=token*256;ids+=token*8;prob+=token*8;" )?:@""
 ];
 NSMutableString *result=[NSMutableString string];
 for(NSString *k in kernels){if(!k.length)return nil;[result appendString:k];}
 NSString *s=block_kernel(@"graph_linear_prepare",@"uint head [[threadgroup_position_in_grid]]",
  @"uint head=block.x;out+=token*8192;a+=token*32;b+=token*32;\n device const float *r=raw+token*8192;\n");
 s=block_replace(s,@"history[d]*conv[d*4]+history[8192+d]*conv[d*4+1]+\n        history[16384+d]*conv[d*4+2]+raw[d]*conv[d*4+3]",
  @"(token<3?history[token*8192+d]:raw[(token-3)*8192+d])*conv[d*4]+\n        (token<2?history[(token+1)*8192+d]:raw[(token-2)*8192+d])*conv[d*4+1]+\n        (token<1?history[(token+2)*8192+d]:raw[(token-1)*8192+d])*conv[d*4+2]+r[d]*conv[d*4+3]");
 s=block_replace(s,@"history[d]=history[8192+d];history[8192+d]=history[16384+d];history[16384+d]=raw[d];",@"");
 if(!s)return nil;[result appendString:s];
 // Head positions must be absolute within the block; the other kernels do not
 // use Step.position. Replace only in the derived head kernel above.
 return block_replace(result,@"s.position",@"(s.position+token)");
}
static NSString *block_state_source=@R"metal(
kernel void block_copy(device const float *input [[buffer(0)]],device float *output [[buffer(1)]],
 constant uint &elements [[buffer(2)]],uint at [[thread_position_in_grid]]) {if(at<elements)output[at]=input[at];}

kernel void block_delta(device const float *state [[buffer(0)]],device const float *conv [[buffer(1)]],
 device const float *decay [[buffer(2)]],device const float *beta [[buffer(3)]],device float *out [[buffer(4)]],
 device float *prefix [[buffer(5)]],constant uint &tokens [[buffer(6)]],
 uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group*2+sg,head=row/128,kh=head/2;
 float values[4];for(uint j=0;j<4;j++)values[j]=state[row*128+lane+32*j];
 for(uint t=0;t<tokens;t++){
  float prediction=0;
  for(uint j=0;j<4;j++){uint d=lane+32*j;values[j]=values[j]*decay[t*32+head];prediction+=values[j]*conv[t*8192+2048+kh*128+d];}
  prediction=simd_sum(prediction);float correction=(conv[t*8192+4096+row]-prediction)*beta[t*32+head];float result=0;
  for(uint j=0;j<4;j++){uint d=lane+32*j;float value=values[j]+conv[t*8192+2048+kh*128+d]*correction;
   prefix[t*524288+row*128+d]=value;values[j]=value;result+=value*conv[t*8192+kh*128+d];}
  result=simd_sum(result);if(lane==0)out[t*4096+row]=result;
 }
}
kernel void block_commit_history(device float *history [[buffer(0)]],device const float *raw [[buffer(1)]],
 constant uint &accepted [[buffer(2)]],uint d [[thread_position_in_grid]]) {
 if(d>=8192)return;float next[3];
 for(uint j=0;j<3;j++){uint at=j+accepted;next[j]=at<3?history[at*8192+d]:raw[(at-3)*8192+d];}
 for(uint j=0;j<3;j++)history[j*8192+d]=next[j];
}
)metal";
extern "C" int di_graph_prepare_target_block(void *graph,int maxRows,char *error,int capacity){
 @autoreleasepool {
  if(!graph||(maxRows!=1&&maxRows!=4&&maxRows!=8)){error_text(error,capacity,@"Target block capacity must be 1, 4 or 8");return -1;}
  auto *g=static_cast<GModel*>(graph);
  if(g->blockPending||g->blockFaulted||g->position||g->splitActive||g->kvBf16||!g->decodeFusedFp32||!g->prefill||g->forcedRoutes){error_text(error,capacity,@"Prepare target block after reset with fused FP32 decode and prepared FP32 scratch");return -1;}
  auto *p=g->prefill;
  if(p->maxTokens<uint32_t(maxRows)||p->bf16Gemm||p->expandedGemm||p->packedExperts||p->attentionBf16){error_text(error,capacity,@"Target block requires sufficient FP32 scratch and operands");return -1;}
  if(g->block){if(g->block->maxRows>=uint32_t(maxRows))return 0;error_text(error,capacity,@"Target block capacity is fixed for graph lifetime");return -1;}
  for(auto &l:g->layers)if(!l.loaded){error_text(error,capacity,@"Target block requires all layers loaded");return -1;}
  uint64_t bytes=uint64_t(maxRows)*(30ull*(blockStateElements+8192)*4+248320ull*4+4);
  if(g->compute.device.currentAllocatedSize+bytes>blockAllocationLimit){error_text(error,capacity,@"Target block exceeds 38.5 GiB allocation budget");return -1;}
  auto *b=new GTargetBlock{};b->maxRows=maxRows;b->bytes=bytes;
  NSError *failure=nil;auto options=[MTLCompileOptions new];options.mathMode=MTLMathModeSafe;
  NSString *derived=block_graph_source();
  if(!derived){delete b;error_text(error,capacity,@"Scalar source drift prevented target block derivation");return -1;}
  auto glib=[g->compute.device newLibraryWithSource:[[graph_source stringByAppendingString:derived] stringByAppendingString:block_state_source] options:options error:&failure];
  NSString *exact=block_replace(prefill_source,@"(gate[i]/(1.0f+exp(-gate[i])))*up[i]",@"(gate[i]*(1.0f/(1.0f+exp(-gate[i]))))*up[i]");
  exact=block_replace(exact,@"(gate/(1.0f+exp(-gate)))*up",@"(gate*(1.0f/(1.0f+exp(-gate))))*up");
  if(!exact){delete b;error_text(error,capacity,@"SwiGLU source drift prevented target block derivation");return -1;}
  NSString *qsource=[@"#define VECTOR_QUANT\n#define PREFILL_MATRIX_ROWS 64\n" stringByAppendingString:[quant_source stringByAppendingString:exact]];
  auto qlib=[g->compute.device newLibraryWithSource:qsource options:options error:&failure];
  if(!glib||!qlib){delete b;error_text(error,capacity,failure.localizedDescription);return -1;}
  NSArray *names=@[@"block_graph_norm",@"block_graph_add_norm",@"block_graph_merge_norm",@"block_graph_delta_norm",@"block_graph_head_prepare",@"block_graph_top8",@"block_graph_linear_prepare",@"block_delta"];
  for(int i=0;i<8;i++){b->ps[i]=[g->compute.device newComputePipelineStateWithFunction:[glib newFunctionWithName:names[i]] error:&failure];
   if(!b->ps[i]||b->ps[i].threadExecutionWidth!=32||b->ps[i].maxTotalThreadsPerThreadgroup<256){delete b;error_text(error,capacity,failure.localizedDescription?:@"Unsupported target block pipeline");return -1;}}
  // History commit has a separate kernel; store it outside the forward table.
  auto history=[g->compute.device newComputePipelineStateWithFunction:[glib newFunctionWithName:@"block_commit_history"] error:&failure];
  auto shared=[g->compute.device newComputePipelineStateWithFunction:[qlib newFunctionWithName:@"prefill_shared"] error:&failure];
  auto expert5=[g->compute.device newComputePipelineStateWithFunction:[qlib newFunctionWithName:@"prefill_experts5"] error:&failure];
  auto expert8=[g->compute.device newComputePipelineStateWithFunction:[qlib newFunctionWithName:@"prefill_experts8"] error:&failure];
  if(!history||!shared||!expert5||!expert8){delete b;error_text(error,capacity,failure.localizedDescription?:@"Target block pipeline creation failed");return -1;}
  b->history=history;b->copy=[g->compute.device newComputePipelineStateWithFunction:[glib newFunctionWithName:@"block_copy"] error:&failure];
  if(!b->copy){delete b;error_text(error,capacity,@"Cannot create target raw copy");return -1;}
  for(int i=0;i<40;i++)if(g->layers[i].linear){b->states[i]=g_buffer(g,NSUInteger(maxRows)*blockStateElements);b->raw[i]=g_buffer(g,NSUInteger(maxRows)*8192);
   if(!b->states[i]||!b->raw[i]){delete b;error_text(error,capacity,@"Target block state allocation failed");return -1;}}
  b->logits=g_buffer(g,NSUInteger(maxRows)*248320);b->next=g_buffer(g,maxRows);
  if(!b->logits||!b->next){delete b;error_text(error,capacity,@"Target block classifier allocation failed");return -1;}
  p->blockShared=shared;p->blockExperts5=expert5;p->blockExperts8=expert8;g->block=b;return 0;
 }
}
static void block_run(GModel *g,id<MTLComputeCommandEncoder> e,int kernel,int groups,int rows,int threads){
 [e setComputePipelineState:g->block->ps[kernel]];
 [e dispatchThreadgroups:MTLSizeMake(groups,rows,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];g->dispatches++;
}
extern "C" int di_graph_target_block(void *graph,const int32_t *tokens,int count,int32_t *next,float *logits,float *trace,
 int *dispatches,int *submissions,double *ms,char *error,int capacity){
 @autoreleasepool {
  if(!graph||!tokens||!next||!dispatches||!submissions||!ms||(count!=1&&count!=4&&count!=8)){error_text(error,capacity,@"Invalid target block arguments");return -1;}
  auto *g=static_cast<GModel*>(graph);auto *b=g->block;auto *p=g->prefill;
  if(!b||!p||g->blockPending||g->blockFaulted||g->splitMissing||g->splitActive||uint32_t(count)>b->maxRows||g->position+uint32_t(count)>g->capacity){error_text(error,capacity,@"Target block is unavailable, pending, faulted or exceeds reserved context");return -1;}
  for(int t=0;t<count;t++)if(tokens[t]<0||tokens[t]>=248320){error_text(error,capacity,@"Invalid target token");return -1;}
  if(trace&&p->trace.length<NSUInteger(count)*40*2048*4){p->trace=g_buffer(g,NSUInteger(count)*40*2048);if(!p->trace){error_text(error,capacity,@"Target trace allocation failed");return -1;}}
  struct Scope {GPrefill *p;~Scope(){p->blockEncoding=false;}} scope{p};p->blockEncoding=true;
  g->dispatches=0;g->profileCount=0;*ms=0;*submissions=0;
  b->rows=count;b->position=g->position;
  memcpy(p->tokens.contents,tokens,size_t(count)*4);
  PShape shape{0,0,uint32_t(count),g->position,0,uint32_t(trace!=nullptr),g->epsilon,g->theta,g->capacity};
  GStep s{g->position,0,uint32_t(trace!=nullptr),g->epsilon,g->theta,g->capacity};
  auto command=[g->compute.queue commandBuffer];auto e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  if(!command||!e){error_text(error,capacity,@"Cannot encode target block");return -1;}
  g_bind(e,g->embedding.packed,0);g_bind(e,g->embedding.scales,1);g_bind(e,g->embedding.biases,2);g_bind(e,p->tokens,3);g_bind(e,p->x,4);[e setBytes:&g->embedding.bits length:4 atIndex:5];
  p_run(g,e,PEmbed,MTLSizeMake(16,count,1),128);g_barrier(e);
  g_bind(e,p->x,0);g_bind(e,g->layers[0].parameters,1);g_bind(e,p->norm,2);[e setBytes:&s length:sizeof(s) atIndex:3];block_run(g,e,0,1,count,256);g_barrier(e);
  for(int i=0;i<40;i++){
   auto &l=g->layers[i];auto &m=l.m;shape.layer=i;s.layer=i;
   p_gemv(g,e,m[GQ],p->norm,p->qkv,shape);
   if(l.linear){
    p_gemv(g,e,m[GZ],p->norm,p->z,shape);p_gemv(g,e,m[GA],p->norm,p->a,shape);p_gemv(g,e,m[GB],p->norm,p->b,shape);g_barrier(e);
    g_bind(e,p->qkv,0);g_bind(e,l.history,1);g_bind(e,l.parameters,2,4096*4);g_bind(e,p->conv,3);g_bind(e,p->a,4);g_bind(e,p->b,5);g_bind(e,l.parameters,6,36864*4);g_bind(e,l.parameters,7,36896*4);
    block_run(g,e,6,64,count,128);g_barrier(e);
    g_bind(e,l.state,0);g_bind(e,p->conv,1);g_bind(e,p->a,2);g_bind(e,p->b,3);g_bind(e,p->attention,4);g_bind(e,b->states[i],5);[e setBytes:&count length:4 atIndex:6];block_run(g,e,7,2048,1,64);g_barrier(e);
    // Preserve just the projected convolution inputs needed by prefix commit.
    // Keep a bounded raw projection copy before shared qkv scratch is reused.
    g_bind(e,p->attention,0);g_bind(e,p->z,1);g_bind(e,l.parameters,2,36928*4);[e setBytes:&s length:sizeof(s) atIndex:3];block_run(g,e,3,32,count,128);g_barrier(e);
    // Contiguous device-to-device copy; no host state readback.
    [e setComputePipelineState:b->copy];g_bind(e,p->qkv,0);g_bind(e,b->raw[i],1);uint32_t elements=count*8192;[e setBytes:&elements length:4 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake((elements+255)/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
   }else{
    p_gemv(g,e,m[GK],p->norm,p->k,shape);p_gemv(g,e,m[GV],p->norm,p->v,shape);g_barrier(e);
    g_bind(e,p->qkv,0);g_bind(e,p->k,1);g_bind(e,p->v,2);g_bind(e,l.parameters,3,37056*4);g_bind(e,l.parameters,4,37312*4);g_bind(e,p->q,5);g_bind(e,l.keys,6);g_bind(e,l.values,7);[e setBytes:&s length:sizeof(s) atIndex:8];block_run(g,e,4,18,count,256);g_barrier(e);
    // Each query uses the identical scalar attention reduction and partition.
    for(uint32_t row=0;row<uint32_t(count);row++){
     AttentionShape a{g->position+row,1,g->capacity,(g->position+row+128)/128};
     g_bind(e,p->q,0,NSUInteger(row)*4096*4);g_bind(e,l.keys,1);g_bind(e,l.values,2);g_bind(e,g->attentionParts,3);[e setBytes:&a length:sizeof(a) atIndex:4];p_run(g,e,PDecodeAttention,MTLSizeMake(a.partitions,2,1),128);g_barrier(e);
     g_bind(e,g->attentionParts,0);g_bind(e,p->qkv,1,NSUInteger(row)*8192*4);g_bind(e,p->attention,2,NSUInteger(row)*4096*4);[e setBytes:&a length:sizeof(a) atIndex:3];
     [e setComputePipelineState:g->ps[GAttentionCombine]];[e dispatchThreadgroups:MTLSizeMake(16,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];g->dispatches++;g_barrier(e);
    }
   }
   p_gemv(g,e,m[GO],p->attention,p->out,shape);g_barrier(e);
   g_bind(e,p->x,0);g_bind(e,p->out,1);g_bind(e,l.parameters,2,2048*4);g_bind(e,p->norm,3);[e setBytes:&s length:sizeof(s) atIndex:4];block_run(g,e,1,1,count,256);g_barrier(e);
   p_gemv(g,e,m[GR],p->norm,p->router,shape);p_gate(g,e,m[GSG],m[GSU],p->sharedAct,shape,false);p_gemv(g,e,m[GSR],p->norm,p->sharedRoute,shape);g_barrier(e);
   g_bind(e,p->router,0);g_bind(e,p->ids,1);g_bind(e,p->prob,2);block_run(g,e,5,1,count,32);g_barrier(e);
   p_gate(g,e,m[GEG],m[GEU],p->expertAct,shape,true);g_barrier(e);p_gemv(g,e,m[GED],p->expertAct,p->expertOut,shape,true);p_gemv(g,e,m[GSD],p->sharedAct,p->sharedOut,shape);g_barrier(e);
   g_bind(e,p->x,0);g_bind(e,p->expertOut,1);g_bind(e,p->sharedOut,2);g_bind(e,p->sharedRoute,3);g_bind(e,p->prob,4);g_bind(e,i==39?g->finalNorm:g->layers[i+1].parameters,5);g_bind(e,p->norm,6);g_bind(e,p->trace,7);[e setBytes:&s length:sizeof(s) atIndex:8];block_run(g,e,2,1,count,256);g_barrier(e);
  }
  [e endEncoding];double layers=0;g->blockFaulted=true;
  if(!q_complete(command,&layers,error,capacity))return -1;*ms+=layers;(*submissions)++;
  command=[g->compute.queue commandBuffer];e=[command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  if(!command||!e){error_text(error,capacity,@"Cannot encode target classifier");return -1;}
  p_gemv(g,e,g->classifier,p->norm,b->logits,shape);g_barrier(e);
  for(int row=0;row<count;row++){
   g_bind(e,b->logits,0,NSUInteger(row)*248320*4);g_bind(e,g->argValues,1);g_bind(e,g->argIds,2);g_run(g,e,GArgBlocks,970,32);g_barrier(e);
   g_bind(e,g->argValues,0);g_bind(e,g->argIds,1);g_bind(e,b->next,2,NSUInteger(row)*4);g_run(g,e,GArgFinal,1,32);g_barrier(e);
  }
  [e endEncoding];double head=0;if(!q_complete(command,&head,error,capacity))return -1;*ms+=head;(*submissions)++;
  auto *ids=static_cast<const int32_t*>(b->next.contents);
  for(int row=0;row<count;row++)if(ids[row]<0||ids[row]>=248320){error_text(error,capacity,@"Target block produced invalid/non-finite logits");return -1;}
  memcpy(next,ids,size_t(count)*4);if(logits)memcpy(logits,b->logits.contents,size_t(count)*248320*4);if(trace)memcpy(trace,p->trace.contents,size_t(count)*40*2048*4);
  *dispatches=g->dispatches;g->blockFaulted=false;g->blockPending=true;return 0;
 }
}
extern "C" int di_graph_commit_target_block(void *graph,int accepted,double *ms,char *error,int capacity){
 @autoreleasepool {
  if(!graph||!ms){error_text(error,capacity,@"Invalid target commit");return -1;}auto *g=static_cast<GModel*>(graph);auto *b=g->block;*ms=0;
  if(!b||!g->blockPending||g->blockFaulted||accepted<0||uint32_t(accepted)>b->rows||g->position!=b->position){error_text(error,capacity,@"No matching pending target block/prefix");return -1;}
  if(!accepted){g->blockPending=false;return 0;}
  auto command=[g->compute.queue commandBuffer];auto blit=[command blitCommandEncoder];
  if(!command||!blit){error_text(error,capacity,@"Cannot create target commit command");return -1;}
  for(int i=0;i<40;i++)if(g->layers[i].linear)[blit copyFromBuffer:b->states[i] sourceOffset:NSUInteger(accepted-1)*blockStateElements*4 toBuffer:g->layers[i].state destinationOffset:0 size:blockStateElements*4];
  [blit copyFromBuffer:b->logits sourceOffset:NSUInteger(accepted-1)*248320*4 toBuffer:g->logits destinationOffset:0 size:248320*4];
  [blit copyFromBuffer:b->next sourceOffset:NSUInteger(accepted-1)*4 toBuffer:g->nextToken destinationOffset:0 size:4];[blit endEncoding];
  auto e=[command computeCommandEncoder];if(!e){error_text(error,capacity,@"Cannot encode target history commit");return -1;}
  for(int i=0;i<40;i++)if(g->layers[i].linear){[e setComputePipelineState:b->history];g_bind(e,g->layers[i].history,0);g_bind(e,b->raw[i],1);[e setBytes:&accepted length:4 atIndex:2];
   [e dispatchThreadgroups:MTLSizeMake(32,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];}
  [e endEncoding];g->blockFaulted=true;if(!q_complete(command,ms,error,capacity))return -1;
  g->position+=accepted;g->blockPending=false;g->blockFaulted=false;return 0;
 }
}
// Diagnostic state access is read-only and includes every layer, unlike the
// product's prefix-transfer ABI. Pending prefixes expose only captured rows.
extern "C" int di_graph_target_state(void *graph,int layer,int kind,int prefix,void *out,uint64_t bytes){
 if(!graph||!out||layer<0||layer>=40||kind<0||kind>1)return -1;auto *g=static_cast<GModel*>(graph);auto &l=g->layers[layer];
 if(g->blockFaulted)return -1;
 if(prefix<0){uint64_t expected=l.linear?(kind?3ull*8192*4:uint64_t(blockStateElements)*4):uint64_t(g->position)*512*4;
  if(bytes!=expected)return -1;auto buffer=l.linear?(kind?l.history:l.state):(kind?l.values:l.keys);memcpy(out,buffer.contents,bytes);return 0;}
 auto *b=g->block;if(!g->blockPending||!b||!l.linear||kind||prefix<1||uint32_t(prefix)>b->rows||bytes!=uint64_t(blockStateElements)*4)return -1;
 memcpy(out,static_cast<const char*>(b->states[layer].contents)+uint64_t(prefix-1)*bytes,bytes);return 0;
}
extern "C" int di_graph_target_block_stats(void *graph,uint64_t *bytes,int *position,int *pending){
 if(!graph||!bytes||!position||!pending)return -1;auto *g=static_cast<GModel*>(graph);*bytes=g->block?g->block->bytes:0;*position=g->position;*pending=g->blockPending;return 0;
}
// Controlled scalar transaction reference: execute every candidate row using
// the unchanged scalar graph, snapshot the selected prefix once, then restore
// it after evaluating the rejected suffix. All GPU copies are timed.
extern "C" int di_graph_scalar_target_transaction(void *graph,const int32_t *tokens,int count,int accepted,
 int *dispatches,int *submissions,double *ms,char *error,int capacity){
 @autoreleasepool {
  if(!graph||!tokens||!dispatches||!submissions||!ms||(count!=1&&count!=4&&count!=8)||accepted<1||accepted>count){error_text(error,capacity,@"Invalid scalar transaction");return -1;}
  auto *g=static_cast<GModel*>(graph);auto *b=g->block;
  if(!b||g->blockPending||g->blockFaulted||g->splitMissing||g->splitActive||uint32_t(count)>b->maxRows||g->position+uint32_t(count)>g->capacity){error_text(error,capacity,@"Scalar transaction unavailable");return -1;}
  for(int i=0;i<count;i++)if(tokens[i]<0||tokens[i]>=248320){error_text(error,capacity,@"Invalid scalar token");return -1;}
  const auto start=g->position;*ms=0;*dispatches=0;*submissions=0;
  auto copy=[&](bool restore)->bool{
   auto command=[g->compute.queue commandBuffer];auto e=[command blitCommandEncoder];if(!command||!e)return false;
   auto transfer=[&](id<MTLBuffer> live,id<MTLBuffer> saved,NSUInteger bytes){[e copyFromBuffer:restore?saved:live sourceOffset:0 toBuffer:restore?live:saved destinationOffset:0 size:bytes];};
   for(int i=0;i<40;i++)if(g->layers[i].linear){transfer(g->layers[i].state,b->states[i],blockStateElements*4);transfer(g->layers[i].history,b->raw[i],3*8192*4);}
   transfer(g->logits,b->logits,248320*4);transfer(g->nextToken,b->next,4);[e endEncoding];double elapsed=0;
   if(!q_complete(command,&elapsed,error,capacity))return false;*ms+=elapsed;(*submissions)++;return true;
  };
  for(int row=0;row<count;row++){
   int next=0,kernels=0;double elapsed=0;
   if(di_graph_forward(graph,tokens[row],int(start)+row,1,nullptr,nullptr,&next,&kernels,&elapsed,error,capacity)!=0){g->blockFaulted=true;return -1;}
   if(next<0||next>=248320){g->blockFaulted=true;error_text(error,capacity,@"Scalar reference logits invalid");return -1;}
   *ms+=elapsed;*dispatches+=kernels;(*submissions)++;
   if(accepted<count&&row+1==accepted&&!copy(false)){g->blockFaulted=true;return -1;}
  }
  if(accepted<count){if(!copy(true)){g->blockFaulted=true;return -1;}g->position=start+accepted;}
  return 0;
 }
}
