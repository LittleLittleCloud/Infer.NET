// Original FP32 chunkwise gated-delta rule. See docs/PREFILL-GEMM-DELTA.md
// for the algebra; no external inference or matrix runtime is used.
static NSString *prefill_delta_source = @R"metal(
#include <metal_simdgroup_matrix>
struct DShape {uint tokens,blocks;};
// Each head/block independently forms L, (I+L)^-1, W, U, and causal QK.
// A direct product of <=32 decays handles zero/underflow without division.
kernel void prefill_delta_prepare(device const float *conv [[buffer(0)]],device const float *decay [[buffer(1)]],
 device const float *beta [[buffer(2)]],device float *w [[buffer(3)]],device float *u [[buffer(4)]],
 device float *causal [[buffer(5)]],device float *factors [[buffer(6)]],constant DShape &s [[buffer(7)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint head=group.x,block=group.y,base=block*32,valid=min(32u,s.tokens-base),kh=head/2;
 uint tile=(block*32+head)*32;
 threadgroup float data[128*32],tri[32*32],power[32*32],qpart[32*8],a[32],b[32],prefix[32];
 for(uint i=tid;i<128*32;i+=128){uint d=i/32,t=i%32;data[i]=t<valid?conv[(base+t)*8192+2048+kh*128+d]:0;}
 if(tid<32){a[tid]=tid<valid?decay[(base+tid)*32+head]:1;b[tid]=tid<valid?beta[(base+tid)*32+head]:0;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float p=1,tail=1;for(uint j=0;j<=tid;j++)p*=a[j];for(uint j=tid+1;j<valid;j++)tail*=a[j];prefix[tid]=p;
  factors[(tile+tid)*2]=p;factors[(tile+tid)*2+1]=tid<valid?tail:0;}
 simdgroup_float8x8 kk[4],qk[4];
 #pragma clang loop unroll(full)
 for(uint j=0;j<4;j++){kk[j]=simdgroup_float8x8(0.0f);qk[j]=simdgroup_float8x8(0.0f);}
 for(uint inner=0;inner<16;inner++) {
  for(uint i=tid;i<32*8;i+=128){uint t=i/8,d=inner*8+i%8;qpart[i]=t<valid?conv[(base+t)*8192+kh*128+d]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 ak,aq;simdgroup_load(ak,data+inner*8*32+sg*8,32,ulong2(0),true);simdgroup_load(aq,qpart+sg*8*8,8);
  #pragma clang loop unroll(full)
  for(uint j=0;j<4;j++){simdgroup_float8x8 bk;simdgroup_load(bk,data+inner*8*32+j*8,32);
   simdgroup_multiply_accumulate(kk[j],ak,bk,kk[j]);simdgroup_multiply_accumulate(qk[j],aq,bk,qk[j]);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint j=0;j<4;j++){simdgroup_store(kk[j],tri+sg*8*32+j*8,32);simdgroup_store(qk[j],power+sg*8*32+j*8,32);}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<32*32;i+=128){uint t=i/32,j=i%32;float d=0;
  if(t<valid&&j<=t){d=1;for(uint z=j+1;z<=t;z++)d*=a[z];}
  causal[tile*32+i]=power[i]*d;
  tri[i]=t>j?b[t]*d*tri[i]:0;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 // Independent columns of a unit-lower-triangular inverse. Forward solves
 // avoid the large cancelling powers in a nilpotent Neumann expansion when
 // adjacent keys are almost identical. Only the small 32x32 solve is ordered;
 // all value dimensions and state/output products below use parallel GEMM.
 if(tid<32){
  for(uint t=0;t<32;t++)power[t*32+tid]=t==tid?1:0;
  for(uint t=tid+1;t<32;t++){float sum=tri[t*32+tid];
   for(uint j=tid+1;j<t;j++)sum+=tri[t*32+j]*power[j*32+tid];
   power[t*32+tid]=-sum;}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint kind=0;kind<2;kind++) {
  for(uint i=tid;i<32*128;i+=128){uint t=i/128,d=i%128;float value=0;
   if(t<valid)value=b[t]*(kind?conv[(base+t)*8192+4096+head*128+d]:prefix[t]*conv[(base+t)*8192+2048+kh*128+d]);data[i]=value;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 result[16];
  #pragma clang loop unroll(full)
  for(uint j=0;j<16;j++)result[j]=simdgroup_float8x8(0.0f);
  for(uint k=0;k<4;k++){simdgroup_float8x8 x;simdgroup_load(x,power+sg*8*32+k*8,32);
   #pragma clang loop unroll(full)
   for(uint j=0;j<16;j++){simdgroup_float8x8 v;simdgroup_load(v,data+k*8*128+j*8,128);simdgroup_multiply_accumulate(result[j],x,v,result[j]);}}
  #pragma clang loop unroll(full)
  for(uint j=0;j<16;j++)simdgroup_store(result[j],(kind?u:w)+tile*128+sg*8*128+j*8,128);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}
// One group owns 32 value rows of one head. State stays in FP32 matrix
// registers across 32-token blocks; only block boundaries are sequential.
#if !DELTA_DEVICE
kernel void prefill_delta_apply(device float *state [[buffer(0)]],device const float *conv [[buffer(1)]],
 device const float *w [[buffer(2)]],device const float *u [[buffer(3)]],device const float *causal [[buffer(4)]],
 device const float *factors [[buffer(5)]],device float *out [[buffer(6)]],device float *history [[buffer(7)]],
 device const float *raw [[buffer(8)]],constant DShape &s [[buffer(9)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint head=group.x,row=group.y*32,kh=head/2;threadgroup float temp[128*32];simdgroup_float8x8 st[16];
 #pragma clang loop unroll(full)
 for(uint k=0;k<16;k++)simdgroup_load(st[k],state+(head*128+row+sg*8)*128+k*8,128);
 for(uint block=0;block<s.blocks;block++) {
  uint base=block*32,valid=min(32u,s.tokens-base),tile=(block*32+head)*32;
  for(uint i=tid;i<32*32;i+=128){uint v=i/32,t=i%32;temp[i]=u[(tile+t)*128+row+v];}
  threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_float8x8 update[4],result[4];
  #pragma clang loop unroll(full)
  for(uint j=0;j<4;j++){simdgroup_load(update[j],temp+sg*8*32+j*8,32);result[j]=simdgroup_float8x8(0.0f);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<128*32;i+=128){uint d=i/32,t=i%32;temp[i]=-w[(tile+t)*128+d];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(uint k=0;k<16;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*32+j*8,32);simdgroup_multiply_accumulate(update[j],st[k],b,update[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<128*32;i+=128){uint d=i/32,t=i%32;temp[i]=t<valid?factors[(tile+t)*2]*conv[(base+t)*8192+kh*128+d]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(uint k=0;k<16;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*32+j*8,32);simdgroup_multiply_accumulate(result[j],st[k],b,result[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*32;i+=128){uint k=i/32,t=i%32;temp[i]=causal[tile*32+t*32+k];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(uint k=0;k<4;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*32+j*8,32);simdgroup_multiply_accumulate(result[j],update[k],b,result[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(uint j=0;j<4;j++)simdgroup_store(result[j],temp+sg*8*32+j*8,32);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*32;i+=128){uint v=i/32,t=i%32;if(t<valid)out[(base+t)*4096+head*128+row+v]=temp[i];}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*128;i+=128){uint t=i/128,d=i%128;temp[i]=t<valid?factors[(tile+t)*2+1]*conv[(base+t)*8192+2048+kh*128+d]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);float finalDecay=factors[(tile+valid-1)*2];
  #pragma clang loop unroll(full)
  for(uint j=0;j<16;j++){st[j].thread_elements()*=finalDecay;
   #pragma clang loop unroll(full)
   for(uint k=0;k<4;k++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*128+j*8,128);simdgroup_multiply_accumulate(st[j],update[k],b,st[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint k=0;k<16;k++)simdgroup_store(st[k],state+(head*128+row+sg*8)*128+k*8,128);
 uint d=(head*4+group.y)*128+tid;
 if(d<8192){float next[3];for(uint j=0;j<3;j++){uint at=j+s.tokens;next[j]=at<3?history[at*8192+d]:raw[(at-3)*8192+d];}
  for(uint j=0;j<3;j++)history[j*8192+d]=next[j];}
}
#else
// Direct SIMD matrix loads bypass redundant threadgroup transposes.
kernel void prefill_delta_apply(device float *state [[buffer(0)]],device const float *conv [[buffer(1)]],
 device const float *w [[buffer(2)]],device const float *u [[buffer(3)]],device const float *causal [[buffer(4)]],
 device const float *factors [[buffer(5)]],device float *out [[buffer(6)]],device float *history [[buffer(7)]],
 device const float *raw [[buffer(8)]],constant DShape &s [[buffer(9)]],uint2 group [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint head=group.x,row=group.y*32,kh=head/2;threadgroup float temp[128*32];simdgroup_float8x8 st[16];
 #pragma clang loop unroll(full)
 for(uint k=0;k<16;k++)simdgroup_load(st[k],state+(head*128+row+sg*8)*128+k*8,128);
 for(uint block=0;block<s.blocks;block++) {
  uint base=block*32,valid=min(32u,s.tokens-base),tile=(block*32+head)*32;
  simdgroup_float8x8 update[4],result[4];
  #pragma clang loop unroll(full)
  for(uint j=0;j<4;j++){simdgroup_load(update[j],u+(tile+j*8)*128+row+sg*8,128,ulong2(0),true);result[j]=simdgroup_float8x8(0.0f);}

  #pragma clang loop unroll(full)
  for(uint k=0;k<16;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,w+(tile+j*8)*128+k*8,128,ulong2(0),true);b.thread_elements()*=-1.0f;simdgroup_multiply_accumulate(update[j],st[k],b,update[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<128*32;i+=128){uint d=i/32,t=i%32;temp[i]=t<valid?factors[(tile+t)*2]*conv[(base+t)*8192+kh*128+d]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(uint k=0;k<16;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*32+j*8,32);simdgroup_multiply_accumulate(result[j],st[k],b,result[j]);}}

  #pragma clang loop unroll(full)
  for(uint k=0;k<4;k++){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++){simdgroup_float8x8 b;simdgroup_load(b,causal+tile*32+j*8*32+k*8,32,ulong2(0),true);simdgroup_multiply_accumulate(result[j],update[k],b,result[j]);}}
  if(valid==32){
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++)simdgroup_store(result[j],out+(base+j*8)*4096+head*128+row+sg*8,4096,ulong2(0),true);
  }else{
   // Only the final partial block needs guarded elementwise stores.
   threadgroup_barrier(mem_flags::mem_threadgroup);
   #pragma clang loop unroll(full)
   for(uint j=0;j<4;j++)simdgroup_store(result[j],temp+sg*8*32+j*8,32);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for(uint i=tid;i<32*32;i+=128){uint v=i/32,t=i%32;if(t<valid)out[(base+t)*4096+head*128+row+v]=temp[i];}
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<32*128;i+=128){uint t=i/128,d=i%128;temp[i]=t<valid?factors[(tile+t)*2+1]*conv[(base+t)*8192+2048+kh*128+d]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);float finalDecay=factors[(tile+valid-1)*2];
  #pragma clang loop unroll(full)
  for(uint j=0;j<16;j++){st[j].thread_elements()*=finalDecay;
   #pragma clang loop unroll(full)
   for(uint k=0;k<4;k++){simdgroup_float8x8 b;simdgroup_load(b,temp+k*8*128+j*8,128);simdgroup_multiply_accumulate(st[j],update[k],b,st[j]);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint k=0;k<16;k++)simdgroup_store(st[k],state+(head*128+row+sg*8)*128+k*8,128);
 uint d=(head*4+group.y)*128+tid;
 if(d<8192){float next[3];for(uint j=0;j<3;j++){uint at=j+s.tokens;next[j]=at<3?history[at*8192+d]:raw[(at-3)*8192+d];}
  for(uint j=0;j<3;j++)history[j*8192+d]=next[j];}
}
#endif

)metal";
