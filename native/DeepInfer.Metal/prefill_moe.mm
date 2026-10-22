// Original affine-5/8-bit grouped expert GEMM, with GPU-only routing and
// indirect grids. Packed weights are reused across sixteen routed tokens.
static NSString *prefill_moe_source = @R"metal(
#include <metal_simdgroup_matrix>
#ifndef PREFILL_ROUTE_TILE
#define PREFILL_ROUTE_TILE 16
#endif
kernel void prefill_build_groups(device const uint *ids [[buffer(0)]],device uint *routes [[buffer(1)]],
 device uint *counts [[buffer(2)]],device uint2 *tiles [[buffer(3)]],device atomic_uint *tileCount [[buffer(4)]],
 constant PShape &s [[buffer(5)]],uint expert [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint total=0;
 for(uint base=0;base<s.tokens*8;base+=32){uint route=base+lane;bool selected=route<s.tokens*8&&ids[route]==expert;
  uint offset=simd_prefix_exclusive_sum(uint(selected)),count=simd_sum(uint(selected));
  if(selected)routes[expert*s.tokens+total+offset]=route;total+=count;}
 if(lane==0){counts[expert]=total;if(total){uint n=(total+PREFILL_ROUTE_TILE-1)/PREFILL_ROUTE_TILE,start=atomic_fetch_add_explicit(tileCount,n,memory_order_relaxed);
  for(uint i=0;i<n;i++)tiles[start+i]=uint2(expert,i*PREFILL_ROUTE_TILE);}}
}
kernel void prefill_group_dispatch(device const uint *tileCount [[buffer(0)]],device uint *args [[buffer(1)]],
 constant uint2 &rows [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
 if(tid==0){args[0]=(rows.x+63)/64;args[1]=tileCount[0];args[2]=1;
  args[4]=(rows.y+63)/64;args[5]=tileCount[0];args[6]=1;}
}
kernel void prefill_grouped_gate(device const uint *gw [[buffer(0)]],device const ushort *gs [[buffer(1)]],device const ushort *gb [[buffer(2)]],
 device const uint *uw [[buffer(3)]],device const ushort *us [[buffer(4)]],device const ushort *ub [[buffer(5)]],
 device const float *x [[buffer(6)]],device float *out [[buffer(7)]],constant PShape &s [[buffer(8)]],
 device const uint *routes [[buffer(9)]],device const uint *counts [[buffer(10)]],device const uint2 *tiles [[buffer(11)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint2 tile=tiles[group.y];uint valid=min(16u,counts[tile.x]-tile.y);
 threadgroup float gateWeights[64*32],upWeights[64*32],inputs[32*16],gateOut[64*16],upOut[64*16];threadgroup uint routeIds[16];
 if(tid<16)routeIds[tid]=tid<valid?routes[tile.x*s.tokens+tile.y+tid]:0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 simdgroup_float8x8 g0(0.0f),g1(0.0f),u0(0.0f),u1(0.0f);
 for(uint start=0;start<s.columns;start+=32){
  uint row=group.x*64+tid/4,col=start+(tid%4)*8,index=(tid/4)*32+(tid%4)*8;
  uint gl=0,gh=0,ul=0,uh=0;float gain=0,bias=0,ugain=0,ubias=0;
  if(row<s.rows){ulong r=ulong(tile.x)*s.rows+row,base=r*(s.columns*s.bits/32),p=r*(s.columns/64)+col/64;uint bit=col*s.bits,word=bit/32,shift=bit%32;
   gl=gw[base+word];gh=gw[base+word+1];ul=uw[base+word];uh=uw[base+word+1];
   gl=quant_align_low(gl,gh,shift);gh>>=shift;ul=quant_align_low(ul,uh,shift);uh>>=shift;
   gain=as_type<float>(uint(gs[p])<<16);bias=as_type<float>(uint(gb[p])<<16);ugain=as_type<float>(uint(us[p])<<16);ubias=as_type<float>(uint(ub[p])<<16);}
  #pragma clang loop unroll(full)
  for(uint j=0;j<8;j++){gateWeights[index+j]=float((s.bits==8?quant_octet<8>(gl,gh,j):quant_octet<5>(gl,gh,j)))*gain+bias;upWeights[index+j]=float((s.bits==8?quant_octet<8>(ul,uh,j):quant_octet<5>(ul,uh,j)))*ugain+ubias;}
  for(uint i=tid;i<32*16;i+=256){uint t=i%16;inputs[i]=t<valid?x[(routeIds[t]/8)*s.columns+start+i/16]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint inner=0;inner<4;inner++){simdgroup_float8x8 a,b0,b1;simdgroup_load(b0,inputs+inner*8*16,16);simdgroup_load(b1,inputs+inner*8*16+8,16);
   simdgroup_load(a,gateWeights+sg*8*32+inner*8,32);simdgroup_multiply_accumulate(g0,a,b0,g0);simdgroup_multiply_accumulate(g1,a,b1,g1);
   simdgroup_load(a,upWeights+sg*8*32+inner*8,32);simdgroup_multiply_accumulate(u0,a,b0,u0);simdgroup_multiply_accumulate(u1,a,b1,u1);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(g0,gateOut+sg*8*16,16);simdgroup_store(g1,gateOut+sg*8*16+8,16);
 simdgroup_store(u0,upOut+sg*8*16,16);simdgroup_store(u1,upOut+sg*8*16+8,16);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<64*16;i+=256){uint row=group.x*64+i/16,t=i%16;
  if(row<s.rows&&t<valid){float g=gateOut[i];out[routeIds[t]*s.rows+row]=g/(1+exp(-g))*upOut[i];}}
}
kernel void prefill_grouped_down(device const uint *w [[buffer(0)]],device const ushort *scales [[buffer(1)]],device const ushort *biases [[buffer(2)]],
 device const float *x [[buffer(3)]],device float *out [[buffer(4)]],constant PShape &s [[buffer(5)]],
 device const uint *routes [[buffer(6)]],device const uint *counts [[buffer(7)]],device const uint2 *tiles [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint2 tile=tiles[group.y];uint valid=min(16u,counts[tile.x]-tile.y);
 threadgroup float weights[64*32],inputs[32*16],result[64*16];threadgroup uint routeIds[16];
 if(tid<16)routeIds[tid]=tid<valid?routes[tile.x*s.tokens+tile.y+tid]:0;
 threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_float8x8 a0(0.0f),a1(0.0f);
 for(uint start=0;start<s.columns;start+=32){
  uint row=group.x*64+tid/4,col=start+(tid%4)*8,index=(tid/4)*32+(tid%4)*8;uint low=0,high=0;float scale=0,bias=0;
  if(row<s.rows){ulong r=ulong(tile.x)*s.rows+row,base=r*(s.columns*s.bits/32),p=r*(s.columns/64)+col/64;uint bit=col*s.bits,word=bit/32,shift=bit%32;
   low=w[base+word];high=w[base+word+1];low=quant_align_low(low,high,shift);high>>=shift;
   scale=as_type<float>(uint(scales[p])<<16);bias=as_type<float>(uint(biases[p])<<16);}
  #pragma clang loop unroll(full)
  for(uint j=0;j<8;j++)weights[index+j]=float((s.bits==8?quant_octet<8>(low,high,j):quant_octet<5>(low,high,j)))*scale+bias;
  for(uint i=tid;i<32*16;i+=256){uint t=i%16;inputs[i]=t<valid?x[routeIds[t]*s.columns+start+i/16]:0;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint inner=0;inner<4;inner++){simdgroup_float8x8 a,b;simdgroup_load(a,weights+sg*8*32+inner*8,32);
   simdgroup_load(b,inputs+inner*8*16,16);simdgroup_multiply_accumulate(a0,a,b,a0);
   simdgroup_load(b,inputs+inner*8*16+8,16);simdgroup_multiply_accumulate(a1,a,b,a1);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(a0,result+sg*8*16,16);simdgroup_store(a1,result+sg*8*16+8,16);threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<64*16;i+=256){uint row=group.x*64+i/16,t=i%16;if(row<s.rows&&t<valid)out[routeIds[t]*s.rows+row]=result[i];}
}
)metal";
