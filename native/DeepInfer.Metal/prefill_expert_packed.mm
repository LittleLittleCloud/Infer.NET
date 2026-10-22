// Original packed-5/8-bit expert GEMM: compact BF16 input once, retain bounded
// 32x64 dequantized weights on chip, consume device input through MPP.
// Gate/up uses two SIMD groups; down uses one to avoid idle matrix lanes.
static NSString *prefill_expert_packed_source = @R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef PACKED_EXPERT_ROWS
#define PACKED_EXPERT_ROWS 32
#endif
#ifndef PACKED_EXPERT_K
#define PACKED_EXPERT_K 64
#endif
#ifndef PACKED_EXPERT_SG
#define PACKED_EXPERT_SG (Gate ? 2 : 1)
#endif
#ifndef PACKED_EXPERT_ADAPTIVE
#define PACKED_EXPERT_ADAPTIVE 0
#endif
#ifndef PACKED_EXPERT_PIPELINE
#define PACKED_EXPERT_PIPELINE 0
#endif
#define PACKED_EXPERT_BUFFERS (PACKED_EXPERT_PIPELINE ? 2 : 1)
template<bool Gate>
inline void packed_expert_load(device const uint*w,device const ushort*sc,device const ushort*bi,
 device const uint*uw,device const ushort*us,device const ushort*ub,constant PShape&s,
 uint expert,uint2 group,uint tid,uint start,threadgroup bfloat*weights,threadgroup bfloat*upWeights){
 constexpr uint Rows=PACKED_EXPERT_ROWS,Inner=PACKED_EXPERT_K,Threads=PACKED_EXPERT_SG*32;
  for(uint i=tid;i<Rows*(Inner/8);i+=Threads){uint row=group.x*Rows+i/(Inner/8),col=start+(i%(Inner/8))*8,index=(i/(Inner/8))*Inner+(i%(Inner/8))*8;uint low=0,high=0,ul=0,uh=0;float scale=0,bias=0,ugain=0,ubias=0;
   if(row<s.rows&&col<s.columns){ulong r=ulong(expert)*s.rows+row,base=r*(s.columns*s.bits/32),p=r*(s.columns/64)+col/64;uint bit=col*s.bits,word=bit/32,shift=bit%32;
    low=w[base+word];high=w[base+word+1];low=quant_align_low(low,high,shift);high>>=shift;scale=as_type<float>(uint(sc[p])<<16);bias=as_type<float>(uint(bi[p])<<16);
    if(Gate){ul=uw[base+word];uh=uw[base+word+1];ul=quant_align_low(ul,uh,shift);uh>>=shift;ugain=as_type<float>(uint(us[p])<<16);ubias=as_type<float>(uint(ub[p])<<16);}}
   #pragma clang loop unroll(full)
   for(uint j=0;j<8;j++){weights[index+j]=bfloat(float((s.bits==8?quant_octet<8>(low,high,j):quant_octet<5>(low,high,j)))*scale+bias);if(Gate)upWeights[index+j]=bfloat(float((s.bits==8?quant_octet<8>(ul,uh,j):quant_octet<5>(ul,uh,j)))*ugain+ubias);}
  }
}
template<bool Gate,bool Direct=false,uint Tokens=64>
inline void packed_expert(device const uint*w,device const ushort*sc,device const ushort*bi,
 device const uint*uw,device const ushort*us,device const ushort*ub,device bfloat*x,device float*y,
 constant PShape&s,device const uint*routes,device const uint*counts,device const uint2*tiles,
 uint2 group,uint tid,threadgroup bfloat*weights,threadgroup bfloat*upWeights){
 constexpr uint Rows=PACKED_EXPERT_ROWS,Inner=PACKED_EXPERT_K,Threads=PACKED_EXPERT_SG*32;
 uint2 entry=tiles[group.y];uint valid=min(64u,counts[entry.x]-entry.y);
 auto A=tensor(weights,extents<int,Inner,Rows>{},array<int,2>{1,Inner});auto U=tensor(upWeights,extents<int,Inner,Rows>{},array<int,2>{1,Inner});
 auto initialB=tensor(x,dextents<int,2>{int(Inner),int(valid)},array<int,2>{1,int(s.columns)});
 constexpr auto d=matmul2d_descriptor(Rows,Tokens,Inner,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<d,execution_simdgroups<PACKED_EXPERT_SG>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(A),decltype(initialB),float>();auto up=op.template get_destination_cooperative_tensor<decltype(A),decltype(initialB),float>();
 #pragma clang loop unroll(full)
 for(uint i=0;i<acc.get_capacity();i++){acc[i]=0;if(Gate)up[i]=0;}
#if PACKED_EXPERT_PIPELINE
 packed_expert_load<Gate>(w,sc,bi,uw,us,ub,s,entry.x,group,tid,0,weights,upWeights);
 threadgroup_barrier(mem_flags::mem_threadgroup);
#endif
 for(uint start=0;start<s.columns;start+=Inner){
#if !PACKED_EXPERT_PIPELINE
  packed_expert_load<Gate>(w,sc,bi,uw,us,ub,s,entry.x,group,tid,start,weights,upWeights);
  threadgroup_barrier(mem_flags::mem_threadgroup);
#else
  uint current=(start/Inner)&1,next=current^1;
  if(start+Inner<s.columns)packed_expert_load<Gate>(w,sc,bi,uw,us,ub,s,entry.x,group,tid,start+Inner,weights+next*Rows*Inner,Gate?upWeights+next*Rows*Inner:upWeights);
  auto A=tensor(weights+current*Rows*Inner,extents<int,Inner,Rows>{},array<int,2>{1,Inner});
  auto U=tensor(Gate?upWeights+current*Rows*Inner:upWeights,extents<int,Inner,Rows>{},array<int,2>{1,Inner});
#endif
  auto B=tensor(x+ulong(group.y)*64*s.columns+start,dextents<int,2>{int(min(Inner,s.columns-start)),int(valid)},array<int,2>{1,int(s.columns)});
  op.run(A,B,acc);if(Gate)op.run(U,B,up);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){auto at=acc.get_multidimensional_index(i);uint t=at[0],row=group.x*Rows+at[1];
  if(t<valid&&row<s.rows){float value=acc[i];value=Gate?value/(1+exp(-value))*up[i]:value;
   if(Direct)reinterpret_cast<device bfloat*>(y)[(ulong(group.y)*64+t)*s.rows+row]=bfloat(value);
   else {uint route=routes[entry.x*s.tokens+entry.y+t];y[route*s.rows+row]=value;}}}
}
template<bool Gate,bool Direct=false>
inline void packed_expert_dispatch(device const uint*w,device const ushort*sc,device const ushort*bi,
 device const uint*uw,device const ushort*us,device const ushort*ub,device bfloat*x,device float*y,
 constant PShape&s,device const uint*routes,device const uint*counts,device const uint2*tiles,
 uint2 group,uint tid,threadgroup bfloat*weights,threadgroup bfloat*upWeights){
#if PACKED_EXPERT_ADAPTIVE
 uint2 entry=tiles[group.y];uint valid=min(64u,counts[entry.x]-entry.y);
 if(valid<=16)packed_expert<Gate,Direct,16>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,weights,upWeights);
 else if(valid<=32)packed_expert<Gate,Direct,32>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,weights,upWeights);
 else
#endif
 packed_expert<Gate,Direct,64>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,weights,upWeights);
}
// Gate and up share one 64-row matrix product. Once all K tiles have
// completed, reuse their 8 KiB weight scratch for the up accumulator only.
// Gate values remain in registers; no device intermediate or extra rounding.
template<bool Direct,uint Tokens=64>
inline void packed_expert_fused_gate(device const uint*w,device const ushort*sc,device const ushort*bi,
 device const uint*uw,device const ushort*us,device const ushort*ub,device bfloat*x,device float*y,
 constant PShape&s,device const uint*routes,device const uint*counts,device const uint2*tiles,
 uint2 group,uint tid,threadgroup bfloat*weights){
 constexpr uint Rows=32,Inner=64;
 uint2 entry=tiles[group.y];uint valid=min(64u,counts[entry.x]-entry.y);
 auto A=tensor(weights,extents<int,Inner,Rows*2>{},array<int,2>{1,Inner});
 auto initialB=tensor(x,dextents<int,2>{int(Inner),int(valid)},array<int,2>{1,int(s.columns)});
 constexpr auto d=matmul2d_descriptor(Rows*2,Tokens,Inner,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<d,execution_simdgroups<2>> op;auto acc=op.template get_destination_cooperative_tensor<decltype(A),decltype(initialB),float>();
 #pragma clang loop unroll(full)
 for(uint i=0;i<acc.get_capacity();i++)acc[i]=0;
 for(uint start=0;start<s.columns;start+=Inner){
  packed_expert_load<true>(w,sc,bi,uw,us,ub,s,entry.x,group,tid,start,weights,weights+Rows*Inner);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto B=tensor(x+ulong(group.y)*64*s.columns+start,dextents<int,2>{int(min(Inner,s.columns-start)),int(valid)},array<int,2>{1,int(s.columns)});
  op.run(A,B,acc);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 auto scratch=reinterpret_cast<threadgroup float*>(weights);
 #pragma clang loop unroll(full)
 for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
  auto at=acc.get_multidimensional_index(i);if(at[1]>=Rows)scratch[(at[1]-Rows)*64+at[0]]=acc[i];
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 #pragma clang loop unroll(full)
 for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
  auto at=acc.get_multidimensional_index(i);uint t=at[0],r=at[1],row=group.x*Rows+r;
  if(r<Rows&&t<valid&&row<s.rows){float gate=acc[i],value=gate/(1+exp(-gate))*scratch[r*64+t];
   if(Direct)reinterpret_cast<device bfloat*>(y)[(ulong(group.y)*64+t)*s.rows+row]=bfloat(value);
   else {uint route=routes[entry.x*s.tokens+entry.y+t];y[route*s.rows+row]=value;}
  }
 }
}
#define PACKED_FUSED_GATE(NAME,DIRECT) \
kernel void NAME(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]], \
 device const uint*uw [[buffer(3)]],device const ushort*us [[buffer(4)]],device const ushort*ub [[buffer(5)]],device bfloat*x [[buffer(6)]],device float*y [[buffer(7)]],constant PShape&s [[buffer(8)]], \
 device const uint*routes [[buffer(9)]],device const uint*counts [[buffer(10)]],device const uint2*tiles [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){ \
 threadgroup bfloat a[2*32*64];uint2 entry=tiles[group.y];uint valid=min(64u,counts[entry.x]-entry.y); \
 if(PACKED_EXPERT_ADAPTIVE&&valid<=16)packed_expert_fused_gate<DIRECT,16>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a); \
 else if(PACKED_EXPERT_ADAPTIVE&&valid<=32)packed_expert_fused_gate<DIRECT,32>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a); \
 else packed_expert_fused_gate<DIRECT,64>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a); \
}
PACKED_FUSED_GATE(packed_expert_gate_fused,false)
PACKED_FUSED_GATE(packed_expert_gate_fused_direct,true)
kernel void packed_expert_gate(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],
 device const uint*uw [[buffer(3)]],device const ushort*us [[buffer(4)]],device const ushort*ub [[buffer(5)]],device bfloat*x [[buffer(6)]],device float*y [[buffer(7)]],constant PShape&s [[buffer(8)]],
 device const uint*routes [[buffer(9)]],device const uint*counts [[buffer(10)]],device const uint2*tiles [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 threadgroup bfloat a[PACKED_EXPERT_BUFFERS*PACKED_EXPERT_ROWS*PACKED_EXPERT_K],u[PACKED_EXPERT_BUFFERS*PACKED_EXPERT_ROWS*PACKED_EXPERT_K];packed_expert_dispatch<true>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a,u);
}
kernel void packed_expert_down(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],device bfloat*x [[buffer(3)]],device float*y [[buffer(4)]],constant PShape&s [[buffer(5)]],
 device const uint*routes [[buffer(6)]],device const uint*counts [[buffer(7)]],device const uint2*tiles [[buffer(8)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 threadgroup bfloat a[PACKED_EXPERT_BUFFERS*PACKED_EXPERT_ROWS*PACKED_EXPERT_K],u[1];packed_expert_dispatch<false>(w,sc,bi,w,sc,bi,x,y,s,routes,counts,tiles,group,tid,a,u);
}
kernel void packed_expert_gate_direct(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],
 device const uint*uw [[buffer(3)]],device const ushort*us [[buffer(4)]],device const ushort*ub [[buffer(5)]],device bfloat*x [[buffer(6)]],device float*y [[buffer(7)]],constant PShape&s [[buffer(8)]],
 device const uint*routes [[buffer(9)]],device const uint*counts [[buffer(10)]],device const uint2*tiles [[buffer(11)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 threadgroup bfloat a[PACKED_EXPERT_BUFFERS*PACKED_EXPERT_ROWS*PACKED_EXPERT_K],u[PACKED_EXPERT_BUFFERS*PACKED_EXPERT_ROWS*PACKED_EXPERT_K];packed_expert_dispatch<true,true>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a,u);
}
)metal";
