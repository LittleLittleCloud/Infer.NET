// Original packed affine GEMM with BF16 operands and FP32 cooperative accumulators.
// Buffer boundaries remain FP32. No model runtime or external GEMM implementation.
static NSString *prefill_bf16_source = @R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template<uint Bits,uint N,bool Grouped,bool Gate>
inline void prefill_bf16_product(device const uint *w,device const ushort *scales,device const ushort *biases,
 device const uint *uw,device const ushort *us,device const ushort *ub,device const float *x,device float *out,
 constant PShape &s,device const uint *routes,device const uint *counts,device const uint2 *tiles,
 uint2 group,uint tid,threadgroup bfloat *weights,threadgroup bfloat *upWeights,threadgroup bfloat *inputs,
 threadgroup uint *routeIds) {
 uint expert=0,first=group.y*N,valid=min(N,s.tokens-first);
 if(Grouped){uint2 tile=tiles[group.y];expert=tile.x;first=tile.y;valid=min(N,counts[expert]-first);}
 if(tid<N)routeIds[tid]=tid<valid?(Grouped?routes[expert*s.tokens+first+tid]:first+tid):0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 auto A=tensor(weights,extents<int,64,64>{},array<int,2>{1,64});
 auto U=tensor(upWeights,extents<int,64,64>{},array<int,2>{1,64});
 auto B=tensor(inputs,extents<int,N,64>{},array<int,2>{1,N});
 constexpr auto desc=matmul2d_descriptor(64,N,64,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto acc=op.template get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
 auto up=op.template get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
 for(uint i=0;i<acc.get_capacity();i++){acc[i]=0;if(Gate)up[i]=0;}
 for(uint start=0;start<s.columns;start+=64){
  for(uint i=tid;i<64*8;i+=128){uint row=group.x*64+i/8,col=start+(i%8)*8,index=(i/8)*64+(i%8)*8;
   uint low=0,high=0,ul=0,uh=0;float scale=0,bias=0,ugain=0,ubias=0;
   if(row<s.rows){ulong r=ulong(expert)*s.rows+row,base=r*(s.columns*Bits/32),p=r*(s.columns/64)+col/64;uint bit=col*Bits,word=bit/32,shift=bit%32;
    low=w[base+word];high=w[base+word+1];low=quant_align_low(low,high,shift);high>>=shift;
    scale=as_type<float>(uint(scales[p])<<16);bias=as_type<float>(uint(biases[p])<<16);
    if(Gate){ul=uw[base+word];uh=uw[base+word+1];ul=quant_align_low(ul,uh,shift);uh>>=shift;ugain=as_type<float>(uint(us[p])<<16);ubias=as_type<float>(uint(ub[p])<<16);}}
   #pragma clang loop unroll(full)
   for(uint j=0;j<8;j++){weights[index+j]=bfloat(float(quant_octet<Bits>(low,high,j))*scale+bias);
    if(Gate)upWeights[index+j]=bfloat(float(quant_octet<Bits>(ul,uh,j))*ugain+ubias);}
  }
  for(uint i=tid;i<64*N;i+=128){uint t=i%N,route=routeIds[t];
   inputs[i]=bfloat(t<valid?x[(Grouped&&Gate?route/8:route)*s.columns+start+i/N]:0);}
  threadgroup_barrier(mem_flags::mem_threadgroup);op.run(A,B,acc);if(Gate)op.run(U,B,up);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint i=0;i<acc.get_capacity();i++)if(acc.is_valid_element(i)){
  auto at=acc.get_multidimensional_index(i);uint t=at[0],row=group.x*64+at[1];
  if(row<s.rows&&t<valid){float value=acc[i];out[routeIds[t]*s.rows+row]=Gate?value/(1+exp(-value))*up[i]:value;}
 }

}
#define BF16_DENSE(NAME,BITS) \
kernel void NAME(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]], \
 device const float*x [[buffer(3)]],device float*y [[buffer(4)]],constant PShape&s [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){ \
 threadgroup bfloat a[4096],b[4096],u[1];threadgroup uint ids[64]; \
 prefill_bf16_product<BITS,64,false,false>(w,sc,bi,w,sc,bi,x,y,s,(device const uint*)nullptr,(device const uint*)nullptr,(device const uint2*)nullptr,group,tid,a,u,b,ids);}
BF16_DENSE(prefill_bf16_matrix5,5)
BF16_DENSE(prefill_bf16_matrix8,8)
kernel void prefill_bf16_gate(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],
 device const uint*uw [[buffer(3)]],device const ushort*us [[buffer(4)]],device const ushort*ub [[buffer(5)]],
 device const float*x [[buffer(6)]],device float*y [[buffer(7)]],constant PShape&s [[buffer(8)]],
 device const uint*routes [[buffer(9)]],device const uint*counts [[buffer(10)]],device const uint2*tiles [[buffer(11)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 threadgroup bfloat a[4096],u[4096],b[4096];threadgroup uint ids[64];
 if(s.bits==8)prefill_bf16_product<8,64,true,true>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a,u,b,ids);
 else prefill_bf16_product<5,64,true,true>(w,sc,bi,uw,us,ub,x,y,s,routes,counts,tiles,group,tid,a,u,b,ids);
}
kernel void prefill_bf16_down(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],
 device const float*x [[buffer(3)]],device float*y [[buffer(4)]],constant PShape&s [[buffer(5)]],
 device const uint*routes [[buffer(6)]],device const uint*counts [[buffer(7)]],device const uint2*tiles [[buffer(8)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 threadgroup bfloat a[4096],u[1],b[4096];threadgroup uint ids[64];
 if(s.bits==8)prefill_bf16_product<8,64,true,false>(w,sc,bi,w,sc,bi,x,y,s,routes,counts,tiles,group,tid,a,u,b,ids);
 else prefill_bf16_product<5,64,true,false>(w,sc,bi,w,sc,bi,x,y,s,routes,counts,tiles,group,tid,a,u,b,ids);
}
)metal";
