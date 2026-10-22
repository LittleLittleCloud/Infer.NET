// Original temporary affine expansion + device-tensor MPP GEMM. A bounded
// reusable workspace expands one layer at a time; stored weights stay packed.
static NSString *prefill_expanded_source = @R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
struct EShape {uint rows,columns,tokens,experts,grouped,bits,maxTiles;};
kernel void expanded_weights(device const uint*w [[buffer(0)]],device const ushort*sc [[buffer(1)]],device const ushort*bi [[buffer(2)]],
 device bfloat*out [[buffer(3)]],constant EShape&s [[buffer(4)]],device const uint*counts [[buffer(5)]],uint i [[thread_position_in_grid]]){
 uint groups=s.columns/8,r=i/groups,col=(i%groups)*8;if(r>=s.rows*s.experts)return;
 if(s.grouped&&counts[r/s.rows]==0)return;uint p=r*(s.columns/64)+col/64,bit=col*s.bits,word=bit/32,shift=bit%32;
 ulong base=ulong(r)*(s.columns*s.bits/32);uint low=w[base+word],high=w[base+word+1];low=quant_align_low(low,high,shift);high>>=shift;
 float scale=as_type<float>(uint(sc[p])<<16),bias=as_type<float>(uint(bi[p])<<16);
 #pragma clang loop unroll(full)
 for(uint j=0;j<8;j++)out[ulong(r)*s.columns+col+j]=bfloat(float(s.bits==5?quant_octet<5>(low,high,j):quant_octet<8>(low,high,j))*scale+bias);
}
kernel void expanded_pack(device const float*x [[buffer(0)]],device bfloat*out [[buffer(1)]],constant EShape&s [[buffer(2)]],
 device const uint*routes [[buffer(3)]],device const uint*counts [[buffer(4)]],device const uint2*tiles [[buffer(5)]],device const uint*tileCount [[buffer(6)]],
 uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]){
 uint tile=group.y;if(tile>=(s.grouped?tileCount[0]:(s.tokens+63)/64))return;
 uint expert=0,first=tile*64,valid=min(64u,s.tokens-first);if(s.grouped){uint2 entry=tiles[tile];expert=entry.x;first=entry.y;valid=min(64u,counts[expert]-first);}
 uint at=group.x*256+tid;if(at>=s.columns*64)return;uint token=at/s.columns,col=at%s.columns;
 uint route=token<valid?(s.grouped?routes[expert*s.tokens+first+token]:first+token):0;
 out[ulong(tile)*s.columns*64+at]=bfloat(token<valid?x[(s.grouped==1?route/8:route)*s.columns+col]:0);
}
template<bool Gate>
inline void expanded_product(device bfloat*w,device bfloat*uw,device bfloat*x,device float*y,device const float*gate,
 constant EShape&s,device const uint*routes,device const uint*counts,device const uint2*tiles,uint2 group){
 uint tile=group.y,expert=0,first=tile*64,valid=min(64u,s.tokens-first);
 if(s.grouped){uint2 entry=tiles[tile];expert=entry.x;first=entry.y;valid=min(64u,counts[expert]-first);}
 uint rowBase=group.x*64,validRows=min(64u,s.rows-rowBase);ulong offset=(ulong(expert)*s.rows+rowBase)*s.columns;
 auto A=tensor(w+offset,dextents<int,2>{int(s.columns),int(validRows)},array<int,2>{1,int(s.columns)});
 auto B=tensor(x+ulong(tile)*64*s.columns,dextents<int,2>{int(s.columns),int(valid)},array<int,2>{1,int(s.columns)});
 constexpr auto desc=matmul2d_descriptor(64,64,dynamic_length_v<int>,false,true);
 matmul2d<desc,execution_simdgroups<4>> op;auto result=op.template get_destination_cooperative_tensor<decltype(A),decltype(B),float>();op.run(A,B,result);
 auto up=op.template get_destination_cooperative_tensor<decltype(A),decltype(B),float>();
 if(Gate){auto U=tensor(uw+offset,dextents<int,2>{int(s.columns),int(validRows)},array<int,2>{1,int(s.columns)});op.run(U,B,up);}
 #pragma clang loop unroll(full)
 for(uint i=0;i<result.get_capacity();i++)if(result.is_valid_element(i)){auto at=result.get_multidimensional_index(i);uint t=at[0],r=rowBase+at[1];
  if(t<valid&&r<s.rows){uint route=s.grouped?routes[expert*s.tokens+first+t]:first+t;float value=result[i];y[route*s.rows+r]=Gate?value/(1+exp(-value))*up[i]:value;}}
}
kernel void expanded_matrix(device bfloat*w [[buffer(0)]],device bfloat*x [[buffer(1)]],device float*y [[buffer(2)]],constant EShape&s [[buffer(3)]],
 device const uint*routes [[buffer(4)]],device const uint*counts [[buffer(5)]],device const uint2*tiles [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]]){
 expanded_product<false>(w,w,x,y,y,s,routes,counts,tiles,group);
}
kernel void expanded_gate(device bfloat*w [[buffer(0)]],device bfloat*uw [[buffer(1)]],device bfloat*x [[buffer(2)]],device float*y [[buffer(3)]],constant EShape&s [[buffer(4)]],
 device const uint*routes [[buffer(5)]],device const uint*counts [[buffer(6)]],device const uint2*tiles [[buffer(7)]],uint2 group [[threadgroup_position_in_grid]]){
 expanded_product<true>(w,uw,x,y,y,s,routes,counts,tiles,group);
}
)metal";
