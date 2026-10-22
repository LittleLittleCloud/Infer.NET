// Original online-softmax attention with BF16 MPP operands and FP32 statistics.
// Four tokens x eight GQA heads share each 32-key tile; no full score matrix.
static NSString *prefill_attention_bf16_source = @R"metal(
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
#ifndef BF16_ATTENTION_PARTITION
#define BF16_ATTENTION_PARTITION 1024
#endif
#ifndef BF16_ATTENTION_SG
#define BF16_ATTENTION_SG 4
#endif
kernel void prefill_pack_queries(device const float*q [[buffer(0)]],device bfloat*out [[buffer(1)]],constant uint &tokens [[buffer(2)]],uint i [[thread_position_in_grid]]){
 if(i>=tokens*4096)return;uint d=i%256,h=(i/256)%16,t=i/4096;out[((h/8)*tokens*8+t*8+h%8)*256+d]=bfloat(q[i]);
}
kernel void prefill_attention_bf16(device bfloat*q [[buffer(0)]],device bfloat*keys [[buffer(1)]],
 device bfloat*values [[buffer(2)]],device float*parts [[buffer(3)]],constant AttentionShape&s [[buffer(4)]],
 uint3 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 constexpr uint Rows=BF16_QUERY_TOKENS*8,Keys=BF16_KEY_TOKENS;
 uint part=group.x,kh=group.y,qbase=group.z*BF16_QUERY_TOKENS,start=part*BF16_ATTENTION_PARTITION,end=min(start+BF16_ATTENTION_PARTITION,s.position+s.tokens);
 threadgroup bfloat probabilities[Rows*Keys];threadgroup float scores[Rows*Keys],maxima[Rows],sums[Rows],alpha[Rows];
 auto Q=tensor(q+(kh*s.tokens*8+qbase*8)*256,dextents<int,2>{256,int(min(Rows,(s.tokens-qbase)*8))},array<int,2>{1,256});
 auto P=tensor(probabilities,extents<int,Keys,Rows>{},array<int,2>{1,Keys});
 constexpr auto qkd=matmul2d_descriptor(Rows,Keys,256,false,true);
 constexpr auto pvd=matmul2d_descriptor(Rows,256,Keys,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<qkd,execution_simdgroups<BF16_ATTENTION_SG>> qk;matmul2d<pvd,execution_simdgroups<BF16_ATTENTION_SG>> pv;
 auto initialV=tensor(values,dextents<int,2>{256,int(Keys)},array<int,2>{1,512});
 auto accumulated=pv.get_destination_cooperative_tensor<decltype(P),decltype(initialV),float>();
 #pragma clang loop unroll(full)
 for(uint i=0;i<accumulated.get_capacity();i++)accumulated[i]=0;
 if(tid<Rows){maxima[tid]=-INFINITY;sums[tid]=0;}threadgroup_barrier(mem_flags::mem_threadgroup);
 if(start<=s.position+min(qbase+BF16_QUERY_TOKENS,s.tokens)-1)for(uint block=start;block<end;block+=Keys){
  uint validKeys=min(Keys,end-block);
  auto K=tensor(keys+block*512+kh*256,dextents<int,2>{256,int(validKeys)},array<int,2>{1,512});
  auto dots=qk.get_destination_cooperative_tensor<decltype(Q),decltype(K),float>();qk.run(Q,K,dots);
  auto S=tensor(scores,extents<int,Keys,Rows>{},array<int,2>{1,Keys});dots.store(S);threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint row=sg;row<Rows;row+=BF16_ATTENTION_SG){uint t=qbase+row/8;float laneValues[Keys/32],best=-INFINITY;
   #pragma clang loop unroll(full)
   for(uint j=0;j<Keys/32;j++){uint key=block+lane+j*32;float value=t<s.tokens&&key<end&&key<=s.position+t?scores[row*Keys+lane+j*32]/16.0f:-INFINITY;laneValues[j]=value;best=max(best,value);}
   float next=max(maxima[row],simd_max(best)),scale=maxima[row]==-INFINITY?0:exp(maxima[row]-next),total=0;
   #pragma clang loop unroll(full)
   for(uint j=0;j<Keys/32;j++){float value=laneValues[j],p=value==-INFINITY?0:exp(value-next);total+=p;probabilities[row*Keys+lane+j*32]=bfloat(p);}
   total=simd_sum(total);if(lane==0){maxima[row]=next;sums[row]=sums[row]*scale+total;alpha[row]=scale;}
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
 for(uint i=0;i<accumulated.get_capacity();i++)if(accumulated.is_valid_element(i)){auto at=accumulated.get_multidimensional_index(i);accumulated[i]*=alpha[at[1]];}
  auto V=tensor(values+block*512+kh*256,dextents<int,2>{256,int(validKeys)},array<int,2>{1,512});
  pv.run(P,V,accumulated);threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 #pragma clang loop unroll(full)
 for(uint i=0;i<accumulated.get_capacity();i++)if(accumulated.is_valid_element(i)){auto at=accumulated.get_multidimensional_index(i);uint t=qbase+at[1]/8,h=kh*8+at[1]%8;
  if(t<s.tokens)parts[((t*16+h)*s.partitions+part)*258+2+at[0]]=accumulated[i];}
 if(tid<Rows){uint t=qbase+tid/8,h=kh*8+tid%8;if(t<s.tokens){uint at=((t*16+h)*s.partitions+part)*258;parts[at]=maxima[tid];parts[at+1]=sums[tid];}}
}
)metal";
