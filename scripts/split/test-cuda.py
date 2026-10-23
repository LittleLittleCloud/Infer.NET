#!/usr/bin/env python3
import ctypes,json,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
dll=ctypes.CDLL(str(root/'native/DeepInfer.Cuda/build/deepinfer_cuda.dll'))
f=dll.di_cuda_prefix_selftest;f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_char_p,ctypes.c_int];f.restype=ctypes.c_int
errors=(ctypes.c_double*2)();message=ctypes.create_string_buffer(2048)
assert f(errors,message,len(message))==0,message.value.decode()
result=dict(gemvMaxAbs=errors[0],deltaOutputAndStateMaxAbs=errors[1],passed=errors[0]<1e-5 and errors[1]<1e-6)
print(json.dumps(result,indent=2));assert result['passed']
attention=dll.di_cuda_prefix_attention_test
attention.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_double),ctypes.c_char_p,ctypes.c_int];attention.restype=ctypes.c_int
for length in (17,2048,4097,8193):
 for kv in (2,4):
  maximum=ctypes.c_double()
  assert attention(length,kv,ctypes.byref(maximum),message,len(message))==0,message.value.decode()
  print(json.dumps(dict(attentionTokens=length,kvBytes=kv,maxAbs=maximum.value,passed=maximum.value<1e-5)))
  assert maximum.value<1e-5
