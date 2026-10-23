#!/usr/bin/env python3
import ctypes,json,pathlib
root=pathlib.Path(__file__).resolve().parents[2]
dll=ctypes.CDLL(str(root/'native/DeepInfer.Cuda/build/deepinfer_cuda.dll'))
f=dll.di_cuda_prefix_selftest;f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_char_p,ctypes.c_int];f.restype=ctypes.c_int
errors=(ctypes.c_double*2)();message=ctypes.create_string_buffer(2048)
assert f(errors,message,len(message))==0,message.value.decode()
result=dict(gemvMaxAbs=errors[0],deltaOutputAndStateMaxAbs=errors[1],passed=errors[0]<1e-5 and errors[1]<1e-6)
print(json.dumps(result,indent=2));assert result['passed']
