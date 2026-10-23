#!/usr/bin/env python3
"""Native CUDA layer-major chunk A/B, exact hidden and all recurrent/KV state checks.
Stop the deployed agent first; this script never stops another process.
Synthetic valid token IDs exercise varied routes, not formal prompt throughput.
"""
import argparse,array,math,ctypes as c,hashlib,json,pathlib,time,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--kv-bytes',type=int,choices=[2,4],default=4)
comparison=p.add_mutually_exclusive_group();comparison.add_argument('--compare-fast',action='store_true',help='Compare a baseline FP32 mode with candidate FP32 mode 6');comparison.add_argument('--compare-affine-broadcast',action='store_true',help='Compare packed FP32 mode 5 with experimental routed affine-broadcast mode 7')
p.add_argument('--baseline-mode',type=int,choices=[1,5],default=1,help='FP32 baseline for --compare-fast; 5 matches batch-fp32-packed deployment');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(result):
 if result:raise RuntimeError(err.value.decode())
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p)
destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('execution_mode',[c.c_void_p,c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int]);state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
kv_range=bind('kv_range',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6
h=create(a.weights.encode(),512,a.kv_bytes,err,len(err))
if not h:raise RuntimeError(err.value.decode())
tokens=[100+(i*137)%247000 for i in range(512)]
def finite(data,bf16=False):
 values=array.array('H' if bf16 else 'f');values.frombytes(data)
 assert all((v&0x7f80)!=0x7f80 for v in values) if bf16 else all(math.isfinite(v) for v in values), 'Nonfinite tensor'
records=[];outputs={};states={};traces={};ranges={};modes=(5,7) if a.compare_affine_broadcast else (a.baseline_mode,6) if a.compare_fast else (0,1)
def execute(n,chunks,trace_mode=None):
 hidden=bytearray();position=0;index=0
 while position<n:
  count=min(chunks[index%len(chunks)],n-position);ids=(c.c_int*count)(*tokens[position:position+count]);out=(c.c_float*(count*2048))()
  trace=(c.c_float*(count*20*2048))() if trace_mode is not None else None
  ok(run(h,ids,count,position,out,trace,err,len(err)));hidden.extend(bytes(out))
  if trace is not None:
   traces.setdefault(trace_mode,bytearray()).extend(bytes(trace))
   for layer in range(3,20,4):
    for kind in range(2):
     size=count*512*a.kv_bytes;data=c.create_string_buffer(size)
     ok(kv_range(h,layer,kind,position,count,data,size,err,len(err)))
     ranges.setdefault((trace_mode,layer,kind),bytearray()).extend(data.raw)
     assert kv_range(h,layer,kind,position+count,count,data,size,err,len(err))!=0, 'Future range accepted'
     assert kv_range(h,layer,kind,position,count,data,size-1,err,len(err))!=0, 'Wrong range size accepted'
  position+=count;index+=1
 return bytes(hidden)
try:
 # Tail sizes and multiple recurrent chunks; byte equality also covers BF16 KV storage.
 for selected_mode in modes:
  ok(reset(h,err,len(err)));ok(mode(h,selected_mode));outputs[selected_mode]=execute(131,[1,7,17,32],selected_mode)
  states[selected_mode]=[]
  for layer in range(20):
   for kind in range(2):
    size=131*512*a.kv_bytes if (layer+1)%4==0 else 32*128*128*4 if kind==0 else 3*8192*4
    data=c.create_string_buffer(size);ok(state(h,layer,kind,131,data,size,err,len(err)))
    if (layer+1)%4==0:assert bytes(ranges[(selected_mode,layer,kind)])==data.raw,'Incremental KV differs from final KV'
    finite(data.raw,(layer+1)%4==0 and a.kv_bytes==2);states[selected_mode].append(hashlib.sha256(data.raw).hexdigest());(a.output/f'state-{selected_mode}-{layer}-{kind}.bin').write_bytes(data.raw)
  finite(outputs[selected_mode]);finite(traces[selected_mode]);(a.output/f'hidden-{selected_mode}.bin').write_bytes(outputs[selected_mode]);(a.output/f'trace-{selected_mode}.bin').write_bytes(traces[selected_mode]);print('correctness mode',selected_mode,flush=True)
 assert traces[modes[0]]==traces[modes[1]], 'Layer trace differs'
 assert outputs[modes[0]]==outputs[modes[1]], 'Hidden differs'
 assert states[modes[0]]==states[modes[1]], 'Recurrent/KV state differs'
 # Alternate modes; one warmup then three measured passes per mode.
 for trial in range(4):
  for selected_mode in (modes if trial%2==0 else modes[::-1]):
   ok(reset(h,err,len(err)));ok(mode(h,selected_mode));start=time.perf_counter();output=execute(512,[32]);elapsed=time.perf_counter()-start;finite(output)
   records.append({'trial':trial,'warmup':trial==0,'mode':selected_mode,'seconds':elapsed,'tokensPerSecond':512/elapsed,'hiddenSha256':hashlib.sha256(output).hexdigest()});print(json.dumps(records[-1]),flush=True)
 assert len({r['hiddenSha256'] for r in records})==1
 result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dllSha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'kvBytes':a.kv_bytes,'weightBits':8,'tokensSha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'correctnessTokens':131,'modes':list(modes),'allFinite':True,'allLayerTraceEqual':True,'hiddenEqual':True,'all40StateTensorsEqual':True,'incrementalKvEqualsFinal':True,'kvRangeBoundsRejected':True,'records':records,'medianSeconds':{str(selected_mode):statistics.median(r['seconds'] for r in records if not r['warmup'] and r['mode']==selected_mode) for selected_mode in modes},'kind':'layer-major-fp32-prefix-only-synthetic-token-test-not-e2e-acceptance'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('PASS exact hidden/state and deterministic throughput',flush=True)
finally:destroy(h)
