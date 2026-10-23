#!/usr/bin/env python3
"""Native CUDA readback A/B with one resident model, exact hidden/state checks.
Stop the deployed agent first; this script never stops another process.
Synthetic valid token IDs exercise varied routes, not formal prompt throughput.
"""
import argparse,ctypes as c,hashlib,json,pathlib,time,statistics,subprocess
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--kv-bytes',type=int,choices=[2,4],default=4);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(result):
 if result:raise RuntimeError(err.value.decode())
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p)
destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('readback_mode',[c.c_void_p,c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int]);state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6
h=create(a.weights.encode(),512,a.kv_bytes,err,len(err))
if not h:raise RuntimeError(err.value.decode())
tokens=[100+(i*137)%247000 for i in range(512)]
records=[];outputs={};states={}
def execute(n,chunks):
 hidden=bytearray();position=0;index=0
 while position<n:
  count=min(chunks[index%len(chunks)],n-position);ids=(c.c_int*count)(*tokens[position:position+count]);out=(c.c_float*(count*2048))()
  ok(run(h,ids,count,position,out,None,err,len(err)));hidden.extend(bytes(out));position+=count;index+=1
 return bytes(hidden)
try:
 # Tail sizes and multiple recurrent chunks; byte equality also covers BF16 KV storage.
 for batched in (0,1):
  ok(mode(h,batched));ok(reset(h,err,len(err)));outputs[batched]=execute(131,[1,7,17,32])
  states[batched]=[]
  for layer in range(20):
   for kind in range(2):
    size=131*512*a.kv_bytes if (layer+1)%4==0 else 32*128*128*4 if kind==0 else 3*8192*4
    data=c.create_string_buffer(size);ok(state(h,layer,kind,131,data,size,err,len(err)));states[batched].append(hashlib.sha256(data.raw).hexdigest())
  print('correctness mode',batched,flush=True)
 assert outputs[0]==outputs[1], 'Hidden differs'
 assert states[0]==states[1], 'Recurrent/KV state differs'
 # Alternate modes; one warmup then three measured passes per mode.
 for trial in range(4):
  for batched in ((0,1) if trial%2==0 else (1,0)):
   ok(mode(h,batched));ok(reset(h,err,len(err)));start=time.perf_counter();output=execute(512,[32]);elapsed=time.perf_counter()-start
   records.append({'trial':trial,'warmup':trial==0,'batched':bool(batched),'seconds':elapsed,'tokensPerSecond':512/elapsed,'hiddenSha256':hashlib.sha256(output).hexdigest()});print(json.dumps(records[-1]),flush=True)
 assert len({r['hiddenSha256'] for r in records})==1
 result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dllSha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'kvBytes':a.kv_bytes,'weightBits':8,'tokensSha256':hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest(),'correctnessTokens':131,'hiddenEqual':True,'all40StateTensorsEqual':True,'records':records,'medianSeconds':{str(bool(b)):statistics.median(r['seconds'] for r in records if not r['warmup'] and r['batched']==bool(b)) for b in (0,1)},'kind':'native-prefix-only-synthetic-token-test-not-e2e-acceptance'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('PASS exact hidden/state and deterministic throughput',flush=True)
finally:destroy(h)
