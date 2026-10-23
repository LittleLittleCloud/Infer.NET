#!/usr/bin/env python3
"""Native FP32 prefix A/B with fixed token IDs and unchanged tensor gates.
One process/cohort to avoid retaining the other path's scratch allocation. Stop on
numerical failure. Hidden/state snapshots are retained; these are not E2E timings.
"""
import argparse,ctypes as c,hashlib,json,pathlib,subprocess,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--tokens-file',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--reference',type=pathlib.Path);p.add_argument('--partition-threshold',type=int,default=-1);a=p.parse_args()
assert -1<=a.partition_threshold<=262144
assert (a.reference is None)==(a.partition_threshold==-1),'Candidate requires a retained legacy reference'
a.output.mkdir(parents=True,exist_ok=False);tokens=json.loads(a.tokens_file.read_text());assert 0<len(tokens)<=131072 and all(type(x) is int and 0<=x<248320 for x in tokens)
sha=hashlib.sha256()
with open(a.weights,'rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):sha.update(b)
assert sha.hexdigest()=='ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'
dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(v):
 if v:raise RuntimeError(err.value.decode() or str(v))
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p);destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('execution_mode',[c.c_void_p,c.c_int]);partition=bind('partition_threshold',[c.c_void_p,c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int]);state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6
metadata={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'librarySha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'tokensFileSha256':hashlib.sha256(a.tokens_file.read_bytes()).hexdigest(),'tokens':len(tokens),'kvBytes':4,'weightBits':8,'mode':5,'partitionThreshold':a.partition_threshold,'warmups':1,'trials':3,'kind':'native-prefix-not-e2e','limitations':['No network, Metal suffix, KV export overlap or decode','Wall includes Python call/readback; snapshots and comparisons excluded','FP32 only; formal BF16/128K gates remain open']}
if a.reference:
 refmeta=json.loads((a.reference/'result.json').read_text())
 for key in ['tokensFileSha256','tokens','kvBytes','weightBits','mode','librarySha256']:assert refmeta[key]==metadata[key],key
 assert refmeta['allPassed'] and len(refmeta['records'])==4
checks=[];records=[];metadata.update(checks=checks,records=records,allPassed=False)
def save(): (a.output/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
def compare(name,expected,actual):
 assert expected.shape==actual.shape
 # Chunk the float64 temporary to bound CPU RAM for long KV arrays.
 norm=0.;error=0.;maximum=0.;finite=True
 for start in range(0,expected.size,1<<20):
  x=expected[start:start+(1<<20)].astype(np.float64);y=actual[start:start+(1<<20)].astype(np.float64);d=x-y
  finite=finite and bool(np.isfinite(x).all() and np.isfinite(y).all());norm+=float(np.sum(x*x));error+=float(np.sum(d*d));maximum=max(maximum,float(np.max(np.abs(d))))
 rel=float(np.sqrt(error/max(norm,1e-30)));checks.append({'tensor':name,'relativeL2':rel,'maxAbs':maximum,'pass':finite and rel<1e-4 and maximum<.01})
h=create(a.weights.encode(),len(tokens),4,err,len(err))
if not h:raise RuntimeError(err.value.decode())
try:
 for trial in range(4):
  ok(reset(h,err,len(err)));ok(mode(h,5));ok(partition(h,a.partition_threshold));hidden=np.empty((len(tokens),2048),dtype=np.float32);start=time.perf_counter()
  for pos in range(0,len(tokens),32):
   part=tokens[pos:pos+32];ids=(c.c_int*len(part))(*part);ok(run(h,ids,len(part),pos,hidden[pos:].ctypes.data_as(c.POINTER(c.c_float)),None,err,len(err)))
  elapsed=time.perf_counter()-start;digest=hashlib.sha256(hidden).hexdigest();states=[]
  if trial==0:
   hidden.tofile(a.output/'hidden.f32')
   if a.reference:
    expected=np.memmap(a.reference/'hidden.f32',dtype=np.float32,mode='r',shape=hidden.shape)
    for pos in range(len(tokens)):compare('hidden/p'+str(pos),expected[pos],hidden[pos])
    del expected
  for layer in range(20):
   for tensor in range(2):
    size=len(tokens)*512*4 if (layer+1)%4==0 else 32*128*128*4 if tensor==0 else 3*8192*4
    raw=np.empty(size//4,dtype=np.float32);ok(state(h,layer,tensor,len(tokens),raw.ctypes.data,size,err,len(err)));states.append(hashlib.sha256(raw).hexdigest())
    if trial==0:
     path='state-'+str(layer)+'-'+str(tensor)+'.f32';raw.tofile(a.output/path)
     if a.reference:
      expected=np.memmap(a.reference/path,dtype=np.float32,mode='r');compare(path,expected,raw);del expected
  records.append({'trial':trial,'warmup':trial==0,'wallSeconds':elapsed,'hiddenSha256':digest,'stateSha256':states});save()
  if a.reference and not all(x['pass'] for x in checks):raise RuntimeError('Unchanged numerical gate failed; retained results, performance cohort stopped')
  assert digest==records[0]['hiddenSha256'] and states==records[0]['stateSha256'],'Nondeterministic native prefix'
  print('TRIAL',trial,'seconds',elapsed,flush=True)
 metadata['allPassed']=True;save();print('COMPLETE',a.output,flush=True)
finally:destroy(h)
