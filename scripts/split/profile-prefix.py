#!/usr/bin/env python3
"""Profile the native FP32 layer-major prefix on fixed token IDs. No model replacement.
Requires an idle GPU without a resident agent. Event instrumentation is diagnostic;
report unprofiled and profiled wall times separately, never as end-to-end throughput.
"""
import argparse,ctypes as c,hashlib,json,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--compare-reuse',action='store_true');p.add_argument('--packed',action='store_true');p.add_argument('--packed-experts',action='store_true');p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--tokens-file',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);tokens=json.loads(a.tokens_file.read_text());assert 0<len(tokens)<=262144 and all(type(x) is int and 0<=x<248320 for x in tokens)
sha=hashlib.sha256()
with open(a.weights,'rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):sha.update(block)
assert sha.hexdigest()=='ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77','Pinned prefix mismatch'
dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(result):
 if result:raise RuntimeError(err.value.decode() or 'Native return '+str(result))
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p);destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('execution_mode',[c.c_void_p,c.c_int]);profile=bind('profile_mode',[c.c_void_p,c.c_int]);read=bind('profile_read',[c.c_void_p,c.POINTER(c.c_double),c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int]);state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6
h=create(a.weights.encode(),len(tokens),4,err,len(err))
if not h:raise RuntimeError(err.value.decode())
records=[]
try:
 assert profile(h,-1)!=0 and read(h,None,9)!=0
 runs=[('warmup',0,1),('unprofiled',0,1),('profiled',1,1)]
 if a.compare_reuse:runs=[(f'{label}-{i}',0,modeId) for label,modeId in [('legacy',1),('reuse',5 if a.packed_experts else 4 if a.packed else 3)] for i in range(4)]
 for label,enabled,modeId in runs:
  ok(reset(h,err,len(err)));ok(mode(h,modeId));ok(profile(h,enabled));digest=hashlib.sha256();start=time.perf_counter()
  for pos in range(0,len(tokens),32):
   part=tokens[pos:pos+32];ids=(c.c_int*len(part))(*part);hidden=(c.c_float*(len(part)*2048))();ok(run(h,ids,len(part),pos,hidden,None,err,len(err)));digest.update(bytes(hidden))
  elapsed=time.perf_counter()-start;values=(c.c_double*9)();ok(read(h,values,9));states=[]
  for layer in range(20):
   for tensor in range(2):
    size=len(tokens)*512*4 if (layer+1)%4==0 else 32*128*128*4 if tensor==0 else 3*8192*4
    data=c.create_string_buffer(size);ok(state(h,layer,tensor,len(tokens),data,size,err,len(err)));states.append(hashlib.sha256(data.raw).hexdigest())
  record={'label':label,'mode':modeId,'wallSeconds':elapsed,'hiddenSha256':digest.hexdigest(),'stateSha256':states,'phaseMilliseconds':dict(zip(['norm_and_input_projections','delta_convolution','attention','output_projection_norm','router_shared_gate_up','expert_gate_up','expert_down_shared_down','merge','hidden_readback'],values))};records.append(record);print(json.dumps(record),flush=True)
 assert len({x['hiddenSha256'] for x in records})==1 and all(x['stateSha256']==records[0]['stateSha256'] for x in records),'Profiling changed arithmetic'
 result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'tokens':len(tokens),'tokensFileSha256':hashlib.sha256(a.tokens_file.read_bytes()).hexdigest(),'weightBits':8,'kv':'fp32','prefill':'batch-fp32','dllSha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'allHiddenAnd40StatesEqual':True,'records':records,'kind':'native-prefix-phase-diagnostic-not-e2e','limitations':['CUDA event intervals can include host dispatch gaps','Profiling creates events and adds overhead; compare separate unprofiled wall time','No network, suffix, decode or incremental KV export in this probe']}
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('PASS phase profile leaves hidden and40states unchanged',flush=True)
finally:destroy(h)
