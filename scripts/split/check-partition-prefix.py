#!/usr/bin/env python3
"""Compare native whole-prefix arithmetic with forced partitioned attention.
131 synthetic IDs, irregular tails, each layer/token and40final states. No speed claim.
An idle GPU is required. Does not stop processes or relax the fixed tensor tolerance.
"""
import argparse,array,ctypes as c,hashlib,json,math,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--kv-bytes',type=int,choices=[2,4],default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sha=hashlib.sha256()
with open(a.weights,'rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):sha.update(block)
assert sha.hexdigest()=='ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'
dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(v):
 if v:raise RuntimeError(err.value.decode() or str(v))
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p);destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('execution_mode',[c.c_void_p,c.c_int]);partition=bind('partition_threshold',[c.c_void_p,c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int]);state=bind('state',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_char_p,c.c_int])
assert bind('abi_version',[])()==6
h=create(a.weights.encode(),131,a.kv_bytes,err,len(err))
if not h:raise RuntimeError(err.value.decode())
tokens=[100+(i*137)%247000 for i in range(131)];checks=[]
def values(raw,bf16=False):
 if bf16:
  shorts=array.array('H');shorts.frombytes(raw);v=array.array('I',(x<<16 for x in shorts));raw=v.tobytes()
 result=array.array('f');result.frombytes(raw);return result
def compare(name,expected,actual):
 if len(expected)!=len(actual) or not all(math.isfinite(x) for x in actual):raise RuntimeError('Geometry/nonfinite '+name)
 maximum=0;error=0;norm=0
 for x,y in zip(expected,actual):d=x-y;maximum=max(maximum,abs(d));error+=d*d;norm+=x*x
 rel=math.sqrt(error/max(1e-30,norm));checks.append({'tensor':name,'maxAbs':maximum,'relativeL2':rel,'pass':maximum<.01 and rel<1e-4})
try:
 assert partition(h,-2)!=0 and partition(h,262145)!=0
 for candidate in [False,True]:
  ok(reset(h,err,len(err)));ok(mode(h,5));ok(partition(h,0 if candidate else -1));pos=0;index=0
  while pos<len(tokens):
   n=min([1,7,17,32][index%4],len(tokens)-pos);ids=(c.c_int*n)(*tokens[pos:pos+n]);hidden=(c.c_float*(n*2048))();trace=(c.c_float*(n*20*2048))()
   ok(run(h,ids,n,pos,hidden,trace,err,len(err)));assert partition(h,0)!=0,'Midrequest mode change accepted'
   path=a.output/f'trace-{pos}.f32'
   if not candidate:path.write_bytes(bytes(trace))
   else:
    ref=values(path.read_bytes());actual=values(bytes(trace))
    for t in range(n):
     for layer in range(20):
      offset=(t*20+layer)*2048;compare(f'hidden/p{pos+t}/l{layer}',ref[offset:offset+2048],actual[offset:offset+2048])
   pos+=n;index+=1
  for layer in range(20):
   for tensor in range(2):
    attention=(layer+1)%4==0;size=131*512*a.kv_bytes if attention else 32*128*128*4 if tensor==0 else 3*8192*4
    raw=c.create_string_buffer(size);ok(state(h,layer,tensor,131,raw,size,err,len(err)));path=a.output/f'state-{layer}-{tensor}.bin'
    if not candidate:path.write_bytes(raw.raw)
    else:compare(f'state/l{layer}/t{tensor}',values(path.read_bytes(),attention and a.kv_bytes==2),values(raw.raw,attention and a.kv_bytes==2))
  print('Completed candidate',candidate,flush=True)
 result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'librarySha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'kvBytes':a.kv_bytes,'tokens':131,'allPassed':all(x['pass'] for x in checks),'checks':checks,'kind':'whole-native-prefix-numerical-test-not-e2e'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2));print('Checks',len(checks),'failures',sum(not x['pass'] for x in checks),'maxRelativeL2',max(x['relativeL2'] for x in checks),flush=True)
 if not result['allPassed']:raise SystemExit(1)
finally:destroy(h)
