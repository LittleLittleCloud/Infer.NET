#!/usr/bin/env python3
"""Trace retained versus partitioned CUDA prefix at the first 32K gate failure.
Requires an idle RTX 4090, the pinned weights and exact fixed token ID artifact.
"""
import argparse,ctypes as c,hashlib,json,math,pathlib,subprocess,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--library',required=True);p.add_argument('--tokens-file',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--position',type=int,default=14975);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
tokens=json.loads(a.tokens_file.read_text());assert len(tokens)==32768 and 0<=a.position<32768
sha=hashlib.sha256()
with open(a.weights,'rb') as f:
 for block in iter(lambda:f.read(8<<20),b''):sha.update(block)
assert sha.hexdigest()=='ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77'
dll=c.CDLL(str(pathlib.Path(a.library).resolve()));err=c.create_string_buffer(2048)
def bind(name,args,ret=c.c_int):
 f=getattr(dll,'di_cuda_prefix_'+name);f.argtypes=args;f.restype=ret;return f
def ok(v):
 if v:raise RuntimeError(err.value.decode() or 'native code '+str(v))
create=bind('create',[c.c_char_p,c.c_int,c.c_int,c.c_char_p,c.c_int],c.c_void_p);destroy=bind('destroy',[c.c_void_p],None);reset=bind('reset',[c.c_void_p,c.c_char_p,c.c_int]);mode=bind('execution_mode',[c.c_void_p,c.c_int]);partition=bind('partition_threshold',[c.c_void_p,c.c_int]);run=bind('run',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.c_int,c.POINTER(c.c_float),c.POINTER(c.c_float),c.c_char_p,c.c_int])
probe=bind('route_probe',[c.c_void_p,c.c_int,c.c_int]);route=bind('route_read',[c.c_void_p,c.POINTER(c.c_int),c.c_int,c.POINTER(c.c_float),c.c_int,c.POINTER(c.c_float),c.c_int])
assert bind('abi_version',[])()==6
h=create(a.weights.encode(),32768,4,err,len(err))
if not h:raise RuntimeError(err.value.decode())
traces={};routes={};last=a.position//32*32;end=last+32
try:
 for name,threshold in [('legacy',-1),('partition',8192)]:
  ok(reset(h,err,len(err)));ok(mode(h,5));ok(partition(h,threshold));ok(probe(h,17,a.position));start=time.perf_counter()
  for pos in range(0,end,32):
   part=tokens[pos:pos+32];ids=(c.c_int*len(part))(*part);hidden=(c.c_float*(len(part)*2048))()
   trace=(c.c_float*(len(part)*20*2048))() if pos==last else None
   ok(run(h,ids,len(part),pos,hidden,trace,err,len(err)))
   if trace is not None:
    raw=np.ctypeslib.as_array(trace).copy().reshape((32,20,2048));raw.tofile(a.output/(name+'-trace.f32'));traces[name]=raw
  print(name,'prefixSeconds',time.perf_counter()-start,flush=True)
  ids=(c.c_int*8)();logits=(c.c_float*256)();probs=(c.c_float*8)();ok(route(h,ids,8,logits,256,probs,8))
  ranked=sorted(range(256),key=lambda i:(-logits[i],i))
  routes[name]={'ids':list(ids),'prob':list(probs),'logits':list(logits),'top9':[(i,float(logits[i])) for i in ranked[:9]],'top8Margin':float(logits[ranked[7]]-logits[ranked[8]])}
 checks=[]
 for t in range(32):
  for layer in range(20):
   x=traces['legacy'][t,layer].astype(np.float64);y=traces['partition'][t,layer].astype(np.float64);d=x-y
   relative=math.sqrt(float(np.sum(d*d))/max(1e-30,float(np.sum(x*x))));maximum=float(np.max(np.abs(d)))
   checks.append({'token':last+t,'layer':layer,'relativeL2':relative,'maxAbs':maximum,'pass':math.isfinite(relative) and relative<1e-4 and maximum<.01})
 target=[c for c in checks if c['token']==a.position]
 result={'source':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'librarySha256':hashlib.sha256(pathlib.Path(a.library).read_bytes()).hexdigest(),'tokensFileSha256':hashlib.sha256(a.tokens_file.read_bytes()).hexdigest(),'position':a.position,'chunkStart':last,'firstNonzero':next((x for x in checks if x['relativeL2']>0),None),'firstGateFailure':next((x for x in checks if not x['pass']),None),'targetFirstGateFailure':next((x for x in target if not x['pass']),None),'targetLayerChecks':target,'routes':routes,'checks':checks,'kind':'CUDA-prefix-diagnostic-not-E2E'}
 (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('FIRST_GATE',result['firstGateFailure'],'TARGET_FIRST',result['targetFirstGateFailure'],flush=True)
finally:destroy(h)
