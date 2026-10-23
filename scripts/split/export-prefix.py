#!/usr/bin/env python3
"""Export the pinned first 20 layers without changing quantization. No inference dependency."""
import argparse, hashlib, json, pathlib, struct
REVISION = 'e06a74e6236a60c8367e1a3214e83d8b61b637b0'
p = argparse.ArgumentParser(); p.add_argument('model', type=pathlib.Path); p.add_argument('output', type=pathlib.Path)
a = p.parse_args(); cfg = json.loads((a.model/'config.json').read_text()); q = cfg['quantization']
assert q['bits']==8 and q['group_size']==64 and q['mode']=='affine'
index = json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
provenance = json.loads((a.model/'provenance.json').read_text())
assert REVISION in json.dumps(provenance), 'Unexpected revision'
files={}; headers={}; bases={}
def tensor(name):
    shard=index[name]
    if shard not in files:
        f=(a.model/shard).open('rb'); n=struct.unpack('<Q',f.read(8))[0]; headers[shard]=json.loads(f.read(n)); bases[shard]=8+n; files[shard]=f
    t=headers[shard][name]; f=files[shard]; start,end=t['data_offsets']; f.seek(bases[shard]+start)
    return t,f,end-start
h=hashlib.sha256(); a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.with_suffix('.partial').open('wb') as out:
    def write(data): h.update(data); out.write(data)
    def matrix(prefix):
        if prefix is None: write(struct.pack('<III',0,0,0)); return
        override=q.get(prefix, q); assert override['bits']==8 and override['group_size']==64
        t,f,n=tensor(prefix+'.weight'); shape=t['shape']; assert t['dtype']=='U32'
        experts=shape[0] if len(shape)==3 else 1; rows=shape[-2]; cols=shape[-1]*4
        write(struct.pack('<III',experts,rows,cols))
        for suffix,dtype,size in [('.weight','U32',experts*rows*cols),('.scales','BF16',experts*rows*cols//32),('.biases','BF16',experts*rows*cols//32)]:
            t,f,n=tensor(prefix+suffix); assert t['dtype']==dtype and n==size
            while n:
                data=f.read(min(n,8<<20)); assert data; write(data); n-=len(data)
    def vector(name, count):
        t,f,n=tensor(name); data=f.read(n)
        if t['dtype']=='BF16': data=b''.join(b'\0\0'+data[i:i+2] for i in range(0,n,2))
        else: assert t['dtype']=='F32'
        assert len(data)==count*4
        return data
    write(b'DIPFX001'+REVISION.encode('ascii'))
    write(struct.pack('<ff',cfg['text_config']['rms_norm_eps'],cfg['text_config']['rope_parameters']['rope_theta']))
    matrix('language_model.model.embed_tokens')
    for l in range(20):
        pre=f'language_model.model.layers.{l}.'; linear=(l+1)%4!=0
        att=pre+('linear_attn.' if linear else 'self_attn.')
        names=['in_proj_qkv',None,None,'in_proj_z','in_proj_a','in_proj_b','out_proj'] if linear else ['q_proj','k_proj','v_proj',None,None,None,'o_proj']
        for n in names: matrix(att+n if n else None)
        for n in ['gate','switch_mlp.gate_proj','switch_mlp.up_proj','switch_mlp.down_proj','shared_expert.gate_proj','shared_expert.up_proj','shared_expert.down_proj','shared_expert_gate']: matrix(pre+'mlp.'+n)
        params=bytearray(37568*4)
        specs=[(pre+'input_layernorm.weight',0,2048),(pre+'post_attention_layernorm.weight',2048,2048)]
        specs+= [(att+'conv1d.weight',4096,32768),(att+'A_log',36864,32),(att+'dt_bias',36896,32),(att+'norm.weight',36928,128)] if linear else [(att+'q_norm.weight',37056,256),(att+'k_norm.weight',37312,256)]
        for name,offset,count in specs: params[offset*4:(offset+count)*4]=vector(name,count)
        write(params); print(f'layer={l} bytes={out.tell()}',flush=True)
size=a.output.with_suffix('.partial').stat().st_size; a.output.with_suffix('.partial').replace(a.output)
a.output.with_suffix('.json').write_text(json.dumps(dict(format='DIPFX001',revision=REVISION,sha256=h.hexdigest(),bytes=size,layers=[0,20],quantization='affine8/group64'),indent=2)+'\n')
print(h.hexdigest(),size,flush=True)
