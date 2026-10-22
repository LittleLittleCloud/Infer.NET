#!/usr/bin/env python3
"""Pinned official Qwen2.5-3B-Instruct BF16 model. Retains the upstream Qwen Research License."""
import argparse,hashlib,json,os,pathlib,urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--size',choices=['3b','7b'],default='3b')
size=parser.parse_args().size
DIRECTORY=ROOT/f'models/qwen2.5-{size}-instruct'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
FILES={
 'config.json':'eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b',
 'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
 'tokenizer_config.json':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
 'generation_config.json':'ea35dfb6fc5051b01114f9b995820d55dab01ed33ee490f6378b442af82c09f9',
 'model.safetensors.index.json':'bc8aaa0c87d4335177e01c765f1de0db81661c67c1a72fbfb0d521b09f5ddc56',
 'LICENSE':'ef52482bb785733093dc9a2e8edd8e764c77d12d8e9d8f10a80c9b547d32d0f9',
 'model-00001-of-00002.safetensors':'67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2',
 'model-00002-of-00002.safetensors':'a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d',
}
if size=='7b':
 REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
 FILES={
  'config.json':'7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c',
  'tokenizer.json':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
  'tokenizer_config.json':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
  'generation_config.json':'3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f',
  'model.safetensors.index.json':'624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028',
  'LICENSE':'832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e',
  'model-00001-of-00004.safetensors':'a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7',
  'model-00002-of-00004.safetensors':'f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185',
  'model-00003-of-00004.safetensors':'8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5',
  'model-00004-of-00004.safetensors':'1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd',
 }

def digest(path):
 value=hashlib.sha256()
 with path.open('rb') as stream:
  while chunk:=stream.read(4*1024*1024):value.update(chunk)
 return value.hexdigest()
DIRECTORY.mkdir(exist_ok=True,parents=True)
for name,expected in FILES.items():
 target=DIRECTORY/name;temporary=target.with_suffix(target.suffix+'.download')
 if target.exists() and digest(target)==expected:print('Verified',name,flush=True);continue
 if temporary.exists() and digest(temporary)==expected:os.replace(temporary,target);print('Verified',name,flush=True);continue
 try:
  print('Downloading',name,flush=True);size=0;checkpoint=0
  with urllib.request.urlopen(f'https://huggingface.co/Qwen/Qwen2.5-{size.upper()}-Instruct/resolve/{REVISION}/{name}',timeout=120) as response,temporary.open('wb') as output:
   while chunk:=response.read(4*1024*1024):
    output.write(chunk);size+=len(chunk)
    if size-checkpoint>=256*1024*1024:print(f'{name}: {size//(1024*1024)} MiB',flush=True);checkpoint=size
  if digest(temporary)!=expected:raise RuntimeError(f'SHA-256 mismatch: {name}')
  os.replace(temporary,target)
 finally:temporary.unlink(missing_ok=True)
(DIRECTORY/'provenance.json').write_text(json.dumps({'modelId':f'qwen2.5-{size}-instruct','repository':f'https://huggingface.co/Qwen/Qwen2.5-{size.upper()}-Instruct','revision':REVISION,'license':'Apache-2.0' if size=='7b' else 'Qwen Research License','sha256':FILES},indent=2)+'\n')
