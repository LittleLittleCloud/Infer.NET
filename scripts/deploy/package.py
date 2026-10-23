#!/usr/bin/env python3
"""Package an existing Native AOT publish directory, with content hashes and compatibility metadata."""
import argparse,hashlib,json,pathlib,re,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--publish-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--rid',choices=['osx-arm64','win-x64'],required=True);a=p.parse_args()
repo=pathlib.Path(__file__).resolve().parents[2];source=pathlib.Path(a.publish_dir).resolve();out=pathlib.Path(a.output).resolve()
if out.exists():raise SystemExit('Output already exists; releases are immutable')
for name in (['deepinfer','libdeepinfer_metal.dylib'] if a.rid=='osx-arm64' else ['deepinfer.exe','deepinfer_cuda.dll']):
 if not (source/name).is_file():raise SystemExit('Missing native artifact: '+name)
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise SystemExit('Commit source before packaging')
version=subprocess.check_output([str(source/('deepinfer' if a.rid=='osx-arm64' else 'deepinfer.exe')),'version'],text=True).strip()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
if head not in version:raise SystemExit('Published binary was built from a different commit; rebuild it')
wire_source=(repo/'src/DeepInfer.Runtime/SplitWire.cs').read_text()
wire_version=re.search(r'private const int WireVersion\s*=\s*(\d+)\s*;',wire_source)
if wire_version is None:raise SystemExit('Cannot determine split wire version from source')
shutil.copytree(source,out,ignore=shutil.ignore_patterns('*.pdb','*.dbg','*.dSYM'))
shutil.copytree(repo/'scripts/deploy',out/'deploy',ignore=shutil.ignore_patterns('__pycache__'))
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
files={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()}
manifest={'source':sha,'rid':a.rid,'modelRevision':'e06a74e6236a60c8367e1a3214e83d8b61b637b0','weightBits':8,'kv':'fp32','splitProtocol':int(wire_version.group(1)),'maxContextTokens':262144,'maxOutputTokens':262144,'requiredPlatform':'macOS 26+ / Apple M5' if a.rid=='osx-arm64' else 'Windows x64 / RTX 4090 SM89 / CUDA 13 compatible driver','files':files}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out)
