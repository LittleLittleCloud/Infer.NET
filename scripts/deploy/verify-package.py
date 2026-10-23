#!/usr/bin/env python3
import argparse,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();root=pathlib.Path(a.directory).resolve();m=json.loads((root/'manifest.json').read_text())
for name,expected in m['files'].items():
 path=(root/name).resolve()
 if root not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise SystemExit('Package verification failed: '+name)
print('Verified',m['rid'],m['source'],len(m['files']),'files')
