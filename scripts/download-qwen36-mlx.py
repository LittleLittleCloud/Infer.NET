#!/usr/bin/env python3
"""Download pinned affine Qwen3.6 weights for the original .NET engine and independent reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

REVISIONS = {5: 'c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499',
             8: 'e06a74e6236a60c8367e1a3214e83d8b61b637b0'}
UPSTREAM_REVISION = '995ad96eacd98c81ed38be0c5b274b04031597b0'
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bits', type=int, choices=REVISIONS, default=5)
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--max-workers', type=int, default=3)
    args = parser.parse_args()
    repository = f'mlx-community/Qwen3.6-35B-A3B-{args.bits}bit'
    revision = REVISIONS[args.bits]
    args.directory = args.directory or ROOT / f'models/qwen3.6-35b-a3b-{args.bits}bit-mlx'
    args.directory.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != revision:
        raise RuntimeError('Repository revision does not match the pinned commit')
    files = [s for s in metadata['siblings']
             if s['rfilename'].endswith(('.json', '.safetensors', '.jinja', '.md'))]
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    from huggingface_hub import snapshot_download
    print(f'Checkpoint: {repository}@{revision}', flush=True)
    print(f'Downloading {sum(s["size"] for s in files) / 1e9:.3f} GB', flush=True)
    snapshot_download(repository, revision=revision, local_dir=args.directory,
                      allow_patterns=[s['rfilename'] for s in files],
                      max_workers=args.max_workers)
    hashes = {}
    for entry in files:
        name = entry['rfilename']
        path = args.directory / name
        if path.stat().st_size != entry['size']:
            raise RuntimeError(f'File size mismatch: {name}')
        actual = digest(path)
        if entry.get('lfs'):
            if actual != entry['lfs']['sha256']:
                raise RuntimeError(f'SHA-256 mismatch: {name}')
        elif entry.get('blobId'):
            data = path.read_bytes()
            git_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if git_hash != entry['blobId']:
                raise RuntimeError(f'Git blob hash mismatch: {name}')
        hashes[name] = actual
        print(f'Verified {name}', flush=True)
    license_url = (f'https://huggingface.co/Qwen/Qwen3.6-35B-A3B/'
                   f'resolve/{UPSTREAM_REVISION}/LICENSE')
    with urllib.request.urlopen(license_url, timeout=60) as response:
        (args.directory / 'LICENSE').write_bytes(response.read())
    hashes['LICENSE'] = digest(args.directory / 'LICENSE')
    provenance = dict(repository=repository, revision=revision,
                      upstream='Qwen/Qwen3.6-35B-A3B', upstreamRevision=UPSTREAM_REVISION,
                      license='Apache-2.0', licenseSource=license_url,
                      purpose='Packed weights for original DeepInfer.NET inference and independent MLX verification',
                      sha256=hashes)
    (args.directory / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Verified baseline model:', args.directory, flush=True)


if __name__ == '__main__':
    main()
