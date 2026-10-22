#!/usr/bin/env python3
"""Download the pinned MIT-licensed TinyStories checkpoint and matching llama2.c tokenizer; verify SHA-256."""
import hashlib, os, pathlib, urllib.request
ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [
    ("stories15M.bin", "https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da7698eaf29a0d7de3992de8a46ef624add/stories15M.bin", "cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a"),
    ("tokenizer.bin", "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.bin", "50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361"),
]
for name, url, expected in FILES:
    target = ROOT / "models" / name
    target.parent.mkdir(exist_ok=True)
    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024*1024): value.update(chunk)
        return value.hexdigest()
    if target.exists() and digest(target) == expected:
        print(f"Verified {name}")
        continue
    temporary = target.with_suffix(".download")
    try:
        print(f"Downloading {name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024*1024): stream.write(chunk)
        if digest(temporary) != expected: raise RuntimeError(f"SHA-256 mismatch: {name}")
        os.replace(temporary, target)
    finally: temporary.unlink(missing_ok=True)
