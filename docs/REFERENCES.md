# Source references

Reviewed 2026-10-20:

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb): MIT-licensed reference for legacy checkpoint layout, BPE tokenizer semantics and transformer behavior. DeepInfer implements these in C# and compares against an independently compiled reference executable. Required license attribution is retained in `THIRD-PARTY-NOTICES.md`.
- [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas/tree/0bd21da7698eaf29a0d7de3992de8a46ef624add): MIT-labelled pretrained TinyStories 15M checkpoint, pinned revision and SHA-256 in `models/provenance.json`.
- [Apple Metal runtime library compilation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)): runtime shader compilation avoids requiring an offline Metal compiler for this bridge.
- [kkokosa/dotLLM](https://github.com/kkokosa/dotLLM): inspiration for backend contracts and CUDA Driver API discovery. No dotLLM source or kernels are copied/linked. That project declares GPL-3.0.
- [RazorConsole](https://github.com/RazorConsole/RazorConsole): optional MIT-licensed `RazorConsole.Core` 0.6.0 frontend. Native AOT remains experimental upstream; the local components were tested with one visible IL2104 warning.

NuGet transitive dependencies are recorded in the portable `packages.lock.json` files. RID-specific publish locks remain under `obj` to preserve portable locked restores.

- [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners): the unexecuted CI matrix uses macOS 15 arm64 for the bridge's minimum OS/API requirements and Linux x64 for CPU/synthetic regression.

- [Official Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1): pinned official BF16 shards, config, tokenizer and Qwen Research License; SHA-256 digests retained in model provenance.
- [Transformers Qwen2 implementation](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2): independent architecture/FP32 inference reference. Python dependencies are acceptance tooling only; the product does not invoke them.

- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28): official BF16 model, untied classifier, configuration/tokenizer and Apache-2.0 notice; all four checkpoint shards pinned and SHA-256 verified.
