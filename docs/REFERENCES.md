# Source references

Reviewed 2026-10-20:

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c/tree/350e04fe35433e6d2941dce5a1f53308f87058eb): MIT-licensed reference for legacy checkpoint layout, BPE tokenizer semantics and transformer behavior. The compatible model loaders have been retired; this reference is retained for historical attribution. Required license attribution is retained in `THIRD-PARTY-NOTICES.md`.
- [Apple Metal runtime library compilation](https://developer.apple.com/documentation/metal/mtldevice/makelibrary(source:options:)): runtime shader compilation avoids requiring an offline Metal compiler for this bridge.
- [kkokosa/dotLLM](https://github.com/kkokosa/dotLLM): inspiration for backend contracts and CUDA Driver API discovery. No dotLLM source or kernels are copied/linked. That project declares GPL-3.0.
- [RazorConsole](https://github.com/RazorConsole/RazorConsole): optional MIT-licensed `RazorConsole.Core` 0.6.0 frontend. Native AOT remains experimental upstream; the local components were tested with one visible IL2104 warning.

NuGet transitive dependencies are recorded in the portable `packages.lock.json` files. RID-specific publish locks remain under `obj` to preserve portable locked restores.

- [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners): the unexecuted CI matrix uses macOS 15 arm64 for the bridge's minimum OS/API requirements and Linux x64 for CPU/synthetic regression.
