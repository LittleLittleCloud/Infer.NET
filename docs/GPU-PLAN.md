# GPU status and next milestones

Qwen3.6-35B-A3B is the only supported real model. Dense Qwen2.5 and TinyStories execution paths have been removed; CPU numerical primitives and weight-free RPC simulation remain for tests.

## Metal

The original C# / Metal engine supports the same model's 5bit and 8bit affine checkpoints. The complete forward, attention, recurrent state, Top-8 experts and argmax run on GPU. The CLI/demo launcher defaults to 8bit. See [architecture](ARCHITECTURE.md), [8bit evidence](QWEN36-8BIT.md) and [fixed acceptance gates](QWEN36-8BIT-TARGETS.md).

## CUDA and dual-host execution

The dedicated RTX 4090 prefix agent and Mac Metal serve implement 20/20 layer handoff and state transfer. Generic CUDA full-model loading is not implemented. Deployment and current limitations are documented in [deployment](DEPLOYMENT.md), [split prefill](SPLIT-PREFILL.md) and [decode diagnosis](SPLIT-DECODE-DIAGNOSIS.md). Short-input or operator checks do not establish the complete 128K performance/resource gates.
