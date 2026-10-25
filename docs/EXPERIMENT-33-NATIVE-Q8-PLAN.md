# Experiment 33: native Q8 KV and bounded state memory

The human user approved the concrete Issue [#33](https://github.com/LittleLittleCloud/Infer.NET/issues/33)
and independent pipeline Issue [#84](https://github.com/LittleLittleCloud/Infer.NET/issues/84)
on 2026-10-25. The approved baseline is `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.
The complete pre-registered scope, reference, quality, resource and HumanEval gates
are in Issue #33. Code adoption and merges require separate human review.

This candidate adds explicit Q8 profiles to native Metal/CUDA, uses the existing
little-endian block32 layout (544 bytes per K or V token), retains FP32 recurrence,
copies quantized state bytes across the existing authenticated protocol, and adds
bounded reservation checks. The service default stays FP32. The Q8 profile is
experimental. It is not an adopted baseline or evidence of 128K acceptance.

The independent fixture generator is `scripts/q8-kv-fixtures.py`. Freeze its
manifest before native codec evaluation. `q8_checks.mm` and `q8_checks.cu` compare
GPU bytes and decoded values to those fixtures, including zero/tie and invalid
scale cases. These model-free checks do not establish full-model correctness.

Evaluation proceeds through native codec/attention checks, layer/state/logit
checks, real split inference, pre-registered quality fixtures and resources,
then fixed HumanEval5 and frozen main/candidate HumanEval20. Record every failure
and missing metric; publish the eventual results from an independent main-based
results branch. Q8 versus FP32 is a cross-precision configuration comparison.

The pipeline experiment will start independently from main. This branch does
not incorporate its delivery-window or GEMM changes.
