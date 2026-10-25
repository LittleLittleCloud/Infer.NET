# Experiment 84: bounded prefill delivery windows

The user approved [Issue #84](https://github.com/LittleLittleCloud/Infer.NET/issues/84)
on 2026-10-25, alongside independent Q8 Issue #33. Baseline:
`4f51f241e87999c8cc3f79d733cd95ade9c711cd`. This branch starts from that
main and retains FP32 arithmetic, KV, Delta/convolution state and affine8 weights.

The candidate adds explicit 32/64/128 completed-prefix execution windows while
preserving authenticated 32-token wire frames. Only rows that completed all
20 CUDA layers can be sent. The bounded sender still has one GPU owner, two
queued bundles and cancellation joining. Larger windows reserve correspondingly
bounded scratch; they are not assumed to improve end-to-end performance.

The default remains 32. Every hidden vector, layer trace and all 40 frontend
states must match the retained 32 path byte-for-byte before performance evaluation.
`scripts/split/check-delivery-windows.py` covers tails, boundaries and reset after
longer inputs. Subsequent fixed-request screening and HumanEval5/20 follow the
pre-registered gates in Issue #84. Code/results PRs and adoption review stay separate.

No Q8, BF16, speculation, expert-format changes or default adoption are included.
