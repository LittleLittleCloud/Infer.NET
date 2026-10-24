# Synthetic speculative state reference (Issue #72, S0)

Default-off, Mac-only (`metal-local`) correctness reference for speculative proposal blocks. It validates full 40-layer FP32
state ownership, serial verification, rejection rollback/replay and emission semantics. **It contains no draft model, no real
DFlash2 component and no performance claim.** Proposals come from an oracle (the target's own greedy continuation, optionally
deliberately corrupted), so the counters below measure state correctness, never acceptance quality or speedup.

## Semantics

- `pending` is the target's already-computed prediction for the next output token. It is emitted before any forward and consumed
  only by the next cycle's verification. The final output token is never consumed (same as the standard loop).
- A cycle with block `k = min(B, remaining-2)` (further truncated to land on a checkpoint) captures the state, consumes
  `[pending, d0..d(k-1)]` serially and finds the first mismatch `j`. Accepted tokens are `d0..d(j-1)`; the target's token after
  `inputs[j]` is the correction (or the bonus token when all `k` are accepted).
- On rejection the engine restores the snapshot, replays `inputs[0..j]` and requires the replay prediction to equal the verified
  correction. Any disagreement throws `SpeculativeStateException`.
- Terminal boundaries match the standard loop exactly. Length termination leaves the last emitted token unconsumed
  (`position == prompt + N - 1`). A natural EOS leaves every emitted token consumed and the EOS pending
  (`position == prompt + N`, `FinalPending` is the EOS). EOS is never emitted unless `BenchmarkIgnoreEos`.
- A real EOS among matched proposals of a block truncates the cycle: the engine restores the snapshot, replays only the committed
  prefix `inputs[0..e]` (`e` = EOS offset), cross-checks the replay prediction, and stops. No bonus/correction is counted.
  An EOS that follows a reject offset is reached later as a natural pending EOS. The engine verifies the terminal position and
  throws `SpeculativeStateException` on any mismatch. Cancellation or any failure abandons the target; the next `Reset` and
  prefill recover it.
- Diagnostics distinguish `matched` (proposals equal to the target's tokens, including ones cut off by EOS) from `accepted`
  (committed proposals); `emitted = 1 + accepted` per cycle, and `eosTruncated` marks truncated cycles.
- Patterns: `accept-all`, `reject-all` (first proposal wrong), `offset:N` (N = 0..7 is the first wrong proposal; N ≥ block means
  accept all), `mixed` (deterministic cycle-dependent offsets). Block size is bounded to 1..8.

## Native snapshot ABI (`native/DeepInfer.Metal/graph_snapshot.mm`)

Handles are opaque 64-bit ids resolved through the owning graph (no pointers cross the ABI). Calls validate graph identity, reset
epoch, caller request id, KV precision, model geometry/weight-buffer identity, split/diagnostic-routing state, layer registration
and position. A snapshot copies only the 30 DeltaNet state/convolution-history tensors (~62.7 MiB); the 10 attention layers keep
append-only KV and the restored position alone bounds visibility, so a rejected suffix is never read. Storage is allocated by
`di_graph_snapshot_create` only, so a graph that never speculates allocates nothing. The split prefix-import API is not used.

| Function | Purpose |
| --- | --- |
| `di_graph_snapshot_create/capture/restore/release/info` | lifecycle; at most two live snapshots per graph |
| `di_graph_state_digest`, `di_graph_state_export` | read-only per-layer state identity (attention hashes rows `[0,position)` only) |
| `di_graph_last_logits` | exact copy of the last classifier output |
| `di_graph_poison_kv_suffix` | harness hook: NaN-fills KV rows past the position to prove they are never read |

## Service option (default off)

```
deepinfer split-serve --config mac.json --short-prefill-tokens 262144 \
  --speculative-synthetic mixed --speculative-block-tokens 4
```

Requires metal-local execution (`--short-prefill-tokens 262144`), FP32 KV and `--kv-transfer final`. `--decode-attention`
`fused-fp32` (the deployed baseline, with `--suffix-chunk-tokens 128 --metal-scratch-tokens 128`; local prefill independently uses managed chunk 32) and `legacy` are both supported
and the same native configuration runs with and without the option. Restore rewinds the recurrent/convolution state and the
logical KV position; fused decode attention keeps no cross-token state (its partials live in a scratch buffer every forward fully
rewrites and bounds by the position), so nothing extra needs snapshotting. Other combinations, other commands and `split-agent` reject the option explicitly. When enabled, responses carry
`speculative` diagnostics (per-cycle proposed/verified/accepted/corrected/bonus/emitted counters, snapshot, restore, replay and
verification milliseconds, `firstEmissionMs`, `emissionLifecycleMs`), a `resources` block (separate `prefillMs`, context capacity,
KV bytes, Metal allocated bytes, snapshot host bytes, decode attention, scratch and prefill chunk) and **`timings` is `null`**: block emission has no SOP
N-1 decode interval and none is fabricated. `lastRequest` telemetry is not published. The service uses the target-rollout oracle
(capture, rollout, restore; rollout forwards are counted separately as `oracleForwards`). Candidate-off code paths, allocation and
timings are unchanged and the field is omitted from responses.

## Checks

CPU (no GPU): `dotnet test tests/DeepInfer.Tests -c Release --filter SpeculativeEngineTests`; native compile only:
`bash scripts/check-metal-speculative-state.sh`.

GPU fixtures (run by the owner after stopping the idle worker): `bash scripts/check-metal-speculative-state.sh --run` exercises
every block size 1..8 × every rejection offset and all-accepted, repeated rollback, KV visibility with NaN poison, and the
reset/stale/request/ownership/release/invalidation lifecycle on real Metal buffers (no weights).

Model harness, one model load for all cases:

```
python3 scripts/run-speculative-state.py --exe PATH/deepinfer --token-ids ids.json --poison-suffix   # default matrix
python3 scripts/run-speculative-state.py --exe PATH/deepinfer --token-ids ids.json \
  --output-lengths 128 --patterns offset:3,mixed --blocks 4,8 --ignore-eos                           # custom cross product
python3 scripts/run-speculative-state.py --exe PATH/deepinfer --humaneval-jsonl HumanEval.jsonl --limit 3 ...
```

The default matrix (warm 1 / formal 3 only for the full-length groups): `offsets-128` (every reject offset 0..7, accept-all and
mixed at blocks 8/4/2/1), `patterns-4096` (accept-all and mixed at blocks 8/4, reject-all at block 2), `caps` (blocks 1,2,3,4,8 ×
caps 1..block+2 × four patterns, single run) and `natural-eos` (4096-token cap, stop at the model's EOS, blocks 1..8 plus
mixed/offset:2/reject-all). The harness uses the deployed native profile (`fused-fp32`, chunk 32, scratch 128) by default and
verifies what the graph actually runs. Every run failure is retained in `speculative-check.json`.

`speculative-check` generates one non-speculative baseline per prompt (full-state digests and exact logits at checkpoints), then
replays every pattern/block/length through the real engine and compares tokens, state digests and bitwise logits at checkpoints and at every endpoint (terminal position, final pending token,
terminal state and logits, including EOS and length stops), plus a lifecycle probe
(cancellation, abandonment, snapshot staleness after reset). First divergence writes `failure-N.json` plus the actual and independently regenerated expected logits and state tensors
(`failure-N.*.f32`) with first differing element and max difference; JSON never contains non-finite numbers. HumanEval with the enabled
service option exercises the same engine and native snapshot ABI over the real request path (`speculative` diagnostics in each
result); `timings` is `null` there, so SOP decode throughput must come from candidate-off runs.

## Limitations

Synthetic oracle only; FP32 KV only; single local request; no batching, real draft taps/weights or classifier. Capture cost
scales with the 62.7 MiB recurrent state and is reported, not hidden.
