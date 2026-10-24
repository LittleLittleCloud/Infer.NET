#!/usr/bin/env python3
"""Compare current-main and candidate BF16 Tensor Core CUDA prefixes.

This is a native, synthetic-token diagnostic. It does not measure RPC, Metal,
decode, or user-visible prefill. Load one model at a time on the same GPU.
"""
import argparse
import array
import ctypes as c
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def finite(data):
    floats = array.array("f")
    floats.frombytes(data)
    return all(math.isfinite(value) for value in floats)


def bind(dll, name, args, result=c.c_int):
    fn = getattr(dll, "di_cuda_prefix_" + name)
    fn.argtypes = args
    fn.restype = result
    return fn


def run_case(label, library, weights, output, trials, mode_id=2):
    dll = c.CDLL(str(library.resolve()))
    error = c.create_string_buffer(2048)
    create = bind(dll, "create", [c.c_char_p, c.c_int, c.c_int, c.c_char_p, c.c_int], c.c_void_p)
    destroy = bind(dll, "destroy", [c.c_void_p], None)
    reset = bind(dll, "reset", [c.c_void_p, c.c_char_p, c.c_int])
    mode = bind(dll, "execution_mode", [c.c_void_p, c.c_int])
    run = bind(dll, "run", [c.c_void_p, c.POINTER(c.c_int), c.c_int, c.c_int,
                             c.POINTER(c.c_float), c.POINTER(c.c_float), c.c_char_p, c.c_int])
    state = bind(dll, "state", [c.c_void_p, c.c_int, c.c_int, c.c_int,
                                 c.c_void_p, c.c_int, c.c_char_p, c.c_int])
    if bind(dll, "abi_version", [])() != 6:
        raise RuntimeError(label + ": unexpected CUDA ABI")

    def ok(code):
        if code:
            raise RuntimeError(label + ": " + error.value.decode(errors="replace"))

    handle = create(str(weights).encode(), 512, 4, error, len(error))
    if not handle:
        raise RuntimeError(label + ": model load: " + error.value.decode(errors="replace"))
    token_ids = [100 + (index * 137) % 247000 for index in range(512)]

    def execute(count, sizes, capture_trace):
        hidden, trace = bytearray(), bytearray()
        position = 0
        index = 0
        while position < count:
            size = min(sizes[index % len(sizes)], count - position)
            ids = (c.c_int * size)(*token_ids[position:position + size])
            out = (c.c_float * (size * 2048))()
            detail = (c.c_float * (size * 20 * 2048))() if capture_trace else None
            ok(run(handle, ids, size, position, out, detail, error, len(error)))
            hidden.extend(bytes(out))
            if detail is not None:
                trace.extend(bytes(detail))
            position += size
            index += 1
        if not finite(hidden) or (capture_trace and not finite(trace)):
            raise RuntimeError(label + ": nonfinite hidden or trace")
        return bytes(hidden), bytes(trace)

    try:
        ok(mode(handle, mode_id))  # 2: BF16 Tensor Core; 5: packed FP32.
        correctness_count = 131
        hidden, trace = execute(correctness_count, [1, 7, 17, 32], True)
        (output / (label + "-hidden.bin")).write_bytes(hidden)
        (output / (label + "-trace.bin")).write_bytes(trace)
        state_hashes = []
        for layer in range(20):
            for kind in range(2):
                length = (correctness_count * 512 * 4 if (layer + 1) % 4 == 0 else
                          32 * 128 * 128 * 4 if kind == 0 else 3 * 8192 * 4)
                data = c.create_string_buffer(length)
                ok(state(handle, layer, kind, correctness_count, data, length, error, len(error)))
                if not finite(data.raw):
                    raise RuntimeError(label + ": nonfinite state")
                raw = data.raw
                (output / f"{label}-state-{layer}-{kind}.bin").write_bytes(raw)
                state_hashes.append(sha(raw))
        records = []
        for trial in range(trials + 1):
            ok(reset(handle, error, len(error)))
            ok(mode(handle, mode_id))
            start = time.perf_counter()
            trial_hidden, _ = execute(512, [32], False)
            seconds = time.perf_counter() - start
            records.append({"trial": trial, "warmup": trial == 0, "seconds": seconds,
                            "tokensPerSecond": 512 / seconds, "hiddenSha256": sha(trial_hidden)})
            print(json.dumps({"label": label, **records[-1]}), flush=True)
        if len({item["hiddenSha256"] for item in records}) != 1:
            raise RuntimeError(label + ": nondeterministic full-prefix hidden")
        return {"label": label, "mode": mode_id, "librarySha256": sha(library.read_bytes()),
                "tokenIdsSha256": sha(json.dumps(token_ids, separators=(",", ":")).encode()),
                "hiddenSha256": sha(hidden), "traceSha256": sha(trace),
                "stateSha256": state_hashes, "records": records,
                "medianSeconds": statistics.median(item["seconds"] for item in records if not item["warmup"])}
    finally:
        destroy(handle)


def first_trace_difference(left, right):
    a = array.array("f")
    b = array.array("f")
    a.frombytes(left)
    b.frombytes(right)
    if len(a) != len(b):
        return {"lengthMismatch": [len(a), len(b)]}
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"token": index // (20 * 2048), "layer": (index // 2048) % 20,
                    "element": index % 2048, "baseline": x, "candidate": y}
    return None


def trace_error(left, right):
    a = array.array("f")
    b = array.array("f")
    a.frombytes(left)
    b.frombytes(right)
    if len(a) != len(b):
        raise ValueError("trace lengths differ")
    squared_error = squared_reference = maximum = 0.0
    for x, y in zip(a, b):
        difference = float(x) - float(y)
        squared_error += difference * difference
        squared_reference += float(y) * float(y)
        maximum = max(maximum, abs(difference))
    return {"relativeL2VsFp32": math.sqrt(squared_error / max(squared_reference, 1e-30)),
            "maxAbsVsFp32": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--baseline-library", required=True, type=Path)
    parser.add_argument("--candidate-library", required=True, type=Path)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 3:
        parser.error("at least three measured trials are required")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"kind": "native-bf16-tensor-core-prefix-not-end-to-end",
              "baselineCommit": args.baseline_commit, "candidateCommit": args.candidate_commit,
              "weightBits": 8, "kv": "fp32", "nativeChunkTokens": 32,
              "correctnessTokens": 131, "benchmarkTokens": 512, "measuredTrials": args.trials,
              "source": "scripts/split/benchmark-tensor-core-prefix.py"}
    try:
        baseline = run_case("baseline", args.baseline_library, args.weights, args.output, args.trials)
        candidate = run_case("candidate", args.candidate_library, args.weights, args.output, args.trials)
        fp32 = run_case("candidate-fp32", args.candidate_library, args.weights, args.output, args.trials, 5)
        result.update({"baseline": baseline, "candidate": candidate, "candidateFp32": fp32})
        mismatch = first_trace_difference((args.output / "baseline-trace.bin").read_bytes(),
                                          (args.output / "candidate-trace.bin").read_bytes())
        result["firstTraceDifference"] = mismatch
        result["samePrecisionEqual"] = (baseline["hiddenSha256"] == candidate["hiddenSha256"] and
                                         baseline["stateSha256"] == candidate["stateSha256"] and
                                         mismatch is None)
        result["candidateComputeTimeChangePercent"] = 100 * (candidate["medianSeconds"] /
                                                            baseline["medianSeconds"] - 1)
        bf16_trace = (args.output / "candidate-trace.bin").read_bytes()
        fp32_trace = (args.output / "candidate-fp32-trace.bin").read_bytes()
        result["crossPrecisionDiagnostic"] = {
            "firstTraceDifference": first_trace_difference(bf16_trace, fp32_trace),
            **trace_error(bf16_trace, fp32_trace),
            "candidateBf16VsFp32TimeChangePercent": 100 * (candidate["medianSeconds"] /
                                                         fp32["medianSeconds"] - 1),
            "interpretation": "BF16-vs-FP32 differences are expected and are not a same-precision correctness gate"}
        result["passed"] = result["samePrecisionEqual"]
    except Exception as exc:
        result["passed"] = False
        result["error"] = str(exc)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("baseline", "candidate", "candidateFp32")}, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
