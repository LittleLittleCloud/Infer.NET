#!/usr/bin/env python3
"""Compare selected experts in CUDA mode 5 and mode 6 synthetic prefix traces.

The output is diagnostic: mode 5 is an internal FP32 path, not the fixed
independent reference, and route equality cannot establish output parity.
"""
import argparse
import array
import hashlib
import json
import math
import pathlib
import statistics
import sys


def load(directory, name, expected_sha, tokens):
    path = directory / f"{name}-routes.i32"
    raw = path.read_bytes()
    if len(raw) != tokens * 20 * 8 * 4:
        raise ValueError(f"Route trace size mismatch: {path}")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"Route trace SHA mismatch: {path}")
    values = array.array("i")
    if values.itemsize != 4:
        raise ValueError("Expected 32-bit int array")
    values.frombytes(raw)
    return values


def activation_impact(directory, result, first_layer_differences):
    traces = {}
    for name in ("baseline", "candidate"):
        path = directory / f"{name}-trace.f32"
        raw = path.read_bytes()
        if len(raw) != 128 * 20 * 2048 * 4 or hashlib.sha256(raw).hexdigest() != result[name]["traceSha256"]:
            raise ValueError(f"Layer trace size or SHA mismatch: {path}")
        values = array.array("f")
        values.frombytes(raw)
        traces[name] = values
    route_kind = {item["token"]: "orderOnly" if item["sameSet"] else "expertSetChanged"
                  for item in first_layer_differences}
    groups = {key: [] for key in ("sameOrderedExperts", "orderOnly", "expertSetChanged")}
    for token in range(128):
        start = token * 20 * 2048
        delta2 = ref2 = 0.0
        for i in range(start, start + 2048):
            baseline = float(traces["baseline"][i])
            candidate = float(traces["candidate"][i])
            if not math.isfinite(baseline) or not math.isfinite(candidate):
                raise ValueError("Nonfinite layer-0 activation")
            delta2 += (candidate - baseline) ** 2
            ref2 += baseline ** 2
        groups[route_kind.get(token, "sameOrderedExperts")].append((token, delta2, ref2))
    total_delta2 = sum(item[1] for group in groups.values() for item in group)
    total_ref2 = sum(item[2] for group in groups.values() for item in group)
    calculated = (total_delta2 / max(total_ref2, 1e-30)) ** .5
    if abs(calculated - result["layerDrift"][0]["relativeL2"]) > 1e-10:
        raise ValueError("Layer-0 trace gap disagrees with saved diagnostic")
    answer = {}
    for key, group in groups.items():
        answer[key] = {
            "tokenCount": len(group),
            "aggregateRelativeL2": (sum(item[1] for item in group) /
                                    max(sum(item[2] for item in group), 1e-30)) ** .5 if group else None,
            "medianTokenRelativeL2": statistics.median((item[1] / max(item[2], 1e-30)) ** .5 for item in group) if group else None,
            "layer0SquaredErrorShare": sum(item[1] for item in group) / max(total_delta2, 1e-30),
        }
        if key == "expertSetChanged":
            answer[key]["tokenIds"] = [item[0] for item in group]
    return answer


def main():
    if sys.byteorder != "little":
        raise SystemExit("Route traces require little-endian integers")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--include-activation", action="store_true", help="Correlate layer-0 trace errors with route changes")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    result = json.loads((args.cuda / "result.json").read_text())
    tokens = result["tokens"]
    if (tokens, result["baseline"]["mode"], result["candidate"]["mode"], result["kv"]) != (128, 5, 6, "bf16"):
        raise ValueError("Expected 128-token CUDA mode-5/mode-6 BF16 diagnostic")
    traces = {name: load(args.cuda, name, result[name]["routesSha256"], tokens)
              for name in ("baseline", "candidate", "repeat")}
    if traces["candidate"] != traces["repeat"]:
        raise ValueError("Candidate expert routes were not deterministic")
    layer_counts = [0] * 20
    layer_set_counts = [0] * 20
    different = []
    chunk_size = 32
    for token in range(tokens):
        chunk, local = divmod(token, chunk_size)
        count = min(chunk_size, tokens - chunk * chunk_size)
        for layer in range(20):
            offset = (chunk * 20 * chunk_size + layer * count + local) * 8
            baseline = list(traces["baseline"][offset:offset + 8])
            candidate = list(traces["candidate"][offset:offset + 8])
            if len(baseline) != 8 or len(candidate) != 8:
                raise ValueError("Truncated route entry")
            for selection in (baseline, candidate):
                if len(set(selection)) != 8 or any(x < 0 or x >= 256 for x in selection):
                    raise ValueError(f"Invalid top-eight expert IDs at token {token}, layer {layer}")
            if baseline != candidate:
                layer_counts[layer] += 1
                same_set = set(baseline) == set(candidate)
                if not same_set:
                    layer_set_counts[layer] += 1
                different.append({"token": token, "layer": layer,
                                  "baseline": baseline, "candidate": candidate,
                                  "sameSet": same_set})
    summary = {
        "kind": "isolated-cuda-bf16-versus-fp32-expert-route-gap",
        "tokens": tokens,
        "layers": 20,
        "comparedDecisions": tokens * 20,
        "differentDecisions": len(different),
        "differentExpertSets": sum(not item["sameSet"] for item in different),
        "differentByLayer": layer_counts,
        "differentExpertSetsByLayer": layer_set_counts,
        "firstDifferences": different[:20],
        "firstLayerDifferences": [item for item in different if item["layer"] == 0],
        "baselineRoutesSha256": result["baseline"]["routesSha256"],
        "candidateRoutesSha256": result["candidate"]["routesSha256"],
        "limitations": ["Synthetic token IDs", "CUDA mode 5 is an internal baseline, not an independent FP32 reference",
                        "Selected IDs alone do not measure router margins or establish full-model correctness"]}
    if args.include_activation:
        summary["layer0ActivationByRouteKind"] = activation_impact(args.cuda, result, summary["firstLayerDifferences"])
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("comparedDecisions", "differentDecisions", "differentExpertSets", "differentByLayer")}))


if __name__ == "__main__":
    main()
