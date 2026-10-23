#!/usr/bin/env python3
"""Compare first-layer BF16 stage probes with a high-precision CPU reference.

Diagnostic only: NumPy is needed to analyze the pinned quantized safetensors;
it is not an Infer.NET runtime dependency. GPU readbacks invalidate timing.
"""
import argparse
import hashlib
import json
import pathlib
import struct

import numpy as np

LAYER = "language_model.model.layers.0."
QKV = LAYER + "linear_attn.in_proj_qkv."
EXPERT = LAYER + "mlp.switch_mlp."
ROUTE_COUNT = 8
MODEL_REVISION = "e06a74e6236a60c8367e1a3214e83d8b61b637b0"


class Shards:
    def __init__(self, model):
        self.model = model
        self.index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def tensor(self, key):
        shard_name = self.index[key]
        shard = self.model / shard_name
        if shard_name not in self.headers:
            with shard.open("rb") as source:
                header_bytes = struct.unpack("<Q", source.read(8))[0]
                header = json.loads(source.read(header_bytes))
            self.headers[shard_name] = (8 + header_bytes, header)
        base, header = self.headers[shard_name]
        entry = header[key]
        dtype = {"U32": "<u4", "BF16": "<u2"}[entry["dtype"]]
        return np.memmap(shard, mode="r", dtype=dtype,
                         offset=base + entry["data_offsets"][0],
                         shape=tuple(entry["shape"]))


def bf16_to_f32(raw):
    return (raw.astype(np.uint32) << 16).view(np.float32)


def round_bf16(values):
    raw = np.asarray(values, dtype=np.float32).view(np.uint32)
    return ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def stage(directory, host, name, count):
    path = directory / f"{host}-{name}.f32"
    data = np.fromfile(path, dtype="<f4")
    if data.size != count or not np.isfinite(data).all():
        raise ValueError(f"Invalid {host} {name} stage: {path}")
    return data


def difference(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = a - b
    return {"relativeL2": float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)),
            "maxAbs": float(np.max(np.abs(delta))),
            "p99Abs": float(np.quantile(np.abs(delta), 0.99))}


def dequant_bf16(packed, scales, biases):
    values = packed.astype(np.float32) * np.repeat(scales, 64, axis=-1)
    values += np.repeat(biases, 64, axis=-1)
    return round_bf16(values)


def verify_bundle_weights(bundle, tensors, routes):
    """Check exported QKV and routed expert gate/up bytes against the model."""
    with bundle.open("rb") as source:
        if source.read(8) != b"DIPFX001":
            raise ValueError("Unexpected prefix bundle format")
        source.read(48)  # pinned revision and epsilon/theta
        for matrix_index in range(-1, 15):  # embedding followed by layer-0 matrices
            experts, rows, cols = struct.unpack("<III", source.read(12))
            if matrix_index == 0 and (experts, rows, cols) != (1, 8192, 2048):
                raise ValueError("Unexpected QKV dimensions in prefix bundle")
            if matrix_index in (8, 9) and (experts, rows, cols) != (256, 512, 2048):
                raise ValueError("Unexpected expert dimensions in prefix bundle")
            key = {0: QKV, 8: EXPERT + "gate_proj.",
                   9: EXPERT + "up_proj."}.get(matrix_index)
            for suffix, bytes_per_expert in (("weight", rows * cols),
                                             ("scales", rows * cols // 32),
                                             ("biases", rows * cols // 32)):
                section_start = source.tell()
                if key:
                    expected = tensors.tensor(key + suffix)
                    selected = [0] if matrix_index == 0 else routes
                    for expert in selected:
                        source.seek(section_start + expert * bytes_per_expert)
                        model_bytes = expected.tobytes() if matrix_index == 0 else expected[expert].tobytes()
                        if source.read(bytes_per_expert) != model_bytes:
                            raise ValueError(f"Prefix bundle matrix {matrix_index} expert {expert} {suffix} differs")
                source.seek(section_start + experts * bytes_per_expert)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--metal", required=True, type=pathlib.Path)
    parser.add_argument("--cuda", required=True, type=pathlib.Path)
    parser.add_argument("--bundle", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Use a fresh output file")
    profile = json.loads((args.model / "config.json").read_text())["quantization"]
    if (profile["bits"], profile["group_size"], profile["mode"]) != (8, 64, "affine"):
        raise ValueError("Expected affine8/group64 model")
    if MODEL_REVISION not in (args.model / "provenance.json").read_text():
        raise ValueError("Unexpected model revision")
    metal_profile = json.loads((args.metal / "result.json").read_text())
    cuda_profile = json.loads((args.cuda / "result.json").read_text())
    if (metal_profile["tokens"], cuda_profile["tokens"],
            metal_profile["weightBits"], cuda_profile["weightBits"],
            metal_profile["kv"], cuda_profile["kv"]) != (128, 128, 8, 8, "bf16", "bf16"):
        raise ValueError("Stage probes are not the same 128-token affine8 BF16 profile")
    token_ids = [100 + (i * 137) % 247000 for i in range(128)]
    if (json.loads((args.metal / "tokens.json").read_text()) != token_ids or
            cuda_profile["tokensSha256"] != hashlib.sha256(json.dumps(token_ids).encode()).hexdigest() or
            metal_profile["routerProjection"] != cuda_profile["candidateRouterProjection"] or
            cuda_profile["candidateSharedProjection"] != "fp32-diagnostic"):
        raise ValueError("Probe tokens or router/shared projection profiles differ")
    tensors = Shards(args.model)

    metal_input = stage(args.metal, "metal", "embedding-norm", 2048)
    cuda_input = stage(args.cuda, "cuda", "embedding-norm", 2048)
    if metal_input.tobytes() != cuda_input.tobytes():
        raise ValueError("QKV inputs are not bit-identical")
    x = round_bf16(metal_input).astype(np.float64)
    packed = tensors.tensor(QKV + "weight").view(np.uint8).reshape(8192, 2048)
    scales = bf16_to_f32(tensors.tensor(QKV + "scales"))
    biases = bf16_to_f32(tensors.tensor(QKV + "biases"))
    qkv_reference = np.empty(8192, np.float64)
    qkv_weight_round_differences = 0
    for begin in range(0, 8192, 256):
        end = begin + 256
        weights = dequant_bf16(packed[begin:end], scales[begin:end], biases[begin:end])
        qkv_reference[begin:end] = weights.astype(np.float64) @ x
        # Simulate fused dequantization in float64, rounded once to float32.
        fused = (packed[begin:end].astype(np.float64) *
                 np.repeat(scales[begin:end].astype(np.float64), 64, axis=1) +
                 np.repeat(biases[begin:end].astype(np.float64), 64, axis=1)).astype(np.float32)
        qkv_weight_round_differences += int(np.count_nonzero(round_bf16(fused) != weights))

    qkv_metal = stage(args.metal, "metal", "qkv", 8192)
    qkv_cuda = stage(args.cuda, "cuda", "qkv", 8192)
    router_metal = stage(args.metal, "metal", "router", 256)
    router_cuda = stage(args.cuda, "cuda", "router", 256)
    routes_metal = np.argsort(-router_metal, kind="stable")[:ROUTE_COUNT]
    routes_cuda = np.argsort(-router_cuda, kind="stable")[:ROUTE_COUNT]
    if not np.array_equal(routes_metal, routes_cuda):
        raise ValueError("Token-zero routes differ; this diagnostic assumes identical routes")
    routes = [int(route) for route in routes_metal]
    if args.bundle:
        verify_bundle_weights(args.bundle, tensors, routes)

    input_metal = round_bf16(stage(args.metal, "metal", "residual-norm", 2048)).astype(np.float64)
    input_cuda = round_bf16(stage(args.cuda, "cuda", "residual-norm", 2048)).astype(np.float64)
    weights = {kind: tensors.tensor(EXPERT + kind + ".weight").view(np.uint8).reshape(256, 512, 2048)
               for kind in ("gate_proj", "up_proj")}
    expert_scales = {kind: bf16_to_f32(tensors.tensor(EXPERT + kind + ".scales"))
                     for kind in weights}
    expert_biases = {kind: bf16_to_f32(tensors.tensor(EXPERT + kind + ".biases"))
                     for kind in weights}

    def projection(kind, expert, operand):
        w = dequant_bf16(weights[kind][expert],
                         expert_scales[kind][expert], expert_biases[kind][expert])
        return w.astype(np.float64) @ operand

    expected_metal = np.empty(4096, np.float64)
    expected_cuda = np.empty(4096, np.float64)
    for index, expert in enumerate(routes):
        for operand, output in ((input_metal, expected_metal), (input_cuda, expected_cuda)):
            gate = projection("gate_proj", expert, operand)
            up = projection("up_proj", expert, operand)
            output[index * 512:(index + 1) * 512] = gate / (1 + np.exp(-gate)) * up

    expert_metal = stage(args.metal, "metal", "expert-gate", 4096)
    expert_cuda = stage(args.cuda, "cuda", "expert-gate", 4096)
    observed_delta = expert_metal.astype(np.float64) - expert_cuda.astype(np.float64)
    predicted_delta = expected_metal - expected_cuda
    result = {
        "kind": "isolated-token-zero-layer-zero-bf16-cpu-reference",
        "modelQuantization": "affine8/group64",
        "modelRevision": MODEL_REVISION,
        "bundleQkvAndSelectedExpertsMatchModel": bool(args.bundle),
        "qkv": {
            "inputBitIdentical": True,
            "fusedVsSeparateBf16WeightOperandDifferences": qkv_weight_round_differences,
            "metalVsReference": difference(qkv_metal, qkv_reference),
            "cudaVsReference": difference(qkv_cuda, qkv_reference),
            "metalVsCuda": difference(qkv_metal, qkv_cuda)},
        "expertGate": {
            "routes": routes,
            "bf16InputOperandDifferences": int(np.count_nonzero(input_metal != input_cuda)),
            "metalVsOwnInputReference": difference(expert_metal, expected_metal),
            "cudaVsOwnInputReference": difference(expert_cuda, expected_cuda),
            "predictedInputDrift": difference(expected_metal, expected_cuda),
            "observedDrift": difference(expert_metal, expert_cuda),
            "unexplainedDrift": difference(observed_delta, predicted_delta)},
        "limitations": ["One synthetic token and one layer", "High-precision CPU dot products are diagnostic references",
                        "Does not prove full-model parity, output quality, throughput, or memory safety"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "qkv": result["qkv"],
                      "expertGate": result["expertGate"]}))


if __name__ == "__main__":
    main()
