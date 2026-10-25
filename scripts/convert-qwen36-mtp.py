#!/usr/bin/env python3
"""Convert the official Qwen3.6-35B-A3B MTP tensors into the Infer.NET MTP sidecar.

Experiment official-mtp-20261024 (Issue #76). Standard library only, no tensor math
beyond one exact FP32 addition for the norm scales.

Inputs: the two pinned official BF16 shards (25 and 26). Both complete-file SHA256 values
and byte sizes are verified before any tensor is read. Output: one deterministic
safetensors sidecar plus a JSON manifest.

  * Matrices stay BF16 bit-for-bit. The fused experts.gate_up_proj [E,2*I,H] is split into
    gate_proj and up_proj [E,I,H] by byte slicing (gate = first I rows, up = last I rows).
  * The seven RMSNorm tensors become FP32 scales raw_bf16.float() + 1 with a single FP32
    rounding (the official MTP norms use the (1 + weight) convention, like the target
    checkpoint conversion), including q_norm/k_norm. No second BF16 rounding.
  * No quantization, no FP16 staging, no FP32 matrix expansion.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

CONVERTER_VERSION = 1
FORMAT = "deepinfer-qwen36-mtp-sidecar"
FORMAT_VERSION = 1
SOURCE_REPOSITORY = "Qwen/Qwen3.6-35B-A3B"
SOURCE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TARGET_REPOSITORY = "mlx-community/Qwen3.6-35B-A3B-8bit"
TARGET_REVISION = "e06a74e6236a60c8367e1a3214e83d8b61b637b0"

OFFICIAL_SHARDS = {
    "model-00025-of-00026.safetensors": (
        3832888256, "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154"),
    "model-00026-of-00026.safetensors": (
        2231416848, "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613"),
}

L = "mtp.layers.0."


def geometry(hidden=2048, heads=16, kv_heads=2, head_dim=256, experts=256, inter=512, shared=512):
    q = heads * head_dim * 2  # query + output gate per head
    kv = kv_heads * head_dim
    return {
        "hidden": hidden, "heads": heads, "kvHeads": kv_heads, "headDim": head_dim,
        "experts": experts, "expertIntermediate": inter, "sharedIntermediate": shared,
        "topK": 8, "ropeDims": 64, "ropeTheta": 10000000.0, "layers": 1,
        "qProjRows": q, "kvRows": kv, "oProjColumns": heads * head_dim,
    }


def source_spec(geo):
    h, e, i, s = geo["hidden"], geo["experts"], geo["expertIntermediate"], geo["sharedIntermediate"]
    return {  # source tensor name -> (shape, kind)
        "mtp.fc.weight": ([h, 2 * h], "matrix"),
        "mtp.pre_fc_norm_embedding.weight": ([h], "norm"),
        "mtp.pre_fc_norm_hidden.weight": ([h], "norm"),
        "mtp.norm.weight": ([h], "norm"),
        L + "input_layernorm.weight": ([h], "norm"),
        L + "post_attention_layernorm.weight": ([h], "norm"),
        L + "self_attn.q_proj.weight": ([geo["qProjRows"], h], "matrix"),
        L + "self_attn.k_proj.weight": ([geo["kvRows"], h], "matrix"),
        L + "self_attn.v_proj.weight": ([geo["kvRows"], h], "matrix"),
        L + "self_attn.o_proj.weight": ([h, geo["oProjColumns"]], "matrix"),
        L + "self_attn.q_norm.weight": ([geo["headDim"]], "norm"),
        L + "self_attn.k_norm.weight": ([geo["headDim"]], "norm"),
        L + "mlp.gate.weight": ([e, h], "matrix"),
        L + "mlp.experts.gate_up_proj": ([e, 2 * i, h], "gate_up"),
        L + "mlp.experts.down_proj": ([e, h, i], "matrix"),
        L + "mlp.shared_expert.gate_proj.weight": ([s, h], "matrix"),
        L + "mlp.shared_expert.up_proj.weight": ([s, h], "matrix"),
        L + "mlp.shared_expert.down_proj.weight": ([h, s], "matrix"),
        L + "mlp.shared_expert_gate.weight": ([1, h], "matrix"),
    }


def sidecar_name(source_name, kind, part=None):
    if kind == "norm":
        return source_name[: -len(".weight")] + ".scale_fp32"
    if kind == "gate_up":
        return L + "mlp.experts." + part + "_proj.weight"
    return source_name


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length < 2 or length > 16 * 1024 * 1024:
            raise ValueError("Invalid safetensors header length in " + path)
        header = json.loads(f.read(length))
    return header, 8 + length


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def norm_scales_fp32(raw_bf16):
    """Exact raw.float()+1 with one FP32 rounding; BF16->FP32 is a 16-bit shift."""
    count = len(raw_bf16) // 2
    bits = struct.unpack("<%dH" % count, raw_bf16)
    out = bytearray()
    for b in bits:
        value = struct.unpack("<f", struct.pack("<I", b << 16))[0]
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Non-finite BF16 norm weight")
        out += struct.pack("<f", value + 1.0)  # double add is exact for two FP32 values, pack rounds once
    return bytes(out)


def locate(shards, spec):
    """Find every tensor in exactly one shard and validate dtype/shape."""
    headers = {}
    found = {}
    for path in shards:
        header, data_start = read_header(path)
        headers[path] = (header, data_start)
        for name, info in header.items():
            if name in spec:
                if name in found:
                    raise ValueError("Tensor present in more than one shard: " + name)
                found[name] = (path, info, data_start)
    missing = sorted(set(spec) - set(found))
    if missing:
        raise ValueError("Missing MTP tensors: " + ", ".join(missing))
    for name, (path, info, data_start) in found.items():
        shape, _ = spec[name]
        if info["dtype"] != "BF16":
            raise ValueError("%s dtype %s is not BF16" % (name, info["dtype"]))
        if list(info["shape"]) != shape:
            raise ValueError("%s shape %s != %s" % (name, info["shape"], shape))
        begin, end = info["data_offsets"]
        if end - begin != numel(shape) * 2:
            raise ValueError("%s byte length mismatch" % name)
    return found


def read_tensor(path, data_start, info):
    begin, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + begin)
        data = f.read(end - begin)
    if len(data) != end - begin:
        raise ValueError("Truncated tensor in " + path)
    return data


def split_gate_up(data, experts, inter, hidden):
    row = hidden * 2
    gate = bytearray()
    up = bytearray()
    stride = 2 * inter * row
    for e in range(experts):
        base = e * stride
        gate += data[base: base + inter * row]
        up += data[base + inter * row: base + 2 * inter * row]
    return bytes(gate), bytes(up)


def write_safetensors(path, entries):
    """entries: list of (name, dtype, shape, bytes). Deterministic: sorted names, no metadata."""
    entries = sorted(entries, key=lambda x: x[0])
    header = {}
    offset = 0
    for name, dtype, shape, data in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in entries:
            f.write(data)


def convert(shards, output_dir, geo=None, verify_shards=True, expected_shards=None):
    geo = geo or geometry()
    spec = source_spec(geo)
    expected = OFFICIAL_SHARDS if expected_shards is None else expected_shards
    shard_records = []
    for path in shards:
        name = os.path.basename(path)
        record = {"file": name, "bytes": os.path.getsize(path)}
        if verify_shards:
            if name not in expected:
                raise ValueError("Unexpected shard file name " + name)
            size, digest = expected[name]
            if record["bytes"] != size:
                raise ValueError("%s size %d != pinned %d" % (name, record["bytes"], size))
            record["sha256"] = sha256_file(path)
            if record["sha256"] != digest:
                raise ValueError("%s SHA256 %s != pinned %s" % (name, record["sha256"], digest))
            record["verified"] = True
        else:
            record["sha256"] = sha256_file(path)
            record["verified"] = False
        shard_records.append(record)
    if verify_shards and {r["file"] for r in shard_records} != set(expected):
        raise ValueError("Both official shards 25 and 26 are required")

    found = locate(shards, spec)
    entries = []
    tensors = []
    parameters = 0
    for name in sorted(spec):
        shape, kind = spec[name]
        path, info, data_start = found[name]
        data = read_tensor(path, data_start, info)
        source_hash = hashlib.sha256(data).hexdigest()
        parameters += numel(shape)
        outputs = []
        if kind == "matrix":
            outputs.append((sidecar_name(name, kind), "BF16", shape, data, "copy"))
        elif kind == "norm":
            outputs.append((sidecar_name(name, kind), "F32", shape, norm_scales_fp32(data), "raw_bf16_float_plus_one_fp32"))
        else:
            e, two_i, h = shape
            gate, up = split_gate_up(data, e, two_i // 2, h)
            part_shape = [e, two_i // 2, h]
            outputs.append((sidecar_name(name, kind, "gate"), "BF16", part_shape, gate, "gate_up_first_half"))
            outputs.append((sidecar_name(name, kind, "up"), "BF16", part_shape, up, "gate_up_second_half"))
        for out_name, dtype, out_shape, out_data, transform in outputs:
            entries.append((out_name, dtype, out_shape, out_data))
            tensors.append({
                "name": out_name, "dtype": dtype, "shape": out_shape, "bytes": len(out_data),
                "sha256": hashlib.sha256(out_data).hexdigest(), "transform": transform,
                "sourceName": name, "sourceShard": os.path.basename(path), "sourceShape": shape,
                "sourceSha256": source_hash,
            })
        del data

    os.makedirs(output_dir, exist_ok=True)
    sidecar_path = os.path.join(output_dir, "qwen36-mtp-sidecar.safetensors")
    write_safetensors(sidecar_path, entries)
    manifest = {
        "format": FORMAT, "formatVersion": FORMAT_VERSION, "converterVersion": CONVERTER_VERSION,
        "converterSha256": sha256_file(os.path.abspath(__file__)),
        "experiment": "official-mtp-20261024",
        "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "shards": shard_records,
                   "sourceParameters": parameters},
        "target": {"repository": TARGET_REPOSITORY, "revision": TARGET_REVISION,
                   "sharedEmbeddingAndHead": True},
        "geometry": geo,
        "precision": {"matrices": "BF16", "normScales": "FP32 raw+1", "activations": "FP32", "kv": "FP32"},
        "tensors": sorted(tensors, key=lambda t: t["name"]),
        "sidecar": {"file": os.path.basename(sidecar_path), "bytes": os.path.getsize(sidecar_path),
                    "sha256": sha256_file(sidecar_path)},
    }
    manifest_path = os.path.join(output_dir, "qwen36-mtp-sidecar.manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest, sidecar_path, manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", required=True,
                        help="Directory containing the two verified official shards 25 and 26")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    shards = [os.path.join(args.source_dir, n) for n in sorted(OFFICIAL_SHARDS)]
    manifest, sidecar, path = convert(shards, args.output_dir)
    print(json.dumps({"sidecar": sidecar, "manifest": path, "sidecarSha256": manifest["sidecar"]["sha256"],
                      "converterSha256": manifest["converterSha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
