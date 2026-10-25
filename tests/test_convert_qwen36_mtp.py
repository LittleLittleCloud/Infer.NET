import hashlib
import importlib.util
import json
import os
import pathlib
import struct
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("convert_mtp", os.path.join(HERE, "..", "scripts", "convert-qwen36-mtp.py"))
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)

GEO = conv.geometry(hidden=8, heads=2, kv_heads=1, head_dim=4, experts=3, inter=2, shared=2)


def bf16(values):
    return b"".join(struct.pack("<H", v & 0xFFFF) for v in values)


def tensor_bytes(name, shape, kind):
    n = conv.numel(shape)
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    if kind == "norm":
        # 0x3B80 = 0.00390625: raw+1 is exactly representable in FP32 but not in BF16.
        return bf16([0x3B80 + (i % 5) * 0x40 for i in range(n)])
    return bf16([0x3F00 + ((seed + i * 7) % 0x100) for i in range(n)])


def write_shard(path, names, spec, dtype_override=None, shape_override=None):
    entries = []
    for name in names:
        shape, kind = spec[name]
        data = tensor_bytes(name, shape, kind)
        dtype = (dtype_override or {}).get(name, "BF16")
        entries.append((name, dtype, (shape_override or {}).get(name, shape), data))
    conv.write_safetensors(path, entries)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.spec = conv.source_spec(GEO)
        names = sorted(self.spec)
        self.a = os.path.join(self.dir, "model-00025-of-00026.safetensors")
        self.b = os.path.join(self.dir, "model-00026-of-00026.safetensors")
        write_shard(self.a, [n for n in names if "gate_up_proj" in n], self.spec)
        write_shard(self.b, [n for n in names if "gate_up_proj" not in n], self.spec)
        self.expected = {os.path.basename(p): (os.path.getsize(p), conv.sha256_file(p)) for p in (self.a, self.b)}

    def tearDown(self):
        self.tmp.cleanup()

    def run_convert(self, out, **kw):
        return conv.convert([self.a, self.b], os.path.join(self.dir, out), geo=GEO, expected_shards=self.expected, **kw)

    def read_sidecar(self, path):
        header, start = conv.read_header(path)
        data = pathlib.Path(path).read_bytes()
        return {n: (i["dtype"], i["shape"], data[start + i["data_offsets"][0]: start + i["data_offsets"][1]])
                for n, i in header.items()}

    def test_full_sources_contain_19_tensors(self):
        full = conv.source_spec(conv.geometry())
        self.assertEqual(len(full), 19)
        self.assertEqual(sum(conv.numel(s) for s, _ in full.values()), 844640768)

    def test_matrices_are_preserved_bit_for_bit_and_gate_up_split(self):
        manifest, sidecar, _ = self.run_convert("out")
        tensors = self.read_sidecar(sidecar)
        for name, (shape, kind) in self.spec.items():
            if kind == "matrix":
                dtype, out_shape, data = tensors[name]
                self.assertEqual((dtype, out_shape, data), ("BF16", shape, tensor_bytes(name, shape, kind)))
        e, two_i, h = self.spec[conv.L + "mlp.experts.gate_up_proj"][0]
        fused = tensor_bytes(conv.L + "mlp.experts.gate_up_proj", [e, two_i, h], "gate_up")
        row = h * 2
        for expert in range(e):
            base = expert * two_i * row
            gate = tensors[conv.L + "mlp.experts.gate_proj.weight"][2]
            up = tensors[conv.L + "mlp.experts.up_proj.weight"][2]
            half = (two_i // 2) * row
            self.assertEqual(gate[expert * half:(expert + 1) * half], fused[base:base + half])
            self.assertEqual(up[expert * half:(expert + 1) * half], fused[base + half:base + 2 * half])
        self.assertEqual(tensors[conv.L + "mlp.experts.gate_proj.weight"][1], [e, two_i // 2, h])

    def test_all_seven_norms_are_fp32_raw_plus_one_without_second_rounding(self):
        _, sidecar, _ = self.run_convert("out")
        tensors = self.read_sidecar(sidecar)
        norms = [n for n, (_, k) in self.spec.items() if k == "norm"]
        self.assertEqual(len(norms), 7)
        for name in norms:
            dtype, shape, data = tensors[conv.sidecar_name(name, "norm")]
            self.assertEqual(dtype, "F32")
            raw = tensor_bytes(name, self.spec[name][0], "norm")
            for i in range(len(data) // 4):
                (bits,) = struct.unpack_from("<H", raw, 2 * i)
                expected = struct.unpack("<f", struct.pack("<I", bits << 16))[0] + 1.0
                (actual,) = struct.unpack_from("<f", data, 4 * i)
                self.assertEqual(actual, expected)
            # BF16 rounding of 1.00390625 would lose the +0.00390625 term.
            (first,) = struct.unpack_from("<f", data, 0)
            self.assertEqual(first, 1.00390625)
            self.assertNotIn(conv.sidecar_name(name, "norm").replace(".scale_fp32", ".weight"), tensors)

    def test_output_is_deterministic_and_manifest_hashes_match(self):
        m1, s1, _ = self.run_convert("one")
        m2, s2, _ = self.run_convert("two")
        self.assertEqual(pathlib.Path(s1).read_bytes(), pathlib.Path(s2).read_bytes())
        self.assertEqual(m1["sidecar"]["sha256"], conv.sha256_file(s1))
        self.assertEqual(m1["converterSha256"], conv.sha256_file(conv.__file__))
        self.assertTrue(all(r["verified"] for r in m1["source"]["shards"]))
        self.assertEqual(m1["source"]["revision"], conv.SOURCE_REVISION)
        tensors = self.read_sidecar(s1)
        for entry in m1["tensors"]:
            self.assertEqual(hashlib.sha256(tensors[entry["name"]][2]).hexdigest(), entry["sha256"])

    def test_rejects_wrong_shard_hash(self):
        bad = dict(self.expected)
        name = os.path.basename(self.a)
        bad[name] = (bad[name][0], "0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            conv.convert([self.a, self.b], os.path.join(self.dir, "x"), geo=GEO, expected_shards=bad)

    def test_rejects_single_shard_missing_tensors_shape_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "Both official shards"):
            conv.convert([self.b], os.path.join(self.dir, "x"), geo=GEO, expected_shards=self.expected)
        with self.assertRaisesRegex(ValueError, "Missing MTP tensors"):
            conv.convert([self.b], os.path.join(self.dir, "x"), geo=GEO, verify_shards=False)
        wrong = os.path.join(self.dir, "wrong.safetensors")
        name = conv.L + "self_attn.q_norm.weight"
        write_shard(wrong, [name], self.spec, shape_override={name: [GEO["headDim"] + 1]})
        with self.assertRaises(ValueError):
            conv.locate([wrong, self.a, self.b], self.spec)
        write_shard(wrong, [name], self.spec, dtype_override={name: "F32"})
        with self.assertRaises(ValueError):
            conv.locate([wrong, self.a, self.b], self.spec)

    def test_rejects_non_finite_norm(self):
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            conv.norm_scales_fp32(bf16([0x7F80]))


if __name__ == "__main__":
    unittest.main()
