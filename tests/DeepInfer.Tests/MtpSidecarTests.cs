using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DeepInfer.Backends.Models;
using Xunit;
namespace DeepInfer.Tests;

// Writes a tiny-geometry sidecar in the converter's on-disk layout so the loader and CPU oracle are testable without 1.7 GB of weights.
internal static class TinySidecar
{
    public static readonly Qwen36MtpGeometry Geometry = new(16, 2, 1, 8, 4, 8, 8, 2, 4, 1e4f, 1e-6f);
    public static ushort Bf16(float v) => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);

    public static string Write(string directory, Func<string, float, string?>? mutate = null, bool tamperHash = false, Qwen36MtpGeometry? manifestGeometry = null, bool dropLast = false, Action<System.Text.Json.Nodes.JsonNode>? edit = null)
    {
        Directory.CreateDirectory(directory);
        var layout = Qwen36MtpNames.Layout(Geometry);
        if (dropLast) layout = layout[..^1];
        var random = new Random(7); var header = new Dictionary<string, object>(); var data = new MemoryStream(); var entries = new List<object>();
        foreach (var (name, shape) in layout)
        {
            var isNorm = Array.IndexOf(Qwen36MtpNames.Norms, name) >= 0;
            var count = shape.Aggregate(1, (a, b) => a * b); var begin = data.Position;
            for (var i = 0; i < count; i++)
            {
                var v = isNorm ? 1f + (float)(random.NextDouble() - 0.5) * 0.2f : (float)(random.NextDouble() - 0.5) * 0.4f;
                if (isNorm) data.Write(BitConverter.GetBytes(v)); else data.Write(BitConverter.GetBytes(Bf16(v)));
            }
            var dtype = mutate?.Invoke(name, 0) ?? (isNorm ? "F32" : "BF16");
            header[name] = new { dtype, shape, data_offsets = new[] { begin, data.Position } };
            var transform = isNorm ? "raw_bf16_float_plus_one_fp32" : name == Qwen36MtpNames.ExpertGate ? "gate_up_first_half" : name == Qwen36MtpNames.ExpertUp ? "gate_up_second_half" : "copy";
            var payload = data.GetBuffer().AsSpan((int)begin, (int)(data.Position - begin));
            var payloadHash = Convert.ToHexStringLower(SHA256.HashData(payload));
            var fused = name == Qwen36MtpNames.ExpertGate || name == Qwen36MtpNames.ExpertUp;
            var sourceName = isNorm ? name[..^".scale_fp32".Length] + ".weight" : fused ? "mtp.layers.0.mlp.experts.gate_up_proj" : name;
            var sourceShape = fused ? new[] { shape[0], shape[1] * 2, shape[2] } : shape;
            var sourceHash = transform == "copy" ? payloadHash : Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes("source:" + name)));
            entries.Add(new { name, dtype = isNorm ? "F32" : "BF16", bytes = data.Position - begin, transform, sha256 = payloadHash,
                sourceName, sourceShape, sourceShard = "model-00026-of-00026.safetensors", sourceSha256 = sourceHash });
        }
        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        var path = Path.Combine(directory, "qwen36-mtp-sidecar.safetensors");
        using (var f = File.Create(path)) { f.Write(BitConverter.GetBytes((long)json.Length)); f.Write(json); data.WriteTo(f); }
        var hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        if (tamperHash) hash = new string('0', 64);
        var g = manifestGeometry ?? Geometry;
        var manifest = new
        {
            format = Qwen36MtpPins.Format, formatVersion = Qwen36MtpPins.FormatVersion,
            source = new { repository = "synthetic", revision = "synthetic", sourceParameters = 0L, shards = Array.Empty<object>() },
            target = new { repository = "synthetic", revision = "synthetic", sharedEmbeddingAndHead = true },
            geometry = new { hidden = g.Hidden, heads = g.Heads, kvHeads = g.KvHeads, headDim = g.HeadDim, experts = g.Experts, expertIntermediate = g.ExpertIntermediate,
                sharedIntermediate = g.SharedIntermediate, topK = g.TopK, ropeDims = g.RopeDims, ropeTheta = (double)g.RopeTheta, layers = 1 },
            precision = new { matrices = "BF16", normScales = "FP32 raw+1", activations = "FP32", kv = "FP32" },
            sidecar = new { bytes = new FileInfo(path).Length, sha256 = hash },
            tensors = entries, converterSha256 = new string('a', 64),
        };
        var node = JsonSerializer.SerializeToNode(manifest)!;
        edit?.Invoke(node);
        File.WriteAllText(Path.Combine(directory, "qwen36-mtp-sidecar.manifest.json"), node.ToJsonString());
        return directory;
    }
    public static Qwen36MtpLoadOptions Tiny => new(RequireOfficialPins: false, Geometry: Geometry);
}

public class MtpSidecarTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "deepinfer-mtp-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }

    [Fact]
    public void AcceptsAConsistentSidecarAndExposesExactBf16AndFp32Norms()
    {
        using var w = Qwen36MtpWeights.Load(TinySidecar.Write(root), TinySidecar.Tiny);
        Assert.Equal(64, w.SidecarSha256.Length);
        var fc = w.ReadBf16(Qwen36MtpNames.Fc);
        Assert.Equal(16 * 32, fc.Length);
        Assert.Equal(1f, w.ReadNorm(Qwen36MtpNames.QNorm).Length / 8f);
        Assert.All(w.ReadNorm(Qwen36MtpNames.PreHiddenNorm), v => Assert.InRange(v, 0.85f, 1.15f));
        var range = new float[16]; w.ReadBf16Range(Qwen36MtpNames.Fc, 16, range);
        Assert.Equal(fc.Skip(16).Take(16).Select(b => BitConverter.UInt32BitsToSingle((uint)b << 16)), range);
    }

    [Fact]
    public void OfficialPinsRejectAnyNonOfficialSidecar()
    {
        TinySidecar.Write(root);
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(root, new(RequireOfficialPins: true, Geometry: TinySidecar.Geometry)));
        Assert.ThrowsAny<Exception>(() => Qwen36MtpWeights.Load(root));
    }

    [Fact]
    public void RejectsHashGeometryDtypeAndMissingTensorMismatches()
    {
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(TinySidecar.Write(Path.Combine(root, "hash"), tamperHash: true), TinySidecar.Tiny));
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(TinySidecar.Write(Path.Combine(root, "geometry"), manifestGeometry: TinySidecar.Geometry with { TopK = 3 }), TinySidecar.Tiny));
        Assert.ThrowsAny<Exception>(() => Qwen36MtpWeights.Load(TinySidecar.Write(Path.Combine(root, "dtype"), (n, _) => n == Qwen36MtpNames.Fc ? "F16" : null), TinySidecar.Tiny));
        Assert.ThrowsAny<Exception>(() => Qwen36MtpWeights.Load(TinySidecar.Write(Path.Combine(root, "missing"), dropLast: true), TinySidecar.Tiny));
        Assert.Throws<FileNotFoundException>(() => Qwen36MtpWeights.Load(Path.Combine(root, "nothing"), TinySidecar.Tiny));
    }
}

public class MtpLoaderValidationTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "deepinfer-mtpval-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    private string Case(string name, Action<System.Text.Json.Nodes.JsonNode> edit) => TinySidecar.Write(Path.Combine(root, name), edit: edit);
    private static System.Text.Json.Nodes.JsonNode Tensor(System.Text.Json.Nodes.JsonNode m, string name) =>
        m["tensors"]!.AsArray().First(t => t!["name"]!.GetValue<string>() == name)!;

    [Fact]
    public void RejectsRopeThetaAndActivationOrKvPrecisionMismatches()
    {
        Assert.Contains("rope", Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("rope", m => m["geometry"]!["ropeTheta"] = 1000000.0), TinySidecar.Tiny)).Message);
        Assert.Contains("precision", Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("act", m => m["precision"]!["activations"] = "FP16"), TinySidecar.Tiny)).Message);
        Assert.Contains("precision", Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("kv", m => m["precision"]!["kv"] = "BF16"), TinySidecar.Tiny)).Message);
    }

    [Theory]
    [InlineData("sourceName", "mtp.layers.0.other.weight")]
    [InlineData("sourceShard", "")]
    [InlineData("sourceSha256", "abc")]
    [InlineData("sha256", "xyz")]
    public void RejectsPerTensorProvenanceMismatches(string field, string value)
    {
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case(field, m => Tensor(m, Qwen36MtpNames.Fc)[field] = value), TinySidecar.Tiny));
    }

    [Fact]
    public void RejectsWrongSourceShapeCopyHashDriftAndPayloadHashMismatch()
    {
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("shape", m => Tensor(m, Qwen36MtpNames.ExpertGate)["sourceShape"] = System.Text.Json.Nodes.JsonNode.Parse("[4,8,16]")), TinySidecar.Tiny));
        var other = new string('b', 64);
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("copy", m => Tensor(m, Qwen36MtpNames.Q)["sourceSha256"] = other), TinySidecar.Tiny));
        Assert.Contains("payload SHA256", Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.Load(Case("payload", m => Tensor(m, Qwen36MtpNames.InputNorm)["sha256"] = other), TinySidecar.Tiny)).Message);
    }

    private static (System.Text.Json.JsonElement Source, System.Text.Json.JsonElement Target) Official(Action<System.Text.Json.Nodes.JsonNode>? edit = null)
    {
        var shards = Qwen36MtpPins.Shards.Select(x => new { file = x.Key, bytes = x.Value.Bytes, sha256 = x.Value.Sha256, verified = true }).ToArray();
        var node = System.Text.Json.JsonSerializer.SerializeToNode(new
        {
            source = new { repository = Qwen36MtpPins.SourceRepository, revision = Qwen36MtpPins.SourceRevision, sourceParameters = Qwen36MtpPins.SourceParameters, shards },
            target = new { repository = Qwen36MtpPins.TargetRepository, revision = Qwen36MtpPins.TargetRevision, sharedEmbeddingAndHead = true },
        })!;
        edit?.Invoke(node);
        var element = System.Text.Json.JsonDocument.Parse(node.ToJsonString()).RootElement;
        return (element.GetProperty("source"), element.GetProperty("target"));
    }

    [Fact]
    public void OfficialSourceRequiresExactlyTheTwoUniqueVerifiedShards()
    {
        var ok = Official(); Qwen36MtpWeights.ValidateOfficialSource(ok.Source, ok.Target);
        var duplicate = Official(n => n["source"]!["shards"]![1]!["file"] = "model-00025-of-00026.safetensors");
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(duplicate.Source, duplicate.Target));
        var one = Official(n => n["source"]!["shards"]!.AsArray().RemoveAt(1));
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(one.Source, one.Target));
        var unknown = Official(n => n["source"]!["shards"]![0]!["file"] = "model-00001-of-00026.safetensors");
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(unknown.Source, unknown.Target));
        var badHash = Official(n => n["source"]!["shards"]![0]!["sha256"] = new string('0', 64));
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(badHash.Source, badHash.Target));
        var unverified = Official(n => n["source"]!["shards"]![0]!["verified"] = false);
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(unverified.Source, unverified.Target));
        var wrongTarget = Official(n => n["target"]!["revision"] = "0000");
        Assert.Throws<InvalidDataException>(() => Qwen36MtpWeights.ValidateOfficialSource(wrongTarget.Source, wrongTarget.Target));
    }
}

public class MtpReferenceTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "deepinfer-mtpref-" + Guid.NewGuid().ToString("N"));
    public void Dispose() { if (Directory.Exists(root)) Directory.Delete(root, true); }
    private static float[] Vector(int seed, int n, float scale = 1f) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() - 0.5) * scale).ToArray(); }

    [Fact]
    public void StepsAreFiniteCausalAndCursorDriven()
    {
        using var w = Qwen36MtpWeights.Load(TinySidecar.Write(root), TinySidecar.Tiny);
        var oracle = new Qwen36MtpReference(w);
        Assert.Throws<InvalidOperationException>(() => oracle.Step(Vector(1, 16), Vector(2, 16), 1));
        var y0 = oracle.Step(Vector(1, 16), Vector(2, 16), 0);
        Assert.Equal(1, oracle.ValidRows);
        var y1 = oracle.Step(Vector(3, 16), Vector(4, 16), 1);
        Assert.All(y0.Concat(y1), v => Assert.True(float.IsFinite(v)));
        Assert.Equal(2, oracle.ValidRows);
        Assert.Equal(2, oracle.LastExperts.Length);
        Assert.Equal(oracle.LastExperts.Distinct().Count(), oracle.LastExperts.Length);
    }

    [Fact]
    public void RewritingARowFromTargetHiddenChangesLaterDraftsAndTruncationInvalidatesRows()
    {
        using var w = Qwen36MtpWeights.Load(TinySidecar.Write(root), TinySidecar.Tiny);
        var a = new Qwen36MtpReference(w); var b = new Qwen36MtpReference(w);
        float[] e0 = Vector(1, 16), e1 = Vector(3, 16), e2 = Vector(5, 16), h0 = Vector(2, 16), h1 = Vector(4, 16), h2 = Vector(6, 16);
        a.Step(e0, h0, 0); a.Step(e1, h1, 1);
        b.Step(e0, h0, 0); b.Step(e1, Vector(99, 16), 1);
        var ya = a.Step(e2, h2, 2); var yb = b.Step(e2, h2, 2);
        Assert.True(ya.Zip(yb).Any(p => MathF.Abs(p.First - p.Second) > 1e-6f), "A different conditioning hidden must change later drafts.");
        // Reconditioning: truncate and rewrite row 1 with the genuine hidden; the result then equals the clean oracle.
        b.Truncate(1); b.Step(e1, h1, 1);
        yb = b.Step(e2, h2, 2);
        Assert.Equal(ya, yb);
        Assert.Throws<InvalidOperationException>(() => b.Step(e2, h2, 4));
        b.Truncate(0);
        Assert.Throws<InvalidOperationException>(() => b.Step(e2, h2, 1));
    }

    [Fact]
    public void RowsAtOrAboveTheCursorAreNeverRead()
    {
        using var w = Qwen36MtpWeights.Load(TinySidecar.Write(root), TinySidecar.Tiny);
        var clean = new Qwen36MtpReference(w); var stale = new Qwen36MtpReference(w);
        float[] e = Vector(1, 16), h = Vector(2, 16);
        stale.Step(Vector(9, 16), Vector(8, 16), 0); stale.Step(Vector(7, 16), Vector(6, 16), 1); stale.Truncate(0);
        Assert.Equal(clean.Step(e, h, 0), stale.Step(e, h, 0));
    }

    [Fact]
    public void TopKIsStableDescendingAndTieBreaksByLowerIndex()
    {
        Assert.Equal([3, 1], Qwen36MtpReference.TopK([0.1f, 0.5f, 0.2f, 0.9f], 2));
        Assert.Equal([0, 1], Qwen36MtpReference.TopK([0.5f, 0.5f, 0.1f], 2));
    }
}
