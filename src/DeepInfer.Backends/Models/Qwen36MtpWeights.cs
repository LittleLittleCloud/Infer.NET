using System.Security.Cryptography;
using System.Text.Json;
namespace DeepInfer.Backends.Models;

// Fixed architecture of the single official MTP layer (Issue #76). Everything is explicit so a
// mismatched checkpoint is rejected rather than reinterpreted.
public sealed record Qwen36MtpGeometry(int Hidden, int Heads, int KvHeads, int HeadDim, int Experts,
    int ExpertIntermediate, int SharedIntermediate, int TopK, int RopeDims, float RopeTheta, float Epsilon)
{
    public static Qwen36MtpGeometry Official { get; } = new(2048, 16, 2, 256, 256, 512, 512, 8, 64, 1e7f, 1e-6f);
    public int QProjRows => Heads * HeadDim * 2;
    public int KvRows => KvHeads * HeadDim;
    public int AttentionWidth => Heads * HeadDim;
}

public static class Qwen36MtpPins
{
    public const string Format = "deepinfer-qwen36-mtp-sidecar";
    public const int FormatVersion = 1;
    public const string SourceRepository = "Qwen/Qwen3.6-35B-A3B";
    public const string SourceRevision = "995ad96eacd98c81ed38be0c5b274b04031597b0";
    public const string TargetRepository = "mlx-community/Qwen3.6-35B-A3B-8bit";
    public const string TargetRevision = "e06a74e6236a60c8367e1a3214e83d8b61b637b0";
    public const long SourceParameters = 844_640_768;
    public static readonly IReadOnlyDictionary<string, (long Bytes, string Sha256)> Shards = new Dictionary<string, (long, string)>
    {
        ["model-00025-of-00026.safetensors"] = (3_832_888_256, "778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154"),
        ["model-00026-of-00026.safetensors"] = (2_231_416_848, "1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613"),
    };
}

// Sidecar tensor names (converter output). Matrices are BF16; norm scales are FP32 raw+1.
public static class Qwen36MtpNames
{
    const string L = "mtp.layers.0.";
    public const string Fc = "mtp.fc.weight";
    public const string PreEmbeddingNorm = "mtp.pre_fc_norm_embedding.scale_fp32";
    public const string PreHiddenNorm = "mtp.pre_fc_norm_hidden.scale_fp32";
    public const string FinalNorm = "mtp.norm.scale_fp32";
    public const string InputNorm = L + "input_layernorm.scale_fp32";
    public const string PostNorm = L + "post_attention_layernorm.scale_fp32";
    public const string QNorm = L + "self_attn.q_norm.scale_fp32";
    public const string KNorm = L + "self_attn.k_norm.scale_fp32";
    public const string Q = L + "self_attn.q_proj.weight";
    public const string K = L + "self_attn.k_proj.weight";
    public const string V = L + "self_attn.v_proj.weight";
    public const string O = L + "self_attn.o_proj.weight";
    public const string Router = L + "mlp.gate.weight";
    public const string ExpertGate = L + "mlp.experts.gate_proj.weight";
    public const string ExpertUp = L + "mlp.experts.up_proj.weight";
    // The official fused expert down tensor has no .weight suffix.
    public const string ExpertDown = L + "mlp.experts.down_proj";
    public const string SharedGate = L + "mlp.shared_expert.gate_proj.weight";
    public const string SharedUp = L + "mlp.shared_expert.up_proj.weight";
    public const string SharedDown = L + "mlp.shared_expert.down_proj.weight";
    public const string SharedRoute = L + "mlp.shared_expert_gate.weight";
    public static readonly string[] Norms = [PreEmbeddingNorm, PreHiddenNorm, FinalNorm, InputNorm, PostNorm, QNorm, KNorm];
    public static readonly string[] Matrices = [Fc, Q, K, V, O, Router, ExpertGate, ExpertUp, ExpertDown, SharedGate, SharedUp, SharedDown, SharedRoute];

    public static (string Name, int[] Shape)[] Layout(Qwen36MtpGeometry g) =>
    [
        (Fc, [g.Hidden, 2 * g.Hidden]), (Q, [g.QProjRows, g.Hidden]), (K, [g.KvRows, g.Hidden]), (V, [g.KvRows, g.Hidden]),
        (O, [g.Hidden, g.AttentionWidth]), (Router, [g.Experts, g.Hidden]),
        (ExpertGate, [g.Experts, g.ExpertIntermediate, g.Hidden]), (ExpertUp, [g.Experts, g.ExpertIntermediate, g.Hidden]),
        (ExpertDown, [g.Experts, g.Hidden, g.ExpertIntermediate]),
        (SharedGate, [g.SharedIntermediate, g.Hidden]), (SharedUp, [g.SharedIntermediate, g.Hidden]),
        (SharedDown, [g.Hidden, g.SharedIntermediate]), (SharedRoute, [1, g.Hidden]),
        (PreEmbeddingNorm, [g.Hidden]), (PreHiddenNorm, [g.Hidden]), (FinalNorm, [g.Hidden]),
        (InputNorm, [g.Hidden]), (PostNorm, [g.Hidden]), (QNorm, [g.HeadDim]), (KNorm, [g.HeadDim]),
    ];
}

public sealed record Qwen36MtpLoadOptions(bool VerifySidecarHash = true, bool RequireOfficialPins = true,
    Qwen36MtpGeometry? Geometry = null);

// Read access used by the CPU oracle; matrices are exact BF16 values expanded to FP32 on demand.
public interface IMtpTensorSource
{
    Qwen36MtpGeometry Geometry { get; }
    float[] ReadNorm(string name);
    void ReadBf16Range(string name, long element, Span<float> destination);
}

// Versioned MTP sidecar. Loading is explicit: any mismatch in model/revision, geometry, dtype,
// shape, hash or norm provenance throws. There is no silent fallback and no weight expansion.
public sealed class Qwen36MtpWeights : IMtpTensorSource, IDisposable
{
    private readonly SafeTensors tensors;
    public Qwen36MtpGeometry Geometry { get; }
    public string SidecarPath { get; }
    public string SidecarSha256 { get; }
    public string ConverterSha256 { get; }
    public string ManifestSha256 { get; }
    public long SidecarBytes { get; }

    private Qwen36MtpWeights(SafeTensors tensors, Qwen36MtpGeometry geometry, string path, string sidecarSha, string converterSha, string manifestSha, long bytes)
    { this.tensors = tensors; Geometry = geometry; SidecarPath = path; SidecarSha256 = sidecarSha; ConverterSha256 = converterSha; ManifestSha256 = manifestSha; SidecarBytes = bytes; }

    public static Qwen36MtpWeights Load(string directory, Qwen36MtpLoadOptions? options = null)
    {
        options ??= new();
        var geometry = options.Geometry ?? Qwen36MtpGeometry.Official;
        if (options.RequireOfficialPins && geometry != Qwen36MtpGeometry.Official)
            throw new InvalidDataException("Only the official MTP geometry may be loaded for inference.");
        var manifestPath = Path.Combine(directory, "qwen36-mtp-sidecar.manifest.json");
        var sidecarPath = Path.Combine(directory, "qwen36-mtp-sidecar.safetensors");
        if (!File.Exists(manifestPath) || !File.Exists(sidecarPath)) throw new FileNotFoundException("MTP sidecar or manifest is missing; run scripts/convert-qwen36-mtp.py.", directory);
        var manifestBytes = File.ReadAllBytes(manifestPath);
        using var doc = JsonDocument.Parse(manifestBytes);
        var root = doc.RootElement;
        Require(root.GetProperty("format").GetString() == Qwen36MtpPins.Format, "Unknown sidecar format.");
        Require(root.GetProperty("formatVersion").GetInt32() == Qwen36MtpPins.FormatVersion, "Unsupported sidecar format version.");
        var source = root.GetProperty("source");
        if (options.RequireOfficialPins)
        {
            Require(source.GetProperty("repository").GetString() == Qwen36MtpPins.SourceRepository, "MTP source repository mismatch.");
            Require(source.GetProperty("revision").GetString() == Qwen36MtpPins.SourceRevision, "MTP source revision mismatch.");
            Require(source.GetProperty("sourceParameters").GetInt64() == Qwen36MtpPins.SourceParameters, "MTP parameter count mismatch.");
            ValidateOfficialSource(source, root.GetProperty("target"));
        }
        var geo = root.GetProperty("geometry");
        Require(geo.GetProperty("hidden").GetInt32() == geometry.Hidden && geo.GetProperty("heads").GetInt32() == geometry.Heads &&
            geo.GetProperty("kvHeads").GetInt32() == geometry.KvHeads && geo.GetProperty("headDim").GetInt32() == geometry.HeadDim &&
            geo.GetProperty("experts").GetInt32() == geometry.Experts && geo.GetProperty("expertIntermediate").GetInt32() == geometry.ExpertIntermediate &&
            geo.GetProperty("sharedIntermediate").GetInt32() == geometry.SharedIntermediate && geo.GetProperty("topK").GetInt32() == geometry.TopK &&
            geo.GetProperty("ropeDims").GetInt32() == geometry.RopeDims && geo.GetProperty("layers").GetInt32() == 1, "MTP geometry mismatch.");
        Require(Math.Abs(geo.GetProperty("ropeTheta").GetDouble() - geometry.RopeTheta) <= 1e-6 * geometry.RopeTheta, "MTP rope theta mismatch.");
        var precision = root.GetProperty("precision");
        Require(precision.GetProperty("matrices").GetString() == "BF16" && precision.GetProperty("normScales").GetString() == "FP32 raw+1" &&
            precision.GetProperty("activations").GetString() == "FP32" && precision.GetProperty("kv").GetString() == "FP32", "MTP precision contract mismatch.");
        var sidecar = root.GetProperty("sidecar");
        var info = new FileInfo(sidecarPath);
        Require(sidecar.GetProperty("bytes").GetInt64() == info.Length, "Sidecar byte length does not match the manifest.");
        var expectedHash = sidecar.GetProperty("sha256").GetString() ?? "";
        string actualHash = expectedHash;
        if (options.VerifySidecarHash)
        {
            using var stream = File.OpenRead(sidecarPath);
            actualHash = Convert.ToHexStringLower(SHA256.HashData(stream));
            Require(actualHash == expectedHash, "Sidecar SHA256 does not match the manifest.");
        }
        var layout = Qwen36MtpNames.Layout(geometry);
        var listed = root.GetProperty("tensors").EnumerateArray().ToDictionary(t => t.GetProperty("name").GetString()!, t => t, StringComparer.Ordinal);
        Require(listed.Count == layout.Length, "Manifest tensor set differs from the MTP layout.");
        var tensors = new SafeTensors(sidecarPath);
        try
        {
            foreach (var (name, shape) in layout)
            {
                Require(listed.TryGetValue(name, out var entry), $"Manifest is missing {name}.");
                var isNorm = Array.IndexOf(Qwen36MtpNames.Norms, name) >= 0;
                var tensor = tensors.Get(name, shape);
                Require(tensor.Dtype == (isNorm ? "F32" : "BF16"), $"{name} has dtype {tensor.Dtype}.");
                Require(entry.GetProperty("dtype").GetString() == tensor.Dtype && entry.GetProperty("bytes").GetInt64() == tensor.Bytes, $"{name} manifest entry mismatches the sidecar.");
                var transform = isNorm ? "raw_bf16_float_plus_one_fp32" : name == Qwen36MtpNames.ExpertGate ? "gate_up_first_half" : name == Qwen36MtpNames.ExpertUp ? "gate_up_second_half" : "copy";
                Require(entry.GetProperty("transform").GetString() == transform, $"{name} has an unexpected transform.");
                ValidateProvenance(name, shape, entry, isNorm, options.RequireOfficialPins);
                if (options.VerifySidecarHash) Require(tensors.Sha256Hex(tensor) == entry.GetProperty("sha256").GetString(), $"{name} payload SHA256 does not match the manifest.");
                if (isNorm) _ = tensors.ReadFloat(tensor); // finite check
            }
            return new(tensors, geometry, sidecarPath, actualHash, root.GetProperty("converterSha256").GetString() ?? "",
                Convert.ToHexStringLower(SHA256.HashData(manifestBytes)), info.Length);
        }
        catch { tensors.Dispose(); throw; }
    }

    // Source identity: exactly the two pinned official shards (unique names, sizes, complete SHA256, verified) and the pinned target.
    public static void ValidateOfficialSource(JsonElement source, JsonElement target)
    {
        Require(source.GetProperty("repository").GetString() == Qwen36MtpPins.SourceRepository, "MTP source repository mismatch.");
        Require(source.GetProperty("revision").GetString() == Qwen36MtpPins.SourceRevision, "MTP source revision mismatch.");
        Require(source.GetProperty("sourceParameters").GetInt64() == Qwen36MtpPins.SourceParameters, "MTP parameter count mismatch.");
        var shards = source.GetProperty("shards").EnumerateArray().ToArray();
        var files = shards.Select(x => x.GetProperty("file").GetString() ?? "").ToArray();
        Require(shards.Length == 2 && files.Distinct(StringComparer.Ordinal).Count() == 2 && files.All(Qwen36MtpPins.Shards.ContainsKey),
            "Exactly the two unique official shards must be recorded.");
        foreach (var shard in shards)
        {
            var file = shard.GetProperty("file").GetString()!; var pin = Qwen36MtpPins.Shards[file];
            Require(shard.GetProperty("bytes").GetInt64() == pin.Bytes && shard.GetProperty("sha256").GetString() == pin.Sha256 && shard.GetProperty("verified").GetBoolean(),
                $"Official shard {file} was not verified against its pinned SHA256.");
        }
        Require(target.GetProperty("repository").GetString() == Qwen36MtpPins.TargetRepository &&
            target.GetProperty("revision").GetString() == Qwen36MtpPins.TargetRevision && target.GetProperty("sharedEmbeddingAndHead").GetBoolean(), "Target model/revision mismatch.");
    }

    // Binds each sidecar tensor to its official source tensor: source name/shard/shape/hash, and (for copies) payload equality with the source hash.
    private static void ValidateProvenance(string name, int[] shape, JsonElement entry, bool isNorm, bool official)
    {
        var expectedSource = isNorm ? name[..^".scale_fp32".Length] + ".weight"
            : name == Qwen36MtpNames.ExpertGate || name == Qwen36MtpNames.ExpertUp ? Qwen36MtpNames.ExpertGate.Replace("gate_proj.weight", "gate_up_proj") : name;
        Require(entry.GetProperty("sourceName").GetString() == expectedSource, $"{name} source tensor name mismatch.");
        var shard = entry.GetProperty("sourceShard").GetString() ?? "";
        Require(shard.Length > 0 && (!official || Qwen36MtpPins.Shards.ContainsKey(shard)), $"{name} source shard is not an official pinned shard.");
        var sourceShape = entry.GetProperty("sourceShape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var fused = name == Qwen36MtpNames.ExpertGate || name == Qwen36MtpNames.ExpertUp;
        int[] expectedShape = fused ? [shape[0], shape[1] * 2, shape[2]] : shape;
        Require(sourceShape.SequenceEqual(expectedShape), $"{name} source shape mismatch.");
        foreach (var field in new[] { "sha256", "sourceSha256" })
        {
            var hex = entry.GetProperty(field).GetString() ?? "";
            Require(hex.Length == 64 && hex.All(Uri.IsHexDigit), $"{name} {field} is not a complete SHA256.");
        }
        if (entry.GetProperty("transform").GetString() == "copy")
            Require(entry.GetProperty("sha256").GetString() == entry.GetProperty("sourceSha256").GetString(), $"{name} copy payload hash must equal its source hash.");
    }

    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }

    public int[] Shape(string name) => Layout(name).Shape;
    private (string Name, int[] Shape) Layout(string name) => Qwen36MtpNames.Layout(Geometry).First(x => x.Name == name);
    public long TensorBytes(string name) => tensors.Get(name, Shape(name)).Bytes;
    public void ReadBytes(string name, Span<byte> destination) => tensors.ReadInto(tensors.Get(name, Shape(name)), destination);
    public float[] ReadNorm(string name)
    {
        if (Array.IndexOf(Qwen36MtpNames.Norms, name) < 0) throw new ArgumentException("Not an MTP norm tensor.", nameof(name));
        return tensors.ReadFloat(tensors.Get(name, Shape(name)));
    }
    public void ReadBf16Range(string name, long element, Span<float> destination) =>
        tensors.ReadBFloat16Range(tensors.Get(name, Shape(name)), element, destination);
    public ushort[] ReadBf16(string name)
    {
        if (Array.IndexOf(Qwen36MtpNames.Matrices, name) < 0) throw new ArgumentException("Not an MTP matrix tensor.", nameof(name));
        return tensors.ReadBFloat16(tensors.Get(name, Shape(name)));
    }
    public void Dispose() => tensors.Dispose();
}
