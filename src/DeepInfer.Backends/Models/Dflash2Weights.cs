using System.Security.Cryptography;
using System.Text.Json;
namespace DeepInfer.Backends.Models;

// Pinned DFlash2 draft checkpoint (Issue #80). Every geometry value is explicit; a mismatch is an error, never reinterpreted.
public static class Dflash2Pins
{
    public const string Repository = "incoai/Qwen3.6-35B-A3B-DFlash2";
    public const string Revision = "51ef7b6923ad6c14cb1bb41c37a9041446496ab6";
    public const long WeightBytes = 1_052_513_456;
    public const string WeightSha256 = "d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d";
    public const string ConfigSha256 = "86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59";
    public const string UpstreamSource = "z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a";
}

public static class Dflash2Geometry
{
    public const int Hidden = 2048, Layers = 6, Heads = 32, KvHeads = 8, HeadDim = 128, Intermediate = 6144, Window = 2048, MaxBlock = 8;
    public const int ConvKernel = 2, ConvGroup = 16, SelectorRank = 256, SelectorTopK = 16, Vocabulary = 248320, TapCount = 8;
    public const int MaskToken = 248077, EosToken = 248044;
    public const float Epsilon = 1e-6f, RopeTheta = 1e7f;
    public static readonly int[] TapLayers = [1, 6, 11, 16, 22, 27, 32, 37];
    public const int TapWidth = TapCount * Hidden;
}

// Tensor names and the exact order shared with the native slot ABI (gpu_dflash.mm: DfMat / DfLayerMat / DfLayerVec).
public static class Dflash2Names
{
    public const string Fc = "fc.weight", HiddenNorm = "hidden_norm.weight", FinalNorm = "norm.weight";
    public const string HiddenProjection = "candidate_selector.hidden_projection.weight";
    // The official codebooks have no ".weight" suffix.
    public const string PredecessorCodebook = "candidate_selector.predecessor_codebook", SuccessorCodebook = "candidate_selector.successor_codebook";
    public static string Layer(int layer, string tail) => $"layers.{layer}.{tail}";
    public static readonly string[] LayerMatrixTails =
    [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "attention_conv.kernel_projection.weight", "mlp_conv.kernel_projection.weight",
    ];
    public static readonly string[] LayerVectorTails =
    [
        "input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "attention_conv.base_kernel", "mlp_conv.base_kernel",
    ];
    public static IReadOnlyList<(string Name, int[] Shape)> Matrices { get; } = BuildMatrices();
    // Norm scales are direct BF16 values (not raw+1); conv base kernels are [2,2,2048].
    public static IReadOnlyList<(string Name, int[] Shape)> Vectors { get; } = BuildVectors();
    private static List<(string, int[])> BuildMatrices()
    {
        const int H = Dflash2Geometry.Hidden;
        var list = new List<(string, int[])> { (Fc, [H, Dflash2Geometry.TapWidth]), (HiddenProjection, [Dflash2Geometry.SelectorRank, H]),
            (PredecessorCodebook, [Dflash2Geometry.Vocabulary, Dflash2Geometry.SelectorRank]), (SuccessorCodebook, [Dflash2Geometry.Vocabulary, Dflash2Geometry.SelectorRank]) };
        int[][] shapes = [[4096, H], [1024, H], [1024, H], [H, 4096], [6144, H], [6144, H], [H, 6144], [512, H], [512, H]];
        for (var l = 0; l < Dflash2Geometry.Layers; l++)
            for (var i = 0; i < shapes.Length; i++) list.Add((Layer(l, LayerMatrixTails[i]), shapes[i]));
        return list;
    }
    private static List<(string, int[])> BuildVectors()
    {
        const int H = Dflash2Geometry.Hidden;
        var list = new List<(string, int[])> { (HiddenNorm, [H]), (FinalNorm, [H]) };
        int[][] shapes = [[H], [H], [Dflash2Geometry.HeadDim], [Dflash2Geometry.HeadDim], [2, 2, H], [2, 2, H]];
        for (var l = 0; l < Dflash2Geometry.Layers; l++)
            for (var i = 0; i < shapes.Length; i++) list.Add((Layer(l, LayerVectorTails[i]), shapes[i]));
        return list;
    }
}

public sealed record Dflash2LoadOptions(bool VerifyHash = true, bool RequirePins = true);

// Read access used by the independent CPU oracle.
public interface IDflash2TensorSource
{
    float[] ReadVector(string name);
    void ReadBf16Range(string name, long element, Span<float> destination);
}

// Strict loader: exact pins, config, tensor set/dtypes/shapes and finite norms. No fallback, no conversion, no quantization.
public sealed class Dflash2Weights : IDflash2TensorSource, IDisposable
{
    private readonly SafeTensors tensors;
    public string SourceDirectory { get; }
    public string WeightSha256 { get; }
    public string ConfigSha256 { get; }
    public long WeightBytes { get; }
    private Dflash2Weights(SafeTensors tensors, string directory, string weightSha, string configSha, long bytes)
    { this.tensors = tensors; SourceDirectory = directory; WeightSha256 = weightSha; ConfigSha256 = configSha; WeightBytes = bytes; }

    public static Dflash2Weights Load(string directory, Dflash2LoadOptions? options = null)
    {
        options ??= new();
        var configPath = Path.Combine(directory, "config.json"); var weightPath = Path.Combine(directory, "model.safetensors");
        if (!File.Exists(configPath) || !File.Exists(weightPath)) throw new FileNotFoundException("DFlash2 config.json or model.safetensors is missing.", directory);
        var configBytes = File.ReadAllBytes(configPath);
        var configSha = Convert.ToHexStringLower(SHA256.HashData(configBytes));
        ValidateConfig(configBytes);
        if (options.RequirePins && configSha != Dflash2Pins.ConfigSha256) throw new InvalidDataException("DFlash2 config.json hash differs from the pinned revision.");
        var bytes = new FileInfo(weightPath).Length;
        if (options.RequirePins && bytes != Dflash2Pins.WeightBytes) throw new InvalidDataException($"DFlash2 weight size {bytes} differs from pinned {Dflash2Pins.WeightBytes}.");
        var weightSha = "";
        if (options.VerifyHash)
        {
            using var file = File.OpenRead(weightPath);
            weightSha = Convert.ToHexStringLower(SHA256.HashData(file));
            if (options.RequirePins && weightSha != Dflash2Pins.WeightSha256) throw new InvalidDataException("DFlash2 weight SHA-256 differs from the pinned revision.");
        }
        else if (options.RequirePins) throw new InvalidDataException("Pinned DFlash2 loading requires the full SHA-256 verification.");
        var tensors = new SafeTensors(weightPath);
        try
        {
            var expected = new HashSet<string>(StringComparer.Ordinal);
            foreach (var (name, shape) in Dflash2Names.Matrices.Concat(Dflash2Names.Vectors))
            {
                var t = tensors.Get(name, shape);
                if (t.Dtype != "BF16") throw new InvalidDataException($"DFlash2 tensor {name} must be BF16 (found {t.Dtype}); no conversion is performed.");
                expected.Add(name);
            }
            var extra = tensors.Names.Where(n => !expected.Contains(n)).ToArray();
            if (extra.Length > 0) throw new InvalidDataException($"Unexpected DFlash2 tensors: {string.Join(", ", extra.Take(4))}.");
            if (tensors.Names.Count != expected.Count) throw new InvalidDataException("DFlash2 tensor count mismatch.");
            return new Dflash2Weights(tensors, directory, weightSha, configSha, bytes);
        }
        catch { tensors.Dispose(); throw; }
    }

    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException("DFlash2 config: " + message); }
    internal static void ValidateConfig(ReadOnlySpan<byte> json)
    {
        using var doc = JsonDocument.Parse(json.ToArray());
        var r = doc.RootElement;
        Require(r.GetProperty("architectures")[0].GetString() == "DFlash2DraftModel", "architecture");
        Require(r.GetProperty("hidden_size").GetInt32() == Dflash2Geometry.Hidden, "hidden_size");
        Require(r.GetProperty("num_hidden_layers").GetInt32() == Dflash2Geometry.Layers, "num_hidden_layers");
        Require(r.GetProperty("num_attention_heads").GetInt32() == Dflash2Geometry.Heads, "num_attention_heads");
        Require(r.GetProperty("num_key_value_heads").GetInt32() == Dflash2Geometry.KvHeads, "num_key_value_heads");
        Require(r.GetProperty("head_dim").GetInt32() == Dflash2Geometry.HeadDim, "head_dim");
        Require(r.GetProperty("intermediate_size").GetInt32() == Dflash2Geometry.Intermediate, "intermediate_size");
        Require(r.GetProperty("sliding_window").GetInt32() == Dflash2Geometry.Window, "sliding_window");
        Require(r.GetProperty("vocab_size").GetInt32() == Dflash2Geometry.Vocabulary, "vocab_size");
        Require(r.GetProperty("num_target_layers").GetInt32() == 40, "num_target_layers");
        Require(r.GetProperty("hidden_act").GetString() == "silu", "hidden_act");
        Require(!r.GetProperty("is_causal").GetBoolean(), "is_causal must be false");
        Require(Math.Abs(r.GetProperty("rms_norm_eps").GetDouble() - Dflash2Geometry.Epsilon) < 1e-12, "rms_norm_eps");
        Require(!r.GetProperty("attention_bias").GetBoolean(), "attention_bias");
        var rope = r.GetProperty("rope_parameters");
        Require(rope.GetProperty("rope_type").GetString() == "default" && Math.Abs(rope.GetProperty("rope_theta").GetDouble() - Dflash2Geometry.RopeTheta) < 1, "rope");
        var types = r.GetProperty("layer_types").EnumerateArray().Select(x => x.GetString()).ToArray();
        Require(types.Length == Dflash2Geometry.Layers && types.All(t => t == "sliding_attention"), "layer_types");
        var d = r.GetProperty("dflash_config");
        Require(d.GetProperty("block_size").GetInt32() == Dflash2Geometry.MaxBlock, "block_size");
        Require(d.GetProperty("mask_token_id").GetInt32() == Dflash2Geometry.MaskToken, "mask_token_id");
        Require(d.GetProperty("target_layer_ids").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(Dflash2Geometry.TapLayers), "target_layer_ids");
        Require(d.GetProperty("conv_group_size").GetInt32() == Dflash2Geometry.ConvGroup, "conv_group_size");
        Require(d.GetProperty("conv_kernel_size").GetInt32() == Dflash2Geometry.ConvKernel, "conv_kernel_size");
        Require(d.GetProperty("selector_rank").GetInt32() == Dflash2Geometry.SelectorRank, "selector_rank");
        Require(d.GetProperty("selector_top_k").GetInt32() == Dflash2Geometry.SelectorTopK, "selector_top_k");
    }

    public long TensorBytes(string name) => tensors.Get(name, ShapeOf(name)).Bytes;
    private static int[] ShapeOf(string name)
    {
        foreach (var (n, s) in Dflash2Names.Matrices) if (n == name) return s;
        foreach (var (n, s) in Dflash2Names.Vectors) if (n == name) return s;
        throw new ArgumentException("Unknown DFlash2 tensor: " + name);
    }
    public void ReadBytes(string name, Span<byte> destination) => tensors.ReadInto(tensors.Get(name, ShapeOf(name)), destination);
    // Direct BF16 -> FP32 values (draft RMSNorm scales and conv kernels are trained as-is; never raw+1).
    public float[] ReadVector(string name) => tensors.ReadFloat(tensors.Get(name, ShapeOf(name)));
    public void ReadBf16Range(string name, long element, Span<float> destination) => tensors.ReadBFloat16Range(tensors.Get(name, ShapeOf(name)), element, destination);
    public void Dispose() => tensors.Dispose();
}
