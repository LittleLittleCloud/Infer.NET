using System.Text.Json;
using DeepInfer.Backends.Compute;
namespace DeepInfer.Backends.Models;

// Pinned MLX affine checkpoint layout only; model execution is implemented here.
public sealed class Qwen36Weights : IDisposable
{
    public const int Dimension = 2048, Layers = 40, Experts = 256, Selected = 8, Intermediate = 512, Vocabulary = 248320;
    public const int Context = 262144;
    public float Epsilon { get; }
    public float RopeTheta { get; }
    public int WeightBits { get; }
    public QuantMatrix Embedding { get; }
    public QuantMatrix Classifier { get; }
    public float[] FinalNorm { get; }
    public Layer[] Blocks { get; }
    private readonly List<QuantMatrix> allocations = [];
    public sealed class Layer
    {
        public float[] InputNorm = [], PostNorm = [], Conv = [], ALog = [], DtBias = [], GatedNorm = [], QNorm = [], KNorm = [];
        public QuantMatrix Router = null!, ExpertGate = null!, ExpertUp = null!, ExpertDown = null!;
        public QuantMatrix SharedGate = null!, SharedUp = null!, SharedDown = null!, SharedRoute = null!;
        public QuantMatrix Q = null!, K = null!, V = null!, Z = null!, A = null!, B = null!, O = null!;
        public bool Linear;
    }
    public Qwen36Weights(string directory, QuantizedMetal device)
    {
        directory = Path.GetFullPath(directory);
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        var root = document.RootElement; var c = root.GetProperty("text_config");
        if (root.GetProperty("model_type").GetString() != "qwen3_5_moe" ||
            c.GetProperty("hidden_size").GetInt32() != Dimension || c.GetProperty("num_hidden_layers").GetInt32() != Layers ||
            c.GetProperty("num_experts").GetInt32() != Experts || c.GetProperty("num_experts_per_tok").GetInt32() != Selected ||
            c.GetProperty("moe_intermediate_size").GetInt32() != Intermediate || c.GetProperty("vocab_size").GetInt32() != Vocabulary ||
            c.GetProperty("num_attention_heads").GetInt32() != 16 || c.GetProperty("num_key_value_heads").GetInt32() != 2 ||
            c.GetProperty("head_dim").GetInt32() != 256 || c.GetProperty("full_attention_interval").GetInt32() != 4 ||
            c.GetProperty("linear_num_value_heads").GetInt32() != 32 || c.GetProperty("linear_num_key_heads").GetInt32() != 16 ||
            c.GetProperty("linear_key_head_dim").GetInt32() != 128 || c.GetProperty("linear_value_head_dim").GetInt32() != 128 ||
            c.GetProperty("linear_conv_kernel_dim").GetInt32() != 4 ||
            (c.TryGetProperty("norm_topk_prob", out var normalized) && !normalized.GetBoolean()))
            throw new NotSupportedException("Only the Qwen3.6-35B-A3B text geometry is implemented.");
        var rope = c.GetProperty("rope_parameters");
        if (c.GetProperty("max_position_embeddings").GetInt32() != Context)
            throw new NotSupportedException("Expected the pinned 262144-token context geometry.");
        if (rope.GetProperty("partial_rotary_factor").GetSingle() != .25f ||
            (rope.TryGetProperty("rope_type", out var type) && type.GetString() != "default"))
            throw new NotSupportedException("Only unscaled text RoPE is implemented; no YaRN/multimodal input.");
        Epsilon = c.GetProperty("rms_norm_eps").GetSingle(); RopeTheta = rope.GetProperty("rope_theta").GetSingle();
        if (!float.IsFinite(Epsilon) || Epsilon <= 0 || !float.IsFinite(RopeTheta) || RopeTheta <= 1)
            throw new InvalidDataException("Invalid Qwen3.6 normalization/RoPE parameters.");
        var quant = root.GetProperty("quantization");
        WeightBits = quant.GetProperty("bits").GetInt32();
        if (quant.GetProperty("group_size").GetInt32() != 64 || WeightBits is not (5 or 8) || quant.GetProperty("mode").GetString() != "affine")
            throw new NotSupportedException("Expected affine 5-bit or 8-bit/group64 weights.");
        using var index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "model.safetensors.index.json")));
        var map = index.RootElement.GetProperty("weight_map");
        var shards = new Dictionary<string, SafeTensors>(StringComparer.Ordinal);
        SafeTensors FileFor(string name)
        {
            var file = map.GetProperty(name).GetString() ?? throw new InvalidDataException("Missing tensor shard.");
            if (Path.GetFileName(file) != file || file.Contains('\\') || file is "." or "..") throw new InvalidDataException("Unsafe shard path.");
            if (!shards.TryGetValue(file, out var shard)) shards[file] = shard = new(Path.Combine(directory, file));
            return shard;
        }
        QuantMatrix Matrix(string prefix, int rows, int columns, int experts = 1, int bits = 0)
        {
            bits = bits == 0 ? WeightBits : bits;
            if (quant.TryGetProperty(prefix, out var local) &&
                (local.GetProperty("bits").GetInt32() != bits || local.GetProperty("group_size").GetInt32() != 64 ||
                 (local.TryGetProperty("mode", out var mode) && mode.GetString() != "affine")))
                throw new NotSupportedException($"Unsupported quantization override: {prefix}");
            int[] Shape(int width) => experts == 1 ? [rows, width] : [experts, rows, width];
            var f = FileFor(prefix + ".weight"); var w = f.Get(prefix + ".weight", Shape(columns * bits / 32));
            var sf = FileFor(prefix + ".scales"); var s = sf.Get(prefix + ".scales", Shape(columns / 64));
            var bf = FileFor(prefix + ".biases"); var b = bf.Get(prefix + ".biases", Shape(columns / 64));
            if (w.Dtype != "U32" || s.Dtype != "BF16" || b.Dtype != "BF16") throw new NotSupportedException("Expected U32 packed weights and BF16 affine parameters.");
            var m = device.Allocate(experts, rows, columns, bits); allocations.Add(m);
            m.ReadTensor(0, f, w); m.ReadTensor(1, sf, s); m.ReadTensor(2, bf, b);
            return m;
        }
        float[] Vector(string name, params int[] shape) { var f = FileFor(name); return f.ReadFloat(f.Get(name, shape)); }
        try
        {
            Embedding = Matrix("language_model.model.embed_tokens", Vocabulary, Dimension);
            Classifier = Matrix("language_model.lm_head", Vocabulary, Dimension);
            FinalNorm = Vector("language_model.model.norm.weight", Dimension);
            Blocks = new Layer[Layers];
            for (var l = 0; l < Layers; l++)
            {
                var p = $"language_model.model.layers.{l}.";
                var b = new Layer { Linear = (l + 1) % 4 != 0,
                    InputNorm = Vector(p + "input_layernorm.weight", Dimension), PostNorm = Vector(p + "post_attention_layernorm.weight", Dimension),
                    Router = Matrix(p + "mlp.gate", Experts, Dimension, bits: 8),
                    ExpertGate = Matrix(p + "mlp.switch_mlp.gate_proj", Intermediate, Dimension, Experts),
                    ExpertUp = Matrix(p + "mlp.switch_mlp.up_proj", Intermediate, Dimension, Experts),
                    ExpertDown = Matrix(p + "mlp.switch_mlp.down_proj", Dimension, Intermediate, Experts),
                    SharedGate = Matrix(p + "mlp.shared_expert.gate_proj", Intermediate, Dimension),
                    SharedUp = Matrix(p + "mlp.shared_expert.up_proj", Intermediate, Dimension),
                    SharedDown = Matrix(p + "mlp.shared_expert.down_proj", Dimension, Intermediate),
                    SharedRoute = Matrix(p + "mlp.shared_expert_gate", 1, Dimension, bits: 8) };
                if (b.Linear)
                {
                    var a = p + "linear_attn.";
                    b.Q = Matrix(a + "in_proj_qkv", 8192, Dimension); b.Z = Matrix(a + "in_proj_z", 4096, Dimension);
                    b.A = Matrix(a + "in_proj_a", 32, Dimension); b.B = Matrix(a + "in_proj_b", 32, Dimension);
                    b.O = Matrix(a + "out_proj", Dimension, 4096);
                    b.Conv = Vector(a + "conv1d.weight", 8192, 4, 1); b.ALog = Vector(a + "A_log", 32);
                    b.DtBias = Vector(a + "dt_bias", 32); b.GatedNorm = Vector(a + "norm.weight", 128);
                }
                else
                {
                    var a = p + "self_attn.";
                    b.Q = Matrix(a + "q_proj", 8192, Dimension); b.K = Matrix(a + "k_proj", 512, Dimension);
                    b.V = Matrix(a + "v_proj", 512, Dimension); b.O = Matrix(a + "o_proj", Dimension, 4096);
                    b.QNorm = Vector(a + "q_norm.weight", 256); b.KNorm = Vector(a + "k_norm.weight", 256);
                }
                Blocks[l] = b;
            }
        }
        catch { Dispose(); throw; }
        finally { foreach (var f in shards.Values) f.Dispose(); }
    }
    public void Dispose() { foreach (var m in allocations) m.Dispose(); allocations.Clear(); }
}
