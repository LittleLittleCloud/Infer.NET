using System.Text.Json;
using DeepInfer.Backends.Compute;
namespace DeepInfer.Backends.Models;

public sealed class QwenCheckpoint : ITransformerWeights
{
    public LlamaConfig Config { get; }
    public float Epsilon { get; }
    public float RopeTheta { get; }
    public bool SplitHalfRope => true;
    public float[][] AttentionNorm { get; }
    public float[][] FfnNorm { get; }
    public float[] FinalNorm { get; }
    public IMatrix[] Query { get; }
    public IMatrix[] Key { get; }
    public IMatrix[] Value { get; }
    public IMatrix[] AttentionOutput { get; }
    public IMatrix[] Gate { get; }
    public IMatrix[] Down { get; }
    public IMatrix[] Up { get; }
    public IMatrix Classifier { get; }
    public float[][]? QueryBias { get; }
    public float[][]? KeyBias { get; }
    public float[][]? ValueBias { get; }
    private readonly ushort[]? embeddingBf16;
    private readonly float[]? embeddingFloat;
    private readonly List<IMatrix> allocations = [];

    public QwenCheckpoint(string directory, IMatVecDevice device)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Little-endian host required.");
        directory = Path.GetFullPath(directory);
        using var config = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "config.json")));
        var c = config.RootElement;
        int Int(string key) => c.GetProperty(key).GetInt32();
        if (c.GetProperty("model_type").GetString() != "qwen2" || c.GetProperty("hidden_act").GetString() != "silu" ||
            (c.TryGetProperty("use_sliding_window", out var sliding) && sliding.GetBoolean()) ||
            (c.TryGetProperty("rope_scaling", out var scaling) && scaling.ValueKind != JsonValueKind.Null))
            throw new NotSupportedException("Only dense Qwen2/Qwen2.5 with standard RoPE and no sliding window is supported.");
        var dim = Int("hidden_size"); var hidden = Int("intermediate_size"); var layers = Int("num_hidden_layers");
        var heads = Int("num_attention_heads"); var kvHeads = Int("num_key_value_heads"); var vocab = Int("vocab_size");
        if (dim is < 2 or > 4096 || hidden is < 1 or > 32768 || layers is < 1 or > 80 || heads < 1 || dim % heads != 0 ||
            (dim / heads) % 2 != 0 || kvHeads < 1 || heads % kvHeads != 0 || vocab is < 259 or > 200000)
            throw new InvalidDataException("Unsupported Qwen dimensions.");
        Config = new(dim, hidden, layers, heads, kvHeads, vocab, Math.Min(2048, Int("max_position_embeddings")));
        if (Config.Context < 2 || 2L * layers * Config.Context * Config.KvDimension * 4 > 536870912)
            throw new InvalidDataException("Invalid context or KV memory limit exceeded.");
        Epsilon = c.GetProperty("rms_norm_eps").GetSingle(); RopeTheta = c.GetProperty("rope_theta").GetSingle();
        if (!float.IsFinite(Epsilon) || Epsilon <= 0 || !float.IsFinite(RopeTheta) || RopeTheta <= 1) throw new InvalidDataException("Invalid normalization/RoPE configuration.");
        using var index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(directory, "model.safetensors.index.json")));
        var map = index.RootElement.GetProperty("weight_map"); var shards = new Dictionary<string, SafeTensors>(StringComparer.Ordinal);
        try
        {
            SafeTensors Shard(string name)
            {
                var file = map.GetProperty(name).GetString() ?? throw new InvalidDataException("Missing shard file.");
                if (Path.GetFileName(file) != file || file.Contains('\\') || file is "." or "..") throw new InvalidDataException("Shard paths must be file names.");
                if (!shards.TryGetValue(file, out var shard)) shards[file] = shard = new SafeTensors(Path.Combine(directory, file));
                return shard;
            }
            IMatrix Matrix(string name, int rows, int columns)
            {
                var shard = Shard(name); var tensor = shard.Get(name, rows, columns);
                var matrix = tensor.Dtype == "BF16" ? device.UploadBFloat16(shard.ReadBFloat16(tensor), rows, columns) : device.Upload(shard.ReadFloat(tensor), rows, columns);
                allocations.Add(matrix); return matrix;
            }
            float[] Vector(string name, int length) { var shard = Shard(name); return shard.ReadFloat(shard.Get(name, length)); }
            IMatrix[] Matrices(string suffix, int rows, int columns) => Enumerable.Range(0, layers).Select(i => Matrix($"model.layers.{i}.{suffix}", rows, columns)).ToArray();
            float[][] Vectors(string suffix, int length) => Enumerable.Range(0, layers).Select(i => Vector($"model.layers.{i}.{suffix}", length)).ToArray();
            var embeddingShard = Shard("model.embed_tokens.weight"); var embedding = embeddingShard.Get("model.embed_tokens.weight", vocab, dim);
            if (embedding.Dtype == "BF16") embeddingBf16 = embeddingShard.ReadBFloat16(embedding); else embeddingFloat = embeddingShard.ReadFloat(embedding);
            AttentionNorm = Vectors("input_layernorm.weight", dim); FfnNorm = Vectors("post_attention_layernorm.weight", dim); FinalNorm = Vector("model.norm.weight", dim);
            Query = Matrices("self_attn.q_proj.weight", dim, dim); Key = Matrices("self_attn.k_proj.weight", Config.KvDimension, dim);
            Value = Matrices("self_attn.v_proj.weight", Config.KvDimension, dim); AttentionOutput = Matrices("self_attn.o_proj.weight", dim, dim);
            QueryBias = Vectors("self_attn.q_proj.bias", dim); KeyBias = Vectors("self_attn.k_proj.bias", Config.KvDimension); ValueBias = Vectors("self_attn.v_proj.bias", Config.KvDimension);
            Gate = Matrices("mlp.gate_proj.weight", hidden, dim); Down = Matrices("mlp.down_proj.weight", dim, hidden); Up = Matrices("mlp.up_proj.weight", hidden, dim);
            Classifier = c.GetProperty("tie_word_embeddings").GetBoolean()
                ? embeddingBf16 is not null ? device.UploadBFloat16(embeddingBf16, vocab, dim) : device.Upload(embeddingFloat!, vocab, dim)
                : Matrix("lm_head.weight", vocab, dim);
            if (!allocations.Contains(Classifier)) allocations.Add(Classifier);
        }
        catch { Dispose(); throw; }
        finally { foreach (var shard in shards.Values) shard.Dispose(); }
    }
    public void CopyEmbedding(int token, Span<float> output)
    {
        if (embeddingBf16 is not null)
        {
            var row = embeddingBf16.AsSpan(checked(token * Config.Dimension), Config.Dimension);
            for (var i = 0; i < row.Length; i++) output[i] = BitConverter.Int32BitsToSingle(row[i] << 16);
        }
        else embeddingFloat!.AsSpan(checked(token * Config.Dimension), Config.Dimension).CopyTo(output);
    }
    public void Dispose() { foreach (var matrix in allocations) matrix.Dispose(); allocations.Clear(); }
}
