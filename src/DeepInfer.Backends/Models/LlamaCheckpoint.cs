using System.Runtime.InteropServices;
using DeepInfer.Backends.Compute;
namespace DeepInfer.Backends.Models;

public sealed record LlamaConfig(int Dimension, int HiddenDimension, int Layers, int Heads, int KvHeads, int Vocabulary, int Context)
{
    public int HeadSize => Dimension / Heads;
    public int KvDimension => HeadSize * KvHeads;
}

// The legacy llama2.c FP32 format: seven little-endian int32 values, then row-major float tensors.
// Read order and tokenizer compatibility follow llama2.c (MIT); see THIRD-PARTY-NOTICES.md.
public sealed class LlamaCheckpoint : ITransformerWeights
{
    public LlamaConfig Config { get; }
    public float Epsilon => 1e-5f;
    public float RopeTheta => 10000f;
    public bool SplitHalfRope => false;
    public float[][]? QueryBias => null;
    public float[][]? KeyBias => null;
    public float[][]? ValueBias => null;
    public void CopyEmbedding(int token, Span<float> output) => Embedding.AsSpan(token * Config.Dimension, Config.Dimension).CopyTo(output);
    public float[] Embedding { get; }
    public float[][] AttentionNorm { get; }
    public float[][] FfnNorm { get; }
    public float[] FinalNorm { get; }
    public IMatrix[] Query { get; private set; } = [];
    public IMatrix[] Key { get; private set; } = [];
    public IMatrix[] Value { get; private set; } = [];
    public IMatrix[] AttentionOutput { get; private set; } = [];
    public IMatrix[] Gate { get; private set; } = [];
    public IMatrix[] Down { get; private set; } = [];
    public IMatrix[] Up { get; private set; } = [];
    public IMatrix Classifier { get; private set; } = null!;
    private readonly List<IMatrix> allocations = [];

    public LlamaCheckpoint(string path, IMatVecDevice device)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Little-endian host required.");
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);
        if (stream.Length is < 28 or > 536870912) throw new InvalidDataException("Checkpoint must be 28 bytes..512 MiB.");
        try
        {
            var dim = reader.ReadInt32(); var hidden = reader.ReadInt32(); var layers = reader.ReadInt32();
            var heads = reader.ReadInt32(); var kvHeads = reader.ReadInt32(); var rawVocab = reader.ReadInt32(); var context = reader.ReadInt32();
            if (dim is < 2 or > 4096 || hidden is < 1 or > 16384 || layers is < 1 or > 128 || heads < 1 ||
                heads > dim || dim % heads != 0 || (dim / heads) % 2 != 0 || kvHeads < 1 || heads % kvHeads != 0 ||
                rawVocab == int.MinValue || Math.Abs(rawVocab) is < 259 or > 128000 || context is < 2 or > 8192)
                throw new InvalidDataException("Invalid or unsupported llama2.c FP32 configuration.");
            Config = new(dim, hidden, layers, heads, kvHeads, Math.Abs(rawVocab), context);
            var c = Config; var kv = c.KvDimension;
            var count = (long)c.Vocabulary * dim + (long)layers * (2L * dim + 2L * dim * dim + 2L * kv * dim + 3L * dim * hidden) + dim + (long)context * c.HeadSize;
            if (rawVocab < 0) count += (long)c.Vocabulary * dim;
            if (checked(28L + count * sizeof(float)) != stream.Length) throw new InvalidDataException("Checkpoint tensor sizes do not match its header/file length.");
            if (2L * layers * context * kv * sizeof(float) > 268435456) throw new InvalidDataException("KV cache exceeds the 256 MiB safety limit.");
            Embedding = Read(checked(c.Vocabulary * dim));
            AttentionNorm = Norms();
            Query = Matrices(dim, dim); Key = Matrices(kv, dim); Value = Matrices(kv, dim); AttentionOutput = Matrices(dim, dim);
            FfnNorm = Norms();
            Gate = Matrices(hidden, dim); Down = Matrices(dim, hidden); Up = Matrices(hidden, dim);
            FinalNorm = Read(dim);
            stream.Seek(checked((long)context * c.HeadSize * sizeof(float)), SeekOrigin.Current); // obsolete RoPE frequency tensors
            Classifier = Upload(rawVocab > 0 ? Embedding : Read(checked(c.Vocabulary * dim)), c.Vocabulary, dim);
        }
        catch { Dispose(); throw; }
        float[] Read(int count)
        {
            var values = new float[count]; stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
            if (values.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Checkpoint contains non-finite weights.");
            return values;
        }
        float[][] Norms() => Enumerable.Range(0, Config.Layers).Select(_ => Read(Config.Dimension)).ToArray();
        IMatrix Upload(float[] values, int rows, int columns)
        { var matrix = device.Upload(values, rows, columns); allocations.Add(matrix); return matrix; }
        IMatrix[] Matrices(int rows, int columns) => Enumerable.Range(0, Config.Layers)
            .Select(_ => Upload(Read(checked(rows * columns)), rows, columns)).ToArray();
    }
    public void Dispose() { foreach (var matrix in allocations) matrix.Dispose(); allocations.Clear(); }
}
