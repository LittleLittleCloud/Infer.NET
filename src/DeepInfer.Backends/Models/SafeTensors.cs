using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
namespace DeepInfer.Backends.Models;

// No pickle, Python code or dynamic deserialization: only a bounded JSON header and typed tensor bytes.
public sealed class SafeTensors : IDisposable
{
    public sealed record Tensor(string Name, string Dtype, int[] Shape, long Start, long Bytes);
    private readonly FileStream stream;
    private readonly long dataStart;
    private readonly Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
    public SafeTensors(string path)
    {
        stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1048576, FileOptions.RandomAccess);
        try
        {
            Span<byte> lengthBytes = stackalloc byte[8]; stream.ReadExactly(lengthBytes);
            var headerLength = BinaryPrimitives.ReadUInt64LittleEndian(lengthBytes);
            if (headerLength is < 2 or > 16777216 || headerLength > (ulong)(stream.Length - 8)) throw new InvalidDataException("Invalid safetensors header length.");
            var header = new byte[(int)headerLength]; stream.ReadExactly(header); dataStart = checked(8 + (long)headerLength);
            using var json = JsonDocument.Parse(header);
            foreach (var entry in json.RootElement.EnumerateObject())
            {
                if (entry.Name == "__metadata__") continue;
                var v = entry.Value; var dtype = v.GetProperty("dtype").GetString() ?? "";
                var elementBytes = dtype switch { "BF16" => 2, "F32" or "U32" => 4, _ => throw new NotSupportedException($"Unsupported safetensors dtype '{dtype}'.") };
                var shape = v.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32()).ToArray();
                if (shape.Length is < 1 or > 8 || shape.Any(d => d < 1)) throw new InvalidDataException("Invalid tensor shape.");
                long elements = 1; foreach (var dim in shape) elements = checked(elements * dim);
                var offsets = v.GetProperty("data_offsets").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                if (offsets.Length != 2 || offsets[0] < 0 || offsets[1] < offsets[0] || offsets[1] > stream.Length - dataStart ||
                    offsets[1] - offsets[0] != checked(elements * elementBytes) || elements > int.MaxValue)
                    throw new InvalidDataException("Invalid tensor offsets/byte length.");
                if (!tensors.TryAdd(entry.Name, new(entry.Name, dtype, shape, dataStart + offsets[0], offsets[1] - offsets[0])))
                    throw new InvalidDataException("Duplicate tensor name.");
            }
            long previousEnd = dataStart;
            foreach (var t in tensors.Values.OrderBy(x => x.Start))
            {
                if (t.Start < previousEnd) throw new InvalidDataException("Overlapping tensors.");
                previousEnd = checked(t.Start + t.Bytes);
            }
        }
        catch { stream.Dispose(); throw; }
    }
    public Tensor Get(string name, params int[] shape)
    {
        if (!tensors.TryGetValue(name, out var tensor) || !tensor.Shape.SequenceEqual(shape))
            throw new InvalidDataException($"Missing tensor or shape mismatch: {name}.");
        return tensor;
    }
    public ushort[] ReadBFloat16(Tensor tensor)
    {
        if (tensor.Dtype != "BF16") throw new InvalidDataException("BF16 tensor required.");
        var values = new ushort[checked((int)(tensor.Bytes / 2))]; stream.Position = tensor.Start;
        stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan())); return values;
    }
    public void ReadInto(Tensor tensor, Span<byte> destination)
    {
        if (!tensors.TryGetValue(tensor.Name, out var owned) || owned != tensor || destination.Length != tensor.Bytes)
            throw new InvalidDataException("Tensor ownership or destination length mismatch.");
        stream.Position = tensor.Start;
        stream.ReadExactly(destination);
    }
    // Exact byte range of a tensor; lets CPU oracles read one embedding row without loading the whole matrix.
    public void ReadBytesRange(Tensor tensor, long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > tensor.Bytes) throw new InvalidDataException("Invalid byte range.");
        stream.Position = tensor.Start + offset;
        stream.ReadExactly(destination);
    }
    // Exact BF16 element range expanded to FP32 (a 16-bit shift); used by CPU oracles, never by inference.
    public void ReadBFloat16Range(Tensor tensor, long element, Span<float> destination)
    {
        if (tensor.Dtype != "BF16" || element < 0 || (element + destination.Length) * 2 > tensor.Bytes) throw new InvalidDataException("Invalid BF16 range.");
        var raw = new ushort[destination.Length]; stream.Position = tensor.Start + element * 2;
        stream.ReadExactly(MemoryMarshal.AsBytes(raw.AsSpan()));
        for (var i = 0; i < raw.Length; i++) destination[i] = BitConverter.Int32BitsToSingle(raw[i] << 16);
    }
    public float[] ReadFloat(Tensor tensor)
    {
        float[] values;
        if (tensor.Dtype == "F32")
        {
            values = new float[checked((int)(tensor.Bytes / 4))]; stream.Position = tensor.Start;
            stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            var source = ReadBFloat16(tensor); values = new float[source.Length];
            for (var i = 0; i < values.Length; i++) values[i] = BitConverter.Int32BitsToSingle(source[i] << 16);
        }
        if (values.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Non-finite tensor values.");
        return values;
    }
    public string Sha256Hex(Tensor tensor)
    {
        if (!tensors.TryGetValue(tensor.Name, out var owned) || owned != tensor) throw new InvalidDataException("Tensor ownership mismatch.");
        using var hash = System.Security.Cryptography.IncrementalHash.CreateHash(System.Security.Cryptography.HashAlgorithmName.SHA256);
        var buffer = new byte[1 << 20]; stream.Position = tensor.Start;
        for (var left = tensor.Bytes; left > 0;)
        {
            var n = (int)Math.Min(buffer.Length, left);
            stream.ReadExactly(buffer, 0, n); hash.AppendData(buffer, 0, n); left -= n;
        }
        return Convert.ToHexStringLower(hash.GetHashAndReset());
    }
    public void Dispose() => stream.Dispose();
}
