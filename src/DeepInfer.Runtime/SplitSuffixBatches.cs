using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
namespace DeepInfer.Runtime;

public sealed record SplitSuffixBatch(int Position, int Count, float[] Hidden, byte[][]? Kv);

public static class SplitSuffixBatches
{
    private static bool FiniteHidden(byte[] bytes)
    {
        foreach (var value in MemoryMarshal.Cast<byte, float>(bytes.AsSpan()))
            if (!float.IsFinite(value)) return false;
        return true;
    }
    // Stop exactly at the end of hidden/KV batches. The caller owns the following
    // recurrent-state frames and acknowledgement after the receive pump has joined.
    public static async IAsyncEnumerable<SplitSuffixBatch> Read(
        Func<CancellationToken, Task<SplitWire.Frame>> receive, int tokens, int chunk, bool incremental,
        [EnumeratorCancellation] CancellationToken ct)
    {
        if (tokens is < 1 or > DeepInfer.Core.Protocol.Qwen36Context || chunk is not (32 or 128 or 512))
            throw new ArgumentOutOfRangeException(nameof(tokens), "Invalid suffix batch shape.");
        var received = 0; var buffered = 0; var batchPosition = 0;
        var hidden = new float[chunk * 2048];
        byte[][]? kv = incremental ? Enumerable.Range(0, 10).Select(_ => new byte[chunk * 512 * 4]).ToArray() : null;
        while (received < tokens)
        {
            ct.ThrowIfCancellationRequested();
            var frame = await receive(ct); SplitWire.Validate(frame, 4);
            if (frame.Kind != SplitWire.Hidden || frame.Position != received || frame.Count > tokens - received)
                throw new InvalidDataException("Noncontiguous hidden stream.");
            if (!FiniteHidden(frame.Data)) throw new InvalidDataException("Nonfinite hidden.");
            byte[][]? ranges = null;
            if (incremental)
            {
                ranges = new byte[10][];
                for (var layer = 3; layer < 20; layer += 4) for (var tensor = 0; tensor < 2; tensor++)
                {
                    var range = await receive(ct); SplitWire.Validate(range, 4);
                    if (range.Kind != SplitWire.KvRange || range.Layer != layer || range.Tensor != tensor || range.Position != received || range.Count != frame.Count)
                        throw new InvalidDataException("Noncontiguous KV range stream.");
                    ranges[(layer / 4) * 2 + tensor] = range.Data;
                }
            }
            for (var at = 0; at < frame.Count;)
            {
                var take = Math.Min(chunk - buffered, frame.Count - at);
                MemoryMarshal.Cast<byte, float>(frame.Data.AsSpan(at * 2048 * 4, take * 2048 * 4))
                    .CopyTo(hidden.AsSpan(buffered * 2048));
                if (ranges is not null) for (var i = 0; i < 10; i++)
                    ranges[i].AsSpan(at * 512 * 4, take * 512 * 4).CopyTo(kv![i].AsSpan(buffered * 512 * 4));
                at += take; buffered += take;
                if (buffered == chunk || received + at == tokens)
                {
                    var values = buffered == chunk ? hidden : hidden[..(buffered * 2048)];
                    var slices = kv is null ? null : buffered == chunk ? kv : kv.Select(x => x[..(buffered * 512 * 4)]).ToArray();
                    yield return new(batchPosition, buffered, values, slices);
                    batchPosition += buffered; buffered = 0;
                    hidden = new float[chunk * 2048];
                    kv = incremental ? Enumerable.Range(0, 10).Select(_ => new byte[chunk * 512 * 4]).ToArray() : null;
                }
            }
            received += frame.Count;
        }
    }
}
