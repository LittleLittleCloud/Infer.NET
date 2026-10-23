using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
namespace DeepInfer.Runtime;

public sealed record SplitSuffixBatch(int Position, int Count, float[] Hidden, byte[][]? Kv);

public static class SplitSuffixBatches
{
    // Stop exactly at the end of hidden/KV batches. The caller owns the following
    // recurrent-state frames and acknowledgement after the receive pump has joined.
    public static async IAsyncEnumerable<SplitSuffixBatch> Read(
        Func<CancellationToken, Task<SplitWire.Frame>> receive, int tokens, int chunk, bool incremental,
        [EnumeratorCancellation] CancellationToken ct)
    {
        if (tokens is < 1 or > DeepInfer.Core.Protocol.Qwen36Context || chunk is not (32 or 128 or 512))
            throw new ArgumentOutOfRangeException(nameof(tokens), "Invalid suffix batch shape.");
        for (var offset = 0; offset < tokens; offset += chunk)
        {
            var count = Math.Min(chunk, tokens - offset);
            var hidden = new float[count * 2048];
            var kv = incremental ? Enumerable.Range(0, 10).Select(_ => new byte[count * 512 * 4]).ToArray() : null;
            for (var at = 0; at < count; at += 32)
            {
                ct.ThrowIfCancellationRequested();
                var frame = await receive(ct); SplitWire.Validate(frame, 4);
                if (frame.Kind != SplitWire.Hidden || frame.Position != offset + at || frame.Count != Math.Min(32, count - at))
                    throw new InvalidDataException("Noncontiguous hidden stream.");
                MemoryMarshal.Cast<byte, float>(frame.Data).CopyTo(hidden.AsSpan(at * 2048));
                if (kv is not null) for (var layer = 3; layer < 20; layer += 4) for (var tensor = 0; tensor < 2; tensor++)
                {
                    var range = await receive(ct); SplitWire.Validate(range, 4);
                    if (range.Kind != SplitWire.KvRange || range.Layer != layer || range.Tensor != tensor || range.Position != offset + at || range.Count != frame.Count)
                        throw new InvalidDataException("Noncontiguous KV range stream.");
                    range.Data.CopyTo(kv[(layer / 4) * 2 + tensor], at * 512 * 4);
                }
            }
            if (hidden.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Nonfinite hidden.");
            yield return new(offset, count, hidden, kv);
        }
    }
}
