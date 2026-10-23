namespace DeepInfer.Runtime;

/// <summary>Contiguous CUDA compute tiles. A short first tile lets Metal start
/// its suffix while CUDA computes the first full-sized tile.</summary>
public static class PrefixChunkPlan
{
    public static IEnumerable<(int Position, int Count)> Create(int tokens, int chunk, int first)
    {
        if (tokens < 1 || chunk is not (32 or 256 or 512 or 1024 or 2048) ||
            first is not (0 or 128) || first > chunk)
            throw new ArgumentException("Invalid CUDA prefill tile schedule.");
        var position = 0;
        if (first > 0)
        {
            var count = Math.Min(first, tokens);
            yield return (position, count);
            position += count;
        }
        while (position < tokens)
        {
            var count = Math.Min(chunk, tokens - position);
            yield return (position, count);
            position += count;
        }
    }
}
