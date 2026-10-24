using DeepInfer.Backends.Compute;
namespace DeepInfer.Backends.Speculative;

// The native execution path the synthetic reference must share with the deployed baseline: fused FP32 decode
// attention, FP32 KV/GEMM, managed prefill chunk 32 and prepared scratch 128. Restore rewinds the recurrent/convolution
// state and the logical KV position; fused decode attention keeps no state across tokens (its per-token partials live in a
// scratch buffer that every forward fully rewrites), so no profile change is needed for rollback.
public sealed record DeployedProfile(bool FusedFp32Decode, int ScratchTokens, int PrefillChunkTokens)
{
    public const int ManagedPrefillChunk = 32;
    public static readonly DeployedProfile Baseline = new(true, 128, ManagedPrefillChunk);
    public string DecodeAttention => FusedFp32Decode ? "fused-fp32" : "legacy";
    public static DeployedProfile Parse(string decodeAttention, int scratchTokens, int prefillChunkTokens = ManagedPrefillChunk) => new(
        decodeAttention switch { "fused-fp32" => true, "legacy" => false, _ => throw new ArgumentException("decode-attention must be legacy or fused-fp32.") },
        scratchTokens is 128 or 512 ? scratchTokens : throw new ArgumentException("metal-scratch-tokens must be 128 or 512."),
        prefillChunkTokens == ManagedPrefillChunk ? prefillChunkTokens : throw new ArgumentException("The deployed managed prefill chunk is 32 tokens."));

    // Verifies what the native graph actually runs, not what was requested.
    public void Verify(MetalExecutionProfile native, bool fusedActual, int chunkActual)
    {
        if (native.KvElementBytes != 4 || native.Bf16Gemm)
            throw new InvalidOperationException("Native Metal profile is not the FP32 KV/FP32 GEMM deployment profile.");
        if (native.ScratchTokens != ScratchTokens)
            throw new InvalidOperationException($"Native prefill scratch {native.ScratchTokens} differs from the requested {ScratchTokens}.");
        if (fusedActual != FusedFp32Decode)
            throw new InvalidOperationException($"Native decode attention (fused={fusedActual}) differs from the requested {DecodeAttention}.");
        if (chunkActual != PrefillChunkTokens)
            throw new InvalidOperationException($"Managed prefill chunk {chunkActual} differs from the requested {PrefillChunkTokens}.");
    }
}

// One executed correctness configuration. IgnoreEos=false entries stop at the model's natural end token.
public sealed record MatrixEntry(string Group, int Length, SyntheticPattern Pattern, int Block, bool IgnoreEos, bool Repeated);

public static class SpeculativeMatrix
{
    // Patterns that are distinct at a block size: offset N at or beyond the block is the same as accept-all.
    public static bool Redundant(SyntheticPattern pattern, int block) => pattern.Kind == "offset" && pattern.Offset >= block;

    public static IReadOnlyList<MatrixEntry> Cross(IEnumerable<int> lengths, IEnumerable<SyntheticPattern> patterns, IEnumerable<int> blocks, bool ignoreEos)
    {
        var list = new List<MatrixEntry>();
        foreach (var length in lengths)
            foreach (var pattern in patterns)
                foreach (var block in blocks)
                    if (!Redundant(pattern, block)) list.Add(new("custom", length, pattern, block, ignoreEos, true));
        return list;
    }

    // Practical default: every reject offset at 128 tokens, accept-all / reject-all / mixed at 4096, every token cap
    // 1..block+2 for several block sizes (single execution), and natural end-token stops at several cycle alignments.
    // Reject-all uses block 2 at 4096 because it spends block+2 forwards per emitted token.
    public static IReadOnlyList<MatrixEntry> Default()
    {
        var list = new List<MatrixEntry>();
        foreach (var block in new[] { 8, 4, 2, 1 })
        {
            foreach (var pattern in new[] { "accept-all", "offset:0", "offset:1", "offset:2", "offset:3", "offset:4", "offset:5", "offset:6", "offset:7", "mixed" }.Select(SyntheticPattern.Parse))
                if (!Redundant(pattern, block)) list.Add(new("offsets-128", 128, pattern, block, true, true));
        }
        foreach (var block in new[] { 8, 4 })
            foreach (var name in new[] { "accept-all", "mixed" })
                list.Add(new("patterns-4096", 4096, SyntheticPattern.Parse(name), block, true, true));
        list.Add(new("patterns-4096", 4096, SyntheticPattern.Parse("reject-all"), 2, true, true));
        foreach (var block in new[] { 1, 2, 3, 4, 8 })
            for (var cap = 1; cap <= block + 2; cap++)
                foreach (var name in new[] { "accept-all", "reject-all", "offset:1", "mixed" })
                {
                    var pattern = SyntheticPattern.Parse(name);
                    if (!Redundant(pattern, block)) list.Add(new("caps", cap, pattern, block, true, false));
                }
        foreach (var block in Enumerable.Range(1, 8))
            list.Add(new("natural-eos", 4096, SyntheticPattern.Parse("accept-all"), block, false, false));
        foreach (var name in new[] { "mixed", "offset:2", "reject-all" })
            list.Add(new("natural-eos", 4096, SyntheticPattern.Parse(name), name == "reject-all" ? 2 : 8, false, false));
        return list;
    }
}
