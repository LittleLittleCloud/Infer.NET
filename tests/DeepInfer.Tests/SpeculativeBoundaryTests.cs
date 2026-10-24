using System.Text.Json;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Speculative;
using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

public sealed class SpeculativeBoundaryTests
{
    private const int End = 1000; // outside the 0..999 range of ordinary predictions

    private sealed record Serial(int[] Tokens, string Finish, int Position, ulong State, int Pending);

    // Independent non-speculative reference: the exact serial decode loop with the same stop rules.
    private static Serial Reference(int prompt, int cap, int endAfter, bool ignoreEos = false)
    {
        var target = new FakeTarget(prompt, endToken: End, endAfter: endAfter);
        var pending = (int)((target.State >> 7) % 1000); var tokens = new List<int>(); var finish = "length";
        while (true)
        {
            if (!ignoreEos && pending == End) { finish = "eos"; break; }
            tokens.Add(pending);
            if (tokens.Count >= cap) break;
            pending = target.Consume(pending);
        }
        return new([.. tokens], finish, target.Position, target.State, pending);
    }

    private static Task<SpeculativeRun> Run(FakeTarget target, int cap, string pattern, int block, bool ignoreEos = false)
        => SpeculativeEngine.Run(target, (int)((target.State >> 7) % 1000), cap, ignoreEos, x => x == End, SyntheticPattern.Parse(pattern), block,
            new RolloutOracle(), CancellationToken.None, null, null, null, keepDetail: true);

    public static IEnumerable<object[]> Patterns()
    {
        foreach (var block in Enumerable.Range(1, 8))
            foreach (var pattern in new[] { "accept-all", "reject-all", "offset:0", "offset:1", "offset:2", "offset:3", "offset:4", "offset:5", "offset:6", "offset:7", "mixed" })
                yield return [block, pattern];
    }

    [Theory, MemberData(nameof(Patterns))]
    public async Task EndTokenAtEveryOffsetLeavesTheTargetAtTheSerialBoundary(int block, string pattern)
    {
        // endAfter 1..24 places the model's end token at every offset of every cycle alignment, before and after reject offsets.
        for (var endAfter = 1; endAfter <= 24; endAfter++)
        {
            var expected = Reference(4, 200, endAfter);
            Assert.Equal("eos", expected.Finish);
            var target = new FakeTarget(4, endToken: End, endAfter: endAfter);
            var run = await Run(target, 200, pattern, block);
            var label = $"block={block} pattern={pattern} endAfter={endAfter}";
            Assert.True(expected.Tokens.SequenceEqual(run.Output), label);
            Assert.Equal("eos", run.FinishReason);
            Assert.True(expected.Position == target.Position, $"{label}: position {target.Position} vs {expected.Position}");
            Assert.True(expected.State == target.State, $"{label}: state");
            Assert.Equal(expected.Position, run.TerminalPosition);
            Assert.Equal(End, run.FinalPending);
            Assert.Equal(4 + run.Output.Length, target.Position); // natural end: every emitted token is consumed
            Assert.True(run.Diagnostics.Matched >= run.Diagnostics.Accepted);
            Assert.Equal(run.Output.Length, run.Diagnostics.Emitted);
        }
    }

    [Theory, MemberData(nameof(Patterns))]
    public async Task EveryCapUpToBlockPlusTwoMatchesTheSerialTerminalState(int block, string pattern)
    {
        foreach (var endAfter in new[] { int.MaxValue, 2, 5, 9 })
            for (var cap = 1; cap <= block + 2; cap++)
            {
                var expected = Reference(3, cap, endAfter);
                var target = new FakeTarget(3, endToken: End, endAfter: endAfter);
                var run = await Run(target, cap, pattern, block);
                var label = $"block={block} pattern={pattern} cap={cap} endAfter={endAfter}";
                Assert.True(expected.Tokens.SequenceEqual(run.Output), label);
                Assert.True(expected.Finish == run.FinishReason, label);
                Assert.True(expected.Position == target.Position, $"{label}: position {target.Position} vs {expected.Position}");
                Assert.True(expected.State == target.State, $"{label}: state");
                Assert.Equal(expected.Pending, run.FinalPending);
                Assert.False(target.Abandoned);
            }
    }

    [Fact]
    public async Task MatchedAndCommittedTokensAreDistinguishedWhenTheEndTokenTruncatesAnAcceptedBlock()
    {
        // Prompt 2, the end token is the 4th output: outputs 0..3 are real, the pending end token is output 4.
        var target = new FakeTarget(2, endToken: End, endAfter: 4);
        var run = await Run(target, 100, "accept-all", 8);
        var cycle = Assert.Single(run.Diagnostics.Detail!, c => c.Proposed > 0);
        Assert.True(cycle.EosTruncated);
        Assert.Equal(cycle.Proposed, cycle.Matched);          // every proposal matched the model...
        Assert.True(cycle.Accepted < cycle.Matched);          // ...but only those before the end token were committed
        Assert.Equal(0, cycle.Bonus); Assert.Equal(0, cycle.Corrected); // no bonus/correction after a real end token
        Assert.Equal(1 + cycle.Accepted, cycle.Emitted);
        Assert.Equal(cycle.Accepted + 1, cycle.CommittedInputs);
        Assert.Equal(cycle.CommittedInputs, cycle.ReplayForwards);     // restore + replay of only the committed prefix
        Assert.Equal(1, run.Diagnostics.EosTruncations); Assert.Equal(2, target.Restores); // one oracle rollout restore plus the commit restore
        Assert.Equal(2 + run.Output.Length, target.Position);

        // A rejection before the end token is an ordinary correction; the end token arrives as a natural pending token.
        var early = new FakeTarget(2, endToken: End, endAfter: 6);
        var second = await Run(early, 100, "offset:1", 8);
        var first = second.Diagnostics.Detail!.First(c => c.Proposed > 0);
        Assert.False(first.EosTruncated); Assert.Equal(1, first.Corrected); Assert.Equal(1, first.Accepted);
        Assert.Equal(Reference(2, 100, 6).Tokens, second.Output);
        Assert.Equal(End, second.FinalPending);
    }

    [Fact]
    public async Task IgnoreEosDecodesThroughTheEndTokenAndLengthLeavesTheLastTokenUnconsumed()
    {
        var expected = Reference(4, 30, 3, ignoreEos: true);
        Assert.Equal("length", expected.Finish);
        var target = new FakeTarget(4, endToken: End, endAfter: 3);
        var run = await Run(target, 30, "mixed", 5, ignoreEos: true);
        Assert.Equal(expected.Tokens, run.Output); Assert.Equal(expected.State, target.State);
        Assert.Equal(4 + 30 - 1, target.Position); Assert.Equal(0, run.Diagnostics.EosTruncations);
    }

    [Fact]
    public async Task CancellationAbandonsTheTargetAndAResetRecoversAnExactRun()
    {
        var expected = Reference(3, 40, int.MaxValue);
        using var source = new CancellationTokenSource(); var seen = 0;
        var target = new FakeTarget(3, endToken: End);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => SpeculativeEngine.Run(target, (int)((target.State >> 7) % 1000), 40, false, x => x == End,
            SyntheticPattern.Parse("mixed"), 6, new RolloutOracle(), source.Token, _ => { if (++seen == 9) source.Cancel(); return ValueTask.CompletedTask; }, null, null));
        Assert.True(target.Abandoned);
        Assert.ThrowsAny<Exception>(() => target.Consume(1)); // abandoned targets refuse further work
        target.Reset();
        var run = await Run(target, 40, "reject-all", 4);
        Assert.Equal(expected.Tokens, run.Output); Assert.Equal(expected.State, target.State); Assert.Equal(expected.Position, target.Position);
    }

    [Fact]
    public void DeployedProfileIsFusedFp32ScratchOneTwentyEightChunkThirtyTwo()
    {
        var baseline = DeployedProfile.Baseline;
        Assert.True(baseline.FusedFp32Decode); Assert.Equal(128, baseline.ScratchTokens); Assert.Equal(32, baseline.PrefillChunkTokens);
        Assert.Equal(baseline, DeployedProfile.Parse("fused-fp32", 128));
        Assert.False(DeployedProfile.Parse("legacy", 512).FusedFp32Decode); // legacy stays selectable, fused is not rejected
        Assert.Equal("fused-fp32", baseline.DecodeAttention);
        Assert.Throws<ArgumentException>(() => DeployedProfile.Parse("fused", 128));
        Assert.Throws<ArgumentException>(() => DeployedProfile.Parse("fused-fp32", 256));
        Assert.Throws<ArgumentException>(() => DeployedProfile.Parse("fused-fp32", 128, 4));
    }

    [Fact]
    public void DeployedProfileVerificationChecksTheNativeGraphNotTheRequest()
    {
        var native = new MetalExecutionProfile(4, false, true, true, true, true, true, true, 128, true, true, 0, 0, 0);
        DeployedProfile.Baseline.Verify(native, true, 32);
        Assert.Throws<InvalidOperationException>(() => DeployedProfile.Baseline.Verify(native, false, 32));   // fused silently not applied
        Assert.Throws<InvalidOperationException>(() => DeployedProfile.Baseline.Verify(native, true, 4));     // legacy chunk
        Assert.Throws<InvalidOperationException>(() => DeployedProfile.Baseline.Verify(native with { ScratchTokens = 512 }, true, 32));
        Assert.Throws<InvalidOperationException>(() => DeployedProfile.Baseline.Verify(native with { KvElementBytes = 2 }, true, 32));
        Assert.Throws<InvalidOperationException>(() => DeployedProfile.Baseline.Verify(native with { Bf16Gemm = true }, true, 32));
    }

    [Fact]
    public void DefaultMatrixCoversEveryOffsetAtOneTwentyEightAndAllPatternsAtFourThousandNinetySix()
    {
        var entries = SpeculativeMatrix.Default();
        var at128 = entries.Where(e => e.Length == 128 && e.Group == "offsets-128").ToArray();
        for (var offset = 0; offset < 8; offset++)
            Assert.Contains(at128, e => e.Pattern.Name == $"offset:{offset}" && e.Block == 8 && e.Repeated);
        Assert.Contains(at128, e => e.Pattern.Name == "accept-all"); Assert.Contains(at128, e => e.Pattern.Name == "mixed");
        Assert.DoesNotContain(at128, e => SpeculativeMatrix.Redundant(e.Pattern, e.Block));
        var long4096 = entries.Where(e => e.Length == 4096 && e.Group == "patterns-4096").ToArray();
        Assert.Contains(long4096, e => e.Pattern.Name == "accept-all"); Assert.Contains(long4096, e => e.Pattern.Name == "mixed");
        Assert.Contains(long4096, e => e.Pattern.Name == "reject-all"); Assert.All(long4096, e => Assert.True(e.Repeated));
        Assert.DoesNotContain(entries, e => e.Length == 128 && e.Pattern.Name == "reject-all" && e.Block > 2 && e.Group != "caps");
        foreach (var block in new[] { 1, 2, 3, 4, 8 })
            for (var cap = 1; cap <= block + 2; cap++)
                Assert.Contains(entries, e => e.Group == "caps" && e.Block == block && e.Length == cap && e.Pattern.Name == "accept-all" && !e.Repeated);
        Assert.Contains(entries, e => e.Group == "natural-eos" && !e.IgnoreEos);
        Assert.Equal(entries.Count, entries.Select(e => (e.Group, e.Length, e.Pattern.Name, e.Block, e.IgnoreEos)).Distinct().Count());
        // Full-length executions per case stay bounded so the matrix needs no enormous model loading.
        Assert.True(entries.Count(e => e.Repeated) * 4 + entries.Count(e => !e.Repeated) < 400);
    }

    [Fact]
    public void CustomCrossSkipsRedundantOffsetsAndKeepsRequestedRuns()
    {
        var entries = SpeculativeMatrix.Cross([128], [SyntheticPattern.Parse("offset:3"), SyntheticPattern.Parse("mixed")], [2, 4], true);
        Assert.DoesNotContain(entries, e => e.Pattern.Name == "offset:3" && e.Block == 2);
        Assert.Contains(entries, e => e.Pattern.Name == "offset:3" && e.Block == 4);
        Assert.Equal(3, entries.Count); Assert.All(entries, e => { Assert.True(e.IgnoreEos); Assert.True(e.Repeated); });
    }

    [Fact]
    public void ResourcesAreSeparateFromTimingsAndSerializeWithoutNonFiniteNumbers()
    {
        var resources = new SpeculativeResources(8.25, 4096, 4096L * 512 * 4 * 10, 2L << 30, 65_757_000, "fused-fp32", 128, 32);
        var plain = new InferenceResult("r", "w", Protocol.Qwen36Model, "metal", "x", [], false, 1);
        var diagnostics = new SpeculativeDiagnostics("synthetic-proposal-correctness-reference", "mixed", 4, "target-rollout", 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "length", SpeculativeEngine.Note, null, resources);
        var json = JsonSerializer.Serialize(plain with { Speculative = diagnostics, Timings = null }, RpcJson.Default.InferenceResult);
        Assert.Contains("\"timings\":null", json); Assert.DoesNotContain("NaN", json); Assert.DoesNotContain("Infinity", json);
        Assert.Contains("\"snapshotHostBytes\":65757000", json); Assert.Contains("\"prefillMs\":8.25", json);
    }
}
