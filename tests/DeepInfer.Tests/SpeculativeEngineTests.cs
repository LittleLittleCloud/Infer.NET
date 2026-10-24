using System.Text.Json;
using DeepInfer.Backends.Speculative;
using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

// A history-dependent greedy model: the next token depends on every consumed token, so any rejected token that
// leaks into the state (missed restore, wrong replay, double consume) changes every later prediction.
internal sealed class FakeTarget(int prompt, ulong initial = 17, bool brokenRestore = false, int endToken = -1, int endAfter = int.MaxValue) : ISpeculativeTarget
{
    private readonly ulong seed = initial; public ulong State = initial; private readonly int promptLength = prompt; public int Position { get; private set; } = prompt;
    public int Forwards, Captures, Restores, MaxConsumedIndex = -1, Capacity = 1 << 20; public bool Abandoned; public Action<int>? OnConsume;
    private ulong savedState; private int savedPosition = -1;
    public int Consume(int token)
    {
        Assert.False(Abandoned);
        OnConsume?.Invoke(token);
        MaxConsumedIndex = Math.Max(MaxConsumedIndex, Position);
        State = (State ^ (ulong)(token + 1)) * 0x100000001b3ul + (ulong)Position;
        Position++; Forwards++;
        return Predict(State, Position - promptLength);
    }
    // The model "emits" the end token once `endAfter` outputs exist.
    private int Predict(ulong state, int consumedOutputs) => endToken >= 0 && consumedOutputs >= endAfter ? endToken : (int)((state >> 7) % 1000);
    public void EnsureCapacity(int tokens) => Assert.True(tokens <= Capacity);
    public double Capture() { Captures++; savedState = State; savedPosition = Position; return 0.001; }
    public double Restore()
    {
        Assert.True(savedPosition >= 0);
        Restores++;
        if (!brokenRestore) State = savedState;
        Position = savedPosition; return 0.002;
    }
    public void Abandon() => Abandoned = true;
    // A reset after an abandoned request recovers a fresh, usable target (the service re-prefills the graph).
    public void Reset() { Abandoned = false; State = seed; Position = promptLength; savedPosition = -1; MaxConsumedIndex = -1; }
}

public sealed class SpeculativeEngineTests
{
    private static (int[] Tokens, ulong State) Baseline(int prompt, int length, ulong seed = 17)
    {
        var target = new FakeTarget(prompt, seed);
        var pending = (int)((target.State >> 7) % 1000); // arbitrary deterministic first prediction
        var tokens = new List<int>();
        for (var i = 0; i < length; i++)
        {
            tokens.Add(pending);
            if (i + 1 < length) pending = target.Consume(pending);
        }
        return ([.. tokens], target.State);
    }
    private static Task<SpeculativeRun> Run(FakeTarget target, int pending, int length, string pattern, int block, IProposalOracle? oracle = null,
        bool ignoreEos = true, Func<int, bool>? isEnd = null, int[]? checkpoints = null, Action<int, int>? hook = null,
        CancellationToken ct = default, Func<int, ValueTask>? emit = null)
        => SpeculativeEngine.Run(target, pending, length, ignoreEos, isEnd ?? (_ => false), SyntheticPattern.Parse(pattern), block,
            oracle ?? new RolloutOracle(), ct, emit, checkpoints, hook);
    private static int First(FakeTarget target) => (int)((target.State >> 7) % 1000);

    public static IEnumerable<object[]> Matrix()
    {
        foreach (var block in Enumerable.Range(1, 8))
            foreach (var pattern in new[] { "accept-all", "reject-all", "offset:0", "offset:1", "offset:2", "offset:3", "offset:4", "offset:5", "offset:6", "offset:7", "mixed" })
                yield return [block, pattern];
    }

    [Theory, MemberData(nameof(Matrix))]
    public async Task EveryPatternAndBlockSizeReproducesTheNonSpeculativeStream(int block, string pattern)
    {
        foreach (var length in new[] { 1, 2, 3, 4, 9, 10, 33, 100 })
        {
            var expected = Baseline(5, length);
            var target = new FakeTarget(5);
            var run = await Run(target, First(target), length, pattern, block, new ReferenceOracle(expected.Tokens));
            Assert.Equal(expected.Tokens, run.Output);
            Assert.Equal("length", run.FinishReason);
            Assert.Equal(expected.State, target.State); // same state, not merely same tokens
            Assert.Equal(5 + length - 1, target.Position); // last emitted token is never consumed
            Assert.True(target.MaxConsumedIndex <= 5 + length - 2);
        }
    }

    [Fact]
    public async Task RolloutAndSavedBaselineOraclesAgree()
    {
        var expected = Baseline(3, 50);
        var a = new FakeTarget(3); var b = new FakeTarget(3);
        var saved = await Run(a, First(a), 50, "mixed", 5, new ReferenceOracle(expected.Tokens));
        var rolled = await Run(b, First(b), 50, "mixed", 5, new RolloutOracle());
        Assert.Equal(saved.Output, rolled.Output);
        Assert.Equal(a.State, b.State);
        Assert.Equal(0, saved.Diagnostics.OracleForwards);
        Assert.True(rolled.Diagnostics.OracleForwards > 0);
        Assert.Equal(saved.Diagnostics.Proposed, rolled.Diagnostics.Proposed);
    }

    [Fact]
    public async Task CountersMatchTheCycleStructureForRejectAllAndAcceptAll()
    {
        var expected = Baseline(4, 40);
        var rejectTarget = new FakeTarget(4);
        var reject = await Run(rejectTarget, First(rejectTarget), 40, "reject-all", 4, new ReferenceOracle(expected.Tokens));
        var d = reject.Diagnostics;
        Assert.Equal(0, d.Accepted);
        Assert.Equal(d.Corrected, d.Verified); // exactly the wrong first proposal is verified per cycle
        Assert.Equal(0, d.Bonus);
        Assert.Equal(d.Corrected, rejectTarget.Restores);
        Assert.Equal(d.Corrected, d.ReplayForwards); // replay of the single committed input per rejected cycle
        Assert.Equal(40, d.Emitted);
        Assert.All(reject.Diagnostics.Detail!.Where(c => c.Proposed > 0), c =>
        { Assert.Equal(0, c.RejectOffset); Assert.Equal(1, c.Emitted); Assert.Equal(1, c.Corrected); Assert.Equal(c.Proposed + 1, c.VerifyForwards); });

        var acceptTarget = new FakeTarget(4);
        var accept = await Run(acceptTarget, First(acceptTarget), 40, "accept-all", 4, new ReferenceOracle(expected.Tokens));
        var a = accept.Diagnostics;
        Assert.Equal(0, a.Corrected); Assert.Equal(0, acceptTarget.Restores); Assert.Equal(0, a.ReplayForwards);
        Assert.Equal(a.Proposed, a.Accepted); Assert.Equal(a.Proposed, a.Verified);
        Assert.Equal(40, a.Emitted);
        Assert.All(accept.Diagnostics.Detail!.Where(c => c.Proposed > 0), c =>
        { Assert.Equal(-1, c.RejectOffset); Assert.Equal(1, c.Bonus); Assert.Equal(c.Proposed + 1, c.Emitted); });
        Assert.True(a.Cycles < d.Cycles); // accepted proposals really advance several tokens per cycle
        Assert.Equal(a.Emitted, a.Detail!.Sum(c => c.Emitted) + 1); // the final pending token is emitted without a forward
    }

    [Theory, InlineData(0), InlineData(1), InlineData(2), InlineData(3)]
    public async Task RejectOffsetControlsAcceptedPrefixAndReplayLength(int offset)
    {
        var expected = Baseline(2, 30);
        var target = new FakeTarget(2);
        var run = await Run(target, First(target), 30, $"offset:{offset}", 4, new ReferenceOracle(expected.Tokens));
        foreach (var cycle in run.Diagnostics.Detail!.Where(c => c.Proposed > 0))
        {
            if (offset < cycle.Proposed)
            {
                Assert.Equal(offset, cycle.Accepted); Assert.Equal(offset, cycle.RejectOffset);
                Assert.Equal(offset + 1, cycle.Verified); Assert.Equal(offset + 1, cycle.ReplayForwards); Assert.Equal(1, cycle.Corrected);
            }
            else { Assert.Equal(cycle.Proposed, cycle.Accepted); Assert.Equal(0, cycle.ReplayForwards); Assert.Equal(1, cycle.Bonus); }
            Assert.Equal(1 + cycle.Accepted, cycle.Emitted);
        }
        Assert.Equal(expected.Tokens, run.Output);
    }

    [Fact]
    public async Task CheckpointsLandOnCycleBoundariesWithBaselineEquivalentState()
    {
        const int length = 61;
        var oracleTokens = Baseline(6, length).Tokens; var oracle = new ReferenceOracle(oracleTokens);
        var states = new Dictionary<int, ulong>();
        {
            var t = new FakeTarget(6); var pending = First(t);
            for (var i = 0; i < length; i++) { states[i] = t.State; if (i + 1 < length) pending = t.Consume(pending); }
        }
        foreach (var pattern in new[] { "accept-all", "reject-all", "mixed" })
        {
            var target = new FakeTarget(6); var seen = new List<int>();
            await Run(target, First(target), length, pattern, 8, oracle, checkpoints: [0, 7, 8, 9, 30, 60], hook: (emitted, pending) =>
            {
                seen.Add(emitted); Assert.Equal(states[emitted], target.State); Assert.Equal(6 + emitted, target.Position);
                Assert.Equal(oracleTokens[emitted], pending);
            });
            Assert.Equal([0, 7, 8, 9, 30, 60], seen);
        }
    }

    [Fact]
    public async Task PendingEndTokenIsNeverEmittedNorConsumed()
    {
        var target = new FakeTarget(3);
        var run = await Run(target, 77, 10, "mixed", 4, ignoreEos: false, isEnd: x => x == 77);
        Assert.Empty(run.Output); Assert.Equal("eos", run.FinishReason);
        Assert.Equal(0, target.Forwards); Assert.Equal(0, target.Captures);
        var ignored = await Run(new FakeTarget(3), 77, 4, "accept-all", 4, ignoreEos: true, isEnd: x => x == 77);
        Assert.Equal(77, ignored.Output[0]); Assert.Equal(4, ignored.Output.Length); Assert.Equal("length", ignored.FinishReason);
    }

    [Theory, InlineData("accept-all"), InlineData("reject-all"), InlineData("mixed")]
    public async Task EndTokenInsideAnAcceptedBlockStopsBeforeTheEndToken(string pattern)
    {
        // The model produces token 999 after 12 outputs; emitting stops exactly there, never beyond.
        var baselineTarget = new FakeTarget(2, endToken: 999, endAfter: 12);
        var pending = First(baselineTarget); var baseline = new List<int>();
        while (pending != 999) { baseline.Add(pending); pending = baselineTarget.Consume(pending); }
        Assert.True(baseline.Count >= 12);
        var target = new FakeTarget(2, endToken: 999, endAfter: 12);
        var run = await Run(target, First(target), 100, pattern, 8, new RolloutOracle(), ignoreEos: false, isEnd: x => x == 999);
        Assert.Equal(baseline, run.Output); Assert.Equal("eos", run.FinishReason);
        Assert.DoesNotContain(999, run.Output);
    }

    [Theory, InlineData(1), InlineData(2), InlineData(3), InlineData(4)]
    public async Task TokenCapNeverConsumesPastTheLastOutputPosition(int cap)
    {
        foreach (var pattern in new[] { "accept-all", "reject-all", "mixed" })
        {
            var target = new FakeTarget(10);
            var run = await Run(target, First(target), cap, pattern, 8);
            Assert.Equal(cap, run.Output.Length);
            Assert.True(target.MaxConsumedIndex <= 10 + cap - 2);
            Assert.Equal(10 + cap - 1, target.Position);
            if (cap == 1) { Assert.Equal(0, target.Forwards); Assert.Equal(0, target.Captures); }
        }
    }

    [Fact]
    public async Task ContextCapacityIsRequestedBeforeEveryBlock()
    {
        var target = new FakeTarget(100) { Capacity = 100 + 20 };
        var run = await Run(target, First(target), 21, "mixed", 8); // consumes indices 100..119 only
        Assert.Equal(21, run.Output.Length); Assert.Equal(119, target.MaxConsumedIndex);
        var small = new FakeTarget(100) { Capacity = 100 + 10 };
        await Assert.ThrowsAnyAsync<Exception>(() => Run(small, First(small), 21, "mixed", 8));
        Assert.True(small.Abandoned);
    }

    [Fact]
    public async Task CancellationAbandonsTheTargetAndStopsEmission()
    {
        var expected = Baseline(3, 40);
        using var source = new CancellationTokenSource(); var emitted = 0;
        var target = new FakeTarget(3);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Run(target, First(target), 40, "mixed", 4, new ReferenceOracle(expected.Tokens), ct: source.Token,
            emit: _ => { if (++emitted == 6) source.Cancel(); return ValueTask.CompletedTask; }));
        Assert.True(target.Abandoned); Assert.True(emitted is >= 6 and < 40);

        // Cancelling in the middle of a verified block, after rejected tokens were consumed.
        using var inner = new CancellationTokenSource(); var forwards = 0;
        var mid = new FakeTarget(3) { OnConsume = _ => { if (++forwards == 7) inner.Cancel(); } };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Run(mid, First(mid), 40, "reject-all", 8, ct: inner.Token));
        Assert.True(mid.Abandoned);
        using var already = new CancellationTokenSource(); already.Cancel();
        var none = new FakeTarget(3);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Run(none, First(none), 5, "mixed", 4, ct: already.Token));
        Assert.Equal(0, none.Forwards);
    }

    [Fact]
    public async Task BrokenRestoreIsDetectedByReplayCrossCheck()
    {
        var expected = Baseline(3, 20);
        var target = new FakeTarget(3, brokenRestore: true);
        var error = await Assert.ThrowsAsync<SpeculativeStateException>(() => Run(target, First(target), 20, "reject-all", 4, new ReferenceOracle(expected.Tokens)));
        Assert.Contains("Replay prediction", error.Message);
        Assert.True(target.Abandoned);
    }

    [Fact]
    public async Task DivergingSavedBaselineIsRejectedNotMasked()
    {
        var wrong = Baseline(3, 20).Tokens.ToArray(); wrong[1] = (wrong[1] + 1) % 1000;
        var target = new FakeTarget(3);
        var error = await Assert.ThrowsAsync<SpeculativeStateException>(() => Run(target, First(target), 20, "accept-all", 4, new ReferenceOracle(wrong)));
        Assert.Contains("differs", error.Message);
    }

    [Theory, InlineData(0), InlineData(9), InlineData(-1)]
    public async Task BlockSizeIsBoundedToOneThroughEight(int value)
        => await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => Run(new FakeTarget(1), 1, 5, "mixed", value));
    [Theory, InlineData(0), InlineData(-1)]
    public async Task OutputLengthMustBePositive(int value)
        => await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => Run(new FakeTarget(1), 1, value, "mixed", 4));

    [Fact]
    public void PatternParserAcceptsOnlyTheDocumentedGrammar()
    {
        Assert.Equal("accept-all", SyntheticPattern.Parse("accept-all").Name);
        Assert.Equal("offset:7", SyntheticPattern.Parse("offset:7").Name);
        foreach (var bad in new[] { "", "off", "offset", "offset:", "offset:8", "offset:-1", "offset:+1", "offset:1.5", "ACCEPT-ALL", "reject", "offset: 1", "mixed " })
            Assert.Throws<ArgumentException>(() => SyntheticPattern.Parse(bad));
        var mixed = SyntheticPattern.Parse("mixed");
        var offsets = Enumerable.Range(0, 24).Select(c => mixed.RejectOffset(c, 4)).ToArray();
        Assert.All(offsets, x => Assert.InRange(x, 0, 4));
        Assert.Contains(4, offsets); Assert.Contains(0, offsets); Assert.True(offsets.Distinct().Count() >= 3);
        Assert.Equal(3, SyntheticPattern.Parse("offset:5").RejectOffset(0, 3)); // beyond the block means accept all
        Assert.Equal(0, SyntheticPattern.Parse("reject-all").RejectOffset(9, 8)); Assert.Equal(8, SyntheticPattern.Parse("accept-all").RejectOffset(9, 8));
    }

    [Fact]
    public void CorruptedProposalsNeverEqualTruthAndStayInVocabulary()
    {
        for (var truth = 0; truth < 248320; truth += 997)
            for (var cycle = 0; cycle < 40; cycle++)
                for (var index = 0; index < 8; index++)
                { var x = SpeculativeEngine.Corrupt(truth, cycle, index); Assert.NotEqual(truth, x); Assert.InRange(x, 0, 248319); }
        Assert.NotEqual(248319, SpeculativeEngine.Corrupt(248319, 3, 4)); // wraps instead of overflowing
    }

    [Fact]
    public void DiagnosticsAreOmittedFromOffResultsAndStaySeparateFromTimings()
    {
        var plain = new InferenceResult("r", "w", Protocol.Qwen36Model, "metal", "x", [], false, 1);
        Assert.DoesNotContain("speculative", JsonSerializer.Serialize(plain, RpcJson.Default.InferenceResult));
        var resources = new SpeculativeResources(12.5, 4096, 1 << 20, 1 << 30, 1 << 26, "fused-fp32", 128, 32);
        var diagnostics = new SpeculativeDiagnostics("synthetic-proposal-correctness-reference", "mixed", 4, "target-rollout", 1, 4, 3, 3, 2, 1, 0, 3, 0, 5, 2, 5, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, "length", SpeculativeEngine.Note, null, resources);
        var enabled = plain with { Speculative = diagnostics, Timings = null };
        var json = JsonSerializer.Serialize(enabled, RpcJson.Default.InferenceResult);
        Assert.Contains("\"speculative\"", json); Assert.Contains("\"timings\":null", json);
        var back = JsonSerializer.Deserialize(json, RpcJson.Default.InferenceResult)!.Speculative!;
        Assert.Equal(diagnostics.Corrected, back.Corrected); Assert.Equal(resources, back.Resources);
        Assert.Contains("not a performance", SpeculativeEngine.Note);
    }
}
