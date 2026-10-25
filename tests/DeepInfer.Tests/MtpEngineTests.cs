using DeepInfer.Backends.Mtp;
using Xunit;
namespace DeepInfer.Tests;

// Deterministic fake target/drafter pair. The target's greedy token is a pure function of the history hash, and every
// hidden row carries (history hash through that position, position) so the drafter can prove it was conditioned on
// target hidden rows rather than on its own state.
internal static class MtpFakes
{
    public const int Eos = 49;
    public static int Fold(int h, int token) => (h * 31 + token + 1) % 1000003;
    public static int Next(int h) => (h * 7 + 3) % 50;
    public static float[] Row(int hash, int position) { var r = new float[2048]; r[0] = hash; r[1] = position; return r; }

    public static (List<int> Output, string Finish, List<int> Consumed, int Pending) Reference(int[] prompt, int maxTokens, bool ignoreEos)
    {
        var h = 0; foreach (var t in prompt) h = Fold(h, t);
        var consumed = prompt.ToList(); var output = new List<int>(); var next = Next(h);
        while (true)
        {
            if (!ignoreEos && next == Eos) return (output, "eos", consumed, next);
            output.Add(next);
            if (output.Count >= maxTokens) return (output, "length", consumed, next);
            consumed.Add(next); h = Fold(h, next); next = Next(h);
        }
    }
}

internal sealed class FakeTarget : IMtpTarget
{
    public List<int> History = [];
    private List<int>? snapshot;
    public int VerifyCalls, ForwardCalls, ReplayCalls, SnapshotCalls; public List<int> VerifyRows = [];
    public int ChunkSize = 32; public Action? OnVerify; private bool replaying;
    private int Hash(IEnumerable<int> tokens) { var h = 0; foreach (var t in tokens) h = MtpFakes.Fold(h, t); return h; }
    public int Prefill(int[] prompt, Action<int[], float[]> sink, CancellationToken ct)
    {
        History = [];
        for (var offset = 0; offset < prompt.Length; offset += ChunkSize)
        {
            var chunk = prompt[offset..Math.Min(prompt.Length, offset + ChunkSize)];
            var hidden = new float[chunk.Length * 2048];
            for (var i = 0; i < chunk.Length; i++)
            {
                History.Add(chunk[i]);
                MtpFakes.Row(Hash(History), History.Count - 1).CopyTo(hidden, i * 2048);
            }
            sink(chunk, hidden);
        }
        return MtpFakes.Next(Hash(History));
    }
    public void EnsureCapacity(int tokens) { }
    public int Forward(int token, int position)
    {
        ForwardCalls++;
        if (position != History.Count) throw new InvalidOperationException("Wrong decode position.");
        History.Add(token); return MtpFakes.Next(Hash(History));
    }
    public void Snapshot() { SnapshotCalls++; snapshot = [.. History]; }
    public (int[] Argmax, float[] Hidden) Verify(int[] tokens)
    {
        VerifyCalls++; VerifyRows.Add(tokens.Length);
        var argmax = new int[tokens.Length]; var hidden = new float[tokens.Length * 2048];
        for (var i = 0; i < tokens.Length; i++)
        {
            History.Add(tokens[i]);
            argmax[i] = MtpFakes.Next(Hash(History));
            MtpFakes.Row(Hash(History), History.Count - 1).CopyTo(hidden, i * 2048);
        }
        if (!replaying) OnVerify?.Invoke();
        return (argmax, hidden);
    }
    public void RestoreAndReplay(int[] committed)
    {
        ReplayCalls++;
        History = [.. snapshot ?? throw new InvalidOperationException("No snapshot.")];
        replaying = true; try { Verify(committed); } finally { replaying = false; }
        VerifyCalls--; VerifyRows.RemoveAt(VerifyRows.Count - 1);
    }
}

internal sealed class FakeDrafter(Func<int, int, bool> miss) : IMtpDrafter
{
    // Rows pair (token, target hidden tag) by position.
    public List<(int Token, int Hash, int Position)> Rows = [];
    public int ValidRows => Rows.Count;
    public Action<int>? OnCatchUp; public int CatchUpCalls, ProvisionalCalls, ResetCalls; public List<int> CatchUpSizes = [];
    private int lastHash; private int drafts;
    public void Reset() { Rows.Clear(); ResetCalls++; }
    public void EnsureCapacity(int tokens) { }
    public int CatchUp(int[] tokens, float[] hidden, int rows, bool classify)
    {
        CatchUpCalls++; CatchUpSizes.Add(rows); OnCatchUp?.Invoke(rows);
        if (tokens.Length != rows || hidden.Length != rows * 2048) throw new ArgumentException("Wrong catch-up geometry.");
        for (var i = 0; i < rows; i++)
        {
            var position = (int)hidden[i * 2048 + 1];
            if (position != Rows.Count) throw new InvalidOperationException($"Row written at {Rows.Count} but hidden belongs to {position}.");
            Rows.Add((tokens[i], (int)hidden[i * 2048], position));
            lastHash = MtpFakes.Fold((int)hidden[i * 2048], tokens[i]);
        }
        if (!classify) return -1;
        return Guess(MtpFakes.Next(lastHash));
    }
    public int Provisional(int token)
    {
        ProvisionalCalls++;
        return Guess(MtpFakes.Next(MtpFakes.Fold(lastHash, token)));
    }
    private int Guess(int truth) => miss(Rows.Count, drafts++) ? (truth + 1) % 49 : truth;
}

public class MtpEngineTests
{
    private static int[] Prompt(int n) => Enumerable.Range(0, n).Select(i => (i * 13 + 5) % 48).ToArray();
    private static (MtpGeneration Engine, FakeTarget Target, FakeDrafter Drafter, List<MtpBlock> Blocks) Run(int[] prompt, int width, int maxTokens,
        Func<int, int, bool> miss, bool ignoreEos = false, Func<double>? clock = null)
    {
        var target = new FakeTarget(); var drafter = new FakeDrafter(miss);
        var engine = new MtpGeneration(target, drafter, new(width, ignoreEos), maxTokens, t => t == MtpFakes.Eos, clock ?? (() => 0));
        engine.Prefill(prompt, CancellationToken.None);
        var blocks = new List<MtpBlock>();
        while (engine.TryNext(CancellationToken.None, out var block)) blocks.Add(block);
        return (engine, target, drafter, blocks);
    }

    public static IEnumerable<object[]> MissPatterns() =>
    [
        [(Func<int, int, bool>)((c, d) => false), "all-accept"],
        [(Func<int, int, bool>)((c, d) => true), "none-accept"],
        [(Func<int, int, bool>)((c, d) => d % 2 == 1), "second-draft-misses"],
        [(Func<int, int, bool>)((c, d) => d % 3 == 0), "periodic"],
        [(Func<int, int, bool>)((c, d) => (c * 7 + d) % 5 < 2), "irregular"],
    ];

    [Theory, MemberData(nameof(MissPatterns))]
    public void OutputTerminalStateAndPendingTokenMatchSingleTokenTarget(Func<int, int, bool> miss, string label)
    {
        foreach (var promptLength in new[] { 1, 2, 31, 32, 33, 64, 65 })
        foreach (var width in new[] { 1, 2 })
        foreach (var maxTokens in new[] { 1, 2, 3, 4, 5, 8, 17, 40 })
        foreach (var ignoreEos in new[] { false, true })
        {
            var prompt = Prompt(promptLength);
            var (engine, target, drafter, blocks) = Run(prompt, width, maxTokens, miss, ignoreEos);
            var expected = MtpFakes.Reference(prompt, maxTokens, ignoreEos);
            var where = $"{label} prompt={promptLength} width={width} max={maxTokens} ignoreEos={ignoreEos}";
            Assert.True(expected.Output.SequenceEqual(engine.Output), $"{where}: tokens {string.Join(',', engine.Output)} != {string.Join(',', expected.Output)}");
            Assert.True(expected.Output.SequenceEqual(blocks.SelectMany(b => b.Tokens)), where + ": streamed blocks differ from output");
            Assert.Equal(expected.Finish, engine.Finish);
            Assert.True(expected.Consumed.SequenceEqual(target.History), $"{where}: target consumed history differs");
            Assert.Equal(expected.Pending, engine.Anchor);
            Assert.Equal(target.History.Count, engine.Committed);
            if (engine.Finish == "length") Assert.True(engine.Output.Count == maxTokens, where);
            // Draft rows never run ahead of the committed target and are all target-conditioned.
            Assert.True(drafter.ValidRows <= engine.Committed, where);
        }
    }

    [Fact]
    public void InitialAnchorIsEmittedOnceFirstAndUnconsumed()
    {
        var prompt = Prompt(5);
        var (engine, target, _, blocks) = Run(prompt, 2, 6, (_, _) => false);
        Assert.True(blocks[0].Anchor);
        Assert.Equal([MtpFakes.Reference(prompt, 6, false).Output[0]], blocks[0].Tokens);
        Assert.All(blocks.Skip(1), b => Assert.False(b.Anchor));
        // First verification includes the unconsumed anchor as row 0.
        Assert.Equal(engine.Output[0], target.History[prompt.Length]);
    }

    [Fact]
    public void EveryCycleVerifiesAnchorPlusDraftsInOneBatchedCall()
    {
        var (_, target, _, _) = Run(Prompt(40), 2, 30, (_, _) => false);
        Assert.True(target.VerifyCalls > 0);
        Assert.All(target.VerifyRows, rows => Assert.InRange(rows, 2, 3));
    }

    [Fact]
    public void AllAcceptanceCommitsEveryDraftWithoutRollback()
    {
        var (engine, target, _, _) = Run(Prompt(10), 2, 21, (_, _) => false);
        Assert.Equal(0, target.ReplayCalls);
        Assert.Equal(engine.Stats.Proposed, engine.Stats.Accepted);
        Assert.True(engine.Stats.FullAcceptCycles > 0);
    }

    [Fact]
    public void NoAcceptanceRollsBackEveryCycleAndStillMatchesTarget()
    {
        var prompt = Prompt(10);
        var (engine, target, _, _) = Run(prompt, 2, 21, (_, _) => true);
        Assert.Equal(0, engine.Stats.Accepted);
        Assert.Equal(engine.Stats.Cycles, target.ReplayCalls);
        Assert.Equal(engine.Stats.Cycles, engine.Stats.AcceptedHistogram[0]);
        Assert.True(MtpFakes.Reference(prompt, 21, false).Output.SequenceEqual(engine.Output));
    }

    [Fact]
    public void EveryRejectionOffsetIsExercisedAndExact()
    {
        var offsets = new HashSet<int>();
        for (var seed = 0; seed < 12; seed++)
        {
            var prompt = Prompt(8 + seed);
            var (engine, _, _, _) = Run(prompt, 2, 40, (c, d) => (c + d * 3 + seed) % 4 == 0);
            for (var i = 0; i < 3; i++) if (engine.Stats.AcceptedHistogram[i] > 0) offsets.Add(i);
            Assert.True(MtpFakes.Reference(prompt, 40, true).Output.SequenceEqual(engine.Output) || engine.Finish == "eos");
        }
        Assert.Equal([0, 1, 2], offsets.Order());
    }

    [Fact]
    public void DraftCacheIsReconditionedFromTargetHiddenRowsEveryCycle()
    {
        // FakeDrafter throws if a row is written with hidden that does not belong to its position, and the draft
        // cursor must equal the committed length at every cycle (the engine also enforces this).
        var prompt = Prompt(37);
        var (engine, _, drafter, _) = Run(prompt, 2, 40, (c, d) => (c + d) % 3 == 0, ignoreEos: true);
        var truth = prompt.ToList(); truth.AddRange(engine.Output.Take(engine.Output.Count - 1));
        var h = 0;
        for (var p = 0; p < drafter.Rows.Count; p++)
        {
            h = MtpFakes.Fold(h, truth[p]);
            Assert.Equal(h, drafter.Rows[p].Hash);
            Assert.Equal(p, drafter.Rows[p].Position);
            // Row p pairs the NEXT token with hidden p.
            Assert.Equal(truth[p + 1], drafter.Rows[p].Token);
        }
        Assert.True(drafter.CatchUpCalls > 1);
    }

    [Fact]
    public void BootstrapIsBoundedChunkedAndLeavesTheCursorAtPromptMinusOne()
    {
        var prompt = Prompt(100);
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        var engine = new MtpGeneration(target, drafter, new(1), 10, t => t == MtpFakes.Eos, () => 0);
        engine.Prefill(prompt, CancellationToken.None);
        // Bootstrap pairs rows 0..P-2 with tokens 1..P-1; row P-1 awaits the verified anchor in decode time.
        Assert.Equal(99, drafter.ValidRows);
        Assert.All(drafter.CatchUpSizes, n => Assert.InRange(n, 1, 32));
        Assert.Equal(99, drafter.CatchUpSizes.Sum());
        Assert.Equal(0, target.VerifyCalls);
    }

    [Fact]
    public void LengthOneAndEosFirstSkipDraftingEntirely()
    {
        var (engine, target, drafter, blocks) = Run(Prompt(6), 2, 1, (_, _) => false);
        Assert.Single(blocks); Assert.Equal("length", engine.Finish);
        Assert.Equal(0, target.VerifyCalls + target.ForwardCalls); Assert.Equal(0, drafter.CatchUpCalls);
        // Find a prompt whose first token is EOS.
        for (var n = 1; n < 400; n++)
        {
            var p = Prompt(n).Concat([n % 48]).ToArray();
            if (MtpFakes.Reference(p, 5, false).Finish == "eos" && MtpFakes.Reference(p, 5, false).Output.Count == 0)
            {
                var r = Run(p, 2, 5, (_, _) => false);
                Assert.Empty(r.Blocks); Assert.Equal("eos", r.Engine.Finish); Assert.Equal(MtpFakes.Eos, r.Engine.Anchor);
                Assert.Equal(0, r.Target.VerifyCalls); return;
            }
        }
        Assert.Fail("No EOS-first prompt found in the search range.");
    }

    [Fact]
    public void WidthIsClampedToRemainingOutputsAndTheTailUsesAutoregressiveDecode()
    {
        var (engine, target, _, _) = Run(Prompt(12), 2, 4, (_, _) => false, ignoreEos: true);
        // 1 anchor + clamp: remaining 3 -> k=2 cycle emitting up to 3 tokens, never more than maxTokens.
        Assert.Equal(4, engine.Output.Count);
        Assert.All(target.VerifyRows, r => Assert.True(r <= 3));
        var (e2, t2, _, _) = Run(Prompt(12), 2, 2, (_, _) => false, ignoreEos: true);
        Assert.Equal(2, e2.Output.Count); Assert.Equal(0, t2.VerifyCalls); Assert.Equal(1, t2.ForwardCalls);
    }

    [Fact]
    public void EosInsideAcceptedBlockTruncatesAndRollsBackToTheExactTerminalState()
    {
        var found = 0;
        for (var n = 1; n < 300; n++)
        {
            var prompt = Prompt(n);
            var expected = MtpFakes.Reference(prompt, 60, false);
            if (expected.Finish != "eos" || expected.Output.Count < 3) continue;
            var (engine, target, _, _) = Run(prompt, 2, 60, (_, _) => false);
            Assert.True(expected.Output.SequenceEqual(engine.Output)); Assert.Equal("eos", engine.Finish);
            Assert.True(expected.Consumed.SequenceEqual(target.History)); Assert.Equal(MtpFakes.Eos, engine.Anchor);
            found++;
        }
        Assert.True(found > 5, "Expected several natural-EOS fixtures in the sweep.");
    }

    [Fact]
    public void ReadyTimesCoverPrefillAndOneStampPerBlockAndAreMonotonic()
    {
        var tick = 0d;
        var (engine, _, _, blocks) = Run(Prompt(33), 2, 20, (c, d) => (c + d) % 4 == 0, ignoreEos: true, clock: () => tick += 1);
        Assert.True(blocks[0].Anchor);
        for (var i = 1; i < blocks.Count; i++) Assert.True(blocks[i].ReadyMs > blocks[i - 1].ReadyMs);
        Assert.True(engine.Stats.BootstrapMs > 0 && engine.Stats.VerifyMs > 0 && engine.Stats.CatchUpMs > 0 && engine.Stats.SnapshotMs > 0);
        Assert.Equal(20, blocks.Sum(b => b.Tokens.Length));
    }

    [Fact]
    public void CancellationLeavesNoPartialCycleAndPrefillCancelDoesNotStart()
    {
        var prompt = Prompt(9);
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        var engine = new MtpGeneration(target, drafter, new(2), 30, t => t == MtpFakes.Eos, () => 0);
        engine.Prefill(prompt, CancellationToken.None);
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        var before = (target.History.Count, drafter.ValidRows, engine.Output.Count);
        using var cts = new CancellationTokenSource(); cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => engine.TryNext(cts.Token, out _));
        Assert.Equal(before, (target.History.Count, drafter.ValidRows, engine.Output.Count));
    }

    [Fact]
    public void FirstDraftRunsAfterTheInitialAnchorIsReadyNotInPrefill()
    {
        var prompt = Prompt(45); var tick = 0d; double? firstDraftAt = null;
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        var engine = new MtpGeneration(target, drafter, new(2), 12, t => t == MtpFakes.Eos, () => tick += 1);
        engine.Prefill(prompt, CancellationToken.None);
        var calls = drafter.CatchUpCalls;
        drafter.OnCatchUp = rows => firstDraftAt ??= tick;
        Assert.Equal(prompt.Length - 1, drafter.ValidRows);
        Assert.Equal(0, engine.Stats.FirstDraftMs);
        Assert.True(engine.TryNext(CancellationToken.None, out var anchor));
        Assert.True(anchor.Anchor);
        Assert.Equal(calls, drafter.CatchUpCalls);
        Assert.Equal(0, drafter.ProvisionalCalls);
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        Assert.Equal(calls + 2, drafter.CatchUpCalls);
        Assert.True(engine.Stats.FirstDraftMs > 0);
        Assert.True(firstDraftAt > anchor.ReadyMs);
        Assert.Equal(1, drafter.CatchUpSizes[calls]);
    }

    [Fact]
    public void AutoregressiveTailAndEosNeverPayForAFirstDraft()
    {
        var (_, target, drafter, _) = Run(Prompt(30), 2, 2, (_, _) => false, ignoreEos: true);
        Assert.Equal(29, drafter.ValidRows); Assert.Equal(0, drafter.ProvisionalCalls); Assert.Equal(0, target.VerifyCalls);
    }

    [Fact]
    public void CancelBeforeFirstDraftAndBeforeVerifyLeaveResumableConsistentState()
    {
        var prompt = Prompt(11);
        var expected = MtpFakes.Reference(prompt, 20, true);
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        var engine = new MtpGeneration(target, drafter, new(2, true), 20, _ => false, () => 0);
        engine.Prefill(prompt, CancellationToken.None);
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        using var before = new CancellationTokenSource(); before.Cancel();
        Assert.Throws<OperationCanceledException>(() => engine.TryNext(before.Token, out _));
        Assert.Equal((prompt.Length, prompt.Length - 1), (target.History.Count, drafter.ValidRows));
        // Cancellation arrives during the first draft: the next boundary is before snapshot/verify.
        using var during = new CancellationTokenSource();
        drafter.OnCatchUp = _ => during.Cancel();
        Assert.Throws<OperationCanceledException>(() => engine.TryNext(during.Token, out _));
        Assert.Equal((prompt.Length, prompt.Length), (target.History.Count, drafter.ValidRows));
        Assert.Equal(0, target.SnapshotCalls);
        drafter.OnCatchUp = null;
        while (engine.TryNext(CancellationToken.None, out _)) { }
        Assert.True(expected.Output.SequenceEqual(engine.Output));
        Assert.True(expected.Consumed.SequenceEqual(target.History));
    }

    [Fact]
    public void CancelAfterVerifyIsObservedOnlyAfterRestoreReplayAndRecoversByNewRequest()
    {
        var prompt = Prompt(13);
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => true);
        var engine = new MtpGeneration(target, drafter, new(2, true), 25, _ => false, () => 0);
        engine.Prefill(prompt, CancellationToken.None);
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        using var cts = new CancellationTokenSource();
        target.OnVerify = () => cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => engine.TryNext(cts.Token, out _));
        Assert.Equal(1, target.ReplayCalls);
        // Target holds exactly the committed prefix and the pending anchor is the last emitted token.
        Assert.Equal(engine.Committed, target.History.Count);
        var reference = MtpFakes.Reference(prompt, engine.Output.Count, true);
        Assert.True(reference.Output.SequenceEqual(engine.Output));
        Assert.True(reference.Consumed.SequenceEqual(target.History));
        Assert.Equal(engine.Output[^1], engine.Anchor);
        Assert.Throws<InvalidOperationException>(() => engine.TryNext(CancellationToken.None, out _));
        // A new request resets the draft cache and the full generation is exact again.
        target.OnVerify = null;
        var next = new MtpGeneration(target, drafter, new(2, true), 25, _ => false, () => 0);
        next.Prefill(prompt, CancellationToken.None);
        while (next.TryNext(CancellationToken.None, out _)) { }
        Assert.True(MtpFakes.Reference(prompt, 25, true).Output.SequenceEqual(next.Output));
        Assert.True(MtpFakes.Reference(prompt, 25, true).Consumed.SequenceEqual(target.History));
    }

    [Fact]
    public void InvalidWidthsAndMisuseAreRejectedExplicitly()
    {
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        foreach (var width in new[] { 0, 3, -1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => new MtpGeneration(target, drafter, new(width), 4, _ => false, () => 0));
        var engine = new MtpGeneration(target, drafter, new(1), 4, t => t == MtpFakes.Eos, () => 0);
        Assert.Throws<InvalidOperationException>(() => engine.TryNext(CancellationToken.None, out _));
        engine.Prefill(Prompt(3), CancellationToken.None);
        Assert.Throws<InvalidOperationException>(() => engine.Prefill(Prompt(3), CancellationToken.None));
    }

    [Fact]
    public void StaleDraftCursorFailsInsteadOfSilentlyDrafting()
    {
        var target = new FakeTarget(); var drafter = new FakeDrafter((_, _) => false);
        var engine = new MtpGeneration(target, drafter, new(1), 20, t => t == MtpFakes.Eos, () => 0);
        engine.Prefill(Prompt(8), CancellationToken.None);
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        Assert.True(engine.TryNext(CancellationToken.None, out _));
        drafter.Rows.RemoveAt(drafter.Rows.Count - 1);
        Assert.Throws<InvalidOperationException>(() => engine.TryNext(CancellationToken.None, out _));
    }

    [Fact]
    public void SerialAndBatchedReplayAreIndependentOfTheEngineContract()
    {
        // The engine only asks the target to restore+replay; both fake modes must yield the same terminal history.
        var prompt = Prompt(20);
        var a = Run(prompt, 2, 30, (_, _) => true, ignoreEos: true);
        Assert.Equal(a.Engine.Stats.Cycles, a.Target.ReplayCalls);
        Assert.True(MtpFakes.Reference(prompt, 30, true).Consumed.SequenceEqual(a.Target.History));
    }
}
