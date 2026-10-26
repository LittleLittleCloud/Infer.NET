using DeepInfer.Backends.Dflash2;
using Xunit;
namespace DeepInfer.Tests;

// Deterministic fake target/drafter pair. The target's greedy token is a pure function of the history hash. Every tap row
// carries (history hash through that position, position); the drafter derives its proposals only from the rows it ingested,
// so wrong ingestion (stale rows, double ingest, wrong positions) changes the proposals and the verified output.
internal static class DfFakes
{
    public const int Eos = 49;
    public static int Fold(int h, int token) => (h * 31 + token + 1) % 1000003;
    public static int Next(int h) => (h * 7 + 3) % 50;

    public static (List<int> Output, string Finish, int Consumed, int Pending) Reference(int[] prompt, int maxTokens, bool ignoreEos)
    {
        var h = 0; foreach (var t in prompt) h = Fold(h, t);
        var consumed = prompt.Length; var output = new List<int>(); var next = Next(h);
        while (true)
        {
            if (!ignoreEos && next == Eos) return (output, "eos", consumed, next);
            output.Add(next);
            if (output.Count >= maxTokens) return (output, "length", consumed, next);
            consumed++; h = Fold(h, next); next = Next(h);
        }
    }
}

internal class DfTarget : IDflash2Target
{
    public List<int> History = [];
    public List<(int Hash, int Position)> Taps = [];
    private List<int>? snapshot;
    public int VerifyCalls, ForwardCalls, ReplayCalls, SnapshotCalls, Chunk = 32; public List<int> VerifyRows = [];
    private static int Hash(IEnumerable<int> tokens) { var h = 0; foreach (var t in tokens) h = DfFakes.Fold(h, t); return h; }
    public int Prefill(int[] prompt, Action<int, int> sink, CancellationToken ct)
    {
        History = [];
        for (var offset = 0; offset < prompt.Length; offset += Chunk)
        {
            var count = Math.Min(Chunk, prompt.Length - offset);
            Taps = [];
            for (var i = 0; i < count; i++) { History.Add(prompt[offset + i]); Taps.Add((Hash(History), History.Count - 1)); }
            sink(offset, count);
        }
        return DfFakes.Next(Hash(History));
    }
    public void EnsureCapacity(int tokens) { }
    public int Forward(int token, int position)
    {
        ForwardCalls++;
        if (position != History.Count) throw new InvalidOperationException("Wrong decode position.");
        History.Add(token); return DfFakes.Next(Hash(History));
    }
    public virtual void Snapshot() { SnapshotCalls++; snapshot = [.. History]; }
    public virtual int[] Verify(int[] tokens)
    {
        VerifyCalls++; VerifyRows.Add(tokens.Length);
        var argmax = new int[tokens.Length]; Taps = [];
        for (var i = 0; i < tokens.Length; i++)
        {
            History.Add(tokens[i]); var h = Hash(History);
            argmax[i] = DfFakes.Next(h); Taps.Add((h, History.Count - 1));
        }
        return argmax;
    }
    public virtual bool DirectCommit => false;
    public virtual void CommitPrefix(int[] tokens, int accepted)
    { if (accepted < tokens.Length) RestoreAndReplay(tokens.AsSpan(0, accepted).ToArray()); }
    public virtual void RestoreAndReplay(int[] committed)
    {
        ReplayCalls++;
        History = [.. snapshot ?? throw new InvalidOperationException("No snapshot.")];
        Verify(committed);
        VerifyCalls--; VerifyRows.RemoveAt(VerifyRows.Count - 1);
    }
}

internal sealed class DfDirectTarget : DfTarget
{
    private int[] candidates = [];
    public bool Pending { get; private set; }
    public override bool DirectCommit => true;
    public override void Snapshot() { Assert.False(Pending); }
    public override int[] Verify(int[] tokens)
    {
        Assert.False(Pending); var committed = History.ToList();
        var next = base.Verify(tokens); History = committed;
        candidates = tokens.ToArray(); Pending = true; return next;
    }
    public override void CommitPrefix(int[] tokens, int accepted)
    {
        Assert.True(Pending); Assert.Equal(candidates, tokens); Assert.InRange(accepted, 1, tokens.Length);
        History.AddRange(tokens.AsSpan(0, accepted).ToArray());
        Taps = Taps.Take(accepted).ToList(); Pending = false;
    }
    public override void RestoreAndReplay(int[] committed) => Assert.Fail("Direct commit must not replay target inputs.");
}

internal sealed class DfDrafter(DfTarget target) : IDflash2Drafter
{
    public int End = -1, Hash, IngestCalls, IngestedRows; public int FirstIngested = -1;
    public List<int[]> Blocks = [];
    // Accepted counts to force, one per cycle: proposals are right for `Script[i]` rows, then corrupted.
    public Queue<int> Script = new();
    public int ContextEnd => End < 0 ? 0 : End;
    public void Reset() { End = -1; Hash = 0; IngestCalls = IngestedRows = 0; FirstIngested = -1; Blocks.Clear(); }
    public void Ingest(int tapRow, int rows, int firstPosition)
    {
        if (target is DfDirectTarget { Pending: true }) throw new InvalidOperationException("Cannot ingest tentative rows.");
        IngestCalls++; IngestedRows += rows;
        if (End >= 0 && firstPosition != End) throw new InvalidOperationException($"Non-contiguous ingest at {firstPosition}; expected {End}.");
        if (End < 0) FirstIngested = firstPosition;
        for (var i = 0; i < rows; i++)
        {
            var (hash, position) = target.Taps[tapRow + i];
            if (position != firstPosition + i) throw new InvalidOperationException($"Stale tap row: position {position} for {firstPosition + i}.");
            Hash = hash;
        }
        End = firstPosition + rows;
    }
    public int[] Propose(int[] block)
    {
        Blocks.Add(block);
        if (End < 0) throw new InvalidOperationException("Propose before any context.");
        var want = Script.Count > 0 ? Script.Dequeue() : int.MaxValue;
        var proposals = new int[block.Length - 1];
        var h = DfFakes.Fold(Hash, block[0]);
        for (var i = 0; i < proposals.Length; i++)
        {
            var next = DfFakes.Next(h);
            proposals[i] = i < want ? next : (next + 1) % 50 == DfFakes.Eos ? (next + 2) % 50 : (next + 1) % 50;
            h = DfFakes.Fold(h, next);
        }
        return proposals;
    }
}

public class Dflash2EngineTests
{
    private static int[] Prompt(int n, int seed = 5) => Enumerable.Range(0, n).Select(i => (i * 13 + seed) % 40).ToArray();

    private static (Dflash2Generation Gen, DfTarget Target, DfDrafter Draft, List<(int[] Tokens, double Ms)> Blocks, List<int> Output) Run(int block, int maxTokens, int promptLength,
        bool ignoreEos = false, IEnumerable<int>? script = null, int chunk = 32, CancellationToken ct = default)
    {
        var target = new DfTarget { Chunk = chunk }; var draft = new DfDrafter(target);
        if (script is not null) foreach (var s in script) draft.Script.Enqueue(s);
        var clock = 0.0;
        var gen = new Dflash2Generation(target, draft, new Dflash2Options(block, ignoreEos), maxTokens, t => t == DfFakes.Eos, () => clock += 1.0);
        gen.Prefill(Prompt(promptLength), ct);
        var blocks = new List<(int[], double)>(); var output = new List<int>();
        while (gen.TryNext(ct, out var b)) { blocks.Add((b.Tokens, b.ReadyMs)); output.AddRange(b.Tokens); }
        return (gen, target, draft, blocks, output);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void EveryRejectionOffsetMatchesSerialGreedyOutputAndTerminalState(int block)
    {
        var prompt = Prompt(70); var reference = DfFakes.Reference(prompt, 60, true);
        for (var offset = 0; offset < block; offset++)
        {
            var (gen, target, draft, _, output) = Run(block, 60, 70, true, Enumerable.Repeat(offset, 100));
            Assert.Equal(reference.Output, output);
            Assert.Equal(reference.Consumed, gen.Committed); Assert.Equal(reference.Consumed, target.History.Count);
            Assert.Equal(reference.Pending, gen.Anchor);
            // The final cycle skips context ingestion because the request ends; every earlier boundary is exact.
            Assert.InRange(draft.ContextEnd, reference.Consumed - block, reference.Consumed);
            Assert.True(gen.Stats.AcceptedHistogram[Math.Min(offset, block - 1)] > 0);
            if (offset < block - 1) Assert.True(gen.Stats.Replays > 0); else Assert.Equal(0, gen.Stats.Replays);
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void CapsAndTailsClampAtEveryLength(int block)
    {
        var prompt = Prompt(40);
        for (var max = 1; max <= 3 * block + 2; max++)
            foreach (var ignore in new[] { true, false })
            {
                var reference = DfFakes.Reference(prompt, max, ignore);
                var (gen, target, draft, _, output) = Run(block, max, 40, ignore);
                Assert.Equal(reference.Output, output);
                Assert.Equal(reference.Finish, gen.Finish);
                Assert.Equal(reference.Consumed, target.History.Count);
                Assert.Equal(reference.Pending, gen.Anchor);
                Assert.True(output.Count <= max);
                Assert.True(target.VerifyRows.All(r => r <= block));
            }
    }

    [Fact]
    public void NaturalEosAtEveryOffsetStopsWithoutEmittingEosAndLeavesExactState()
    {
        var found = 0;
        for (var seed = 0; seed < 40; seed++)
        {
            var prompt = Prompt(33 + seed, seed); var reference = DfFakes.Reference(prompt, 200, false);
            if (reference.Finish != "eos") continue;
            found++;
            foreach (var block in new[] { 4, 8 })
            {
                var target = new DfTarget(); var draft = new DfDrafter(target); var clock = 0.0;
                var gen = new Dflash2Generation(target, draft, new Dflash2Options(block), 200, t => t == DfFakes.Eos, () => clock += 1);
                gen.Prefill(prompt, default); var output = new List<int>();
                while (gen.TryNext(default, out var b)) output.AddRange(b.Tokens);
                Assert.Equal(reference.Output, output); Assert.Equal("eos", gen.Finish);
                Assert.DoesNotContain(DfFakes.Eos, output);
                Assert.Equal(reference.Consumed, target.History.Count);
                Assert.Equal(DfFakes.Eos, gen.Anchor);
            }
        }
        Assert.True(found > 0, "Fixture must contain natural EOS prompts.");
    }

    [Fact]
    public void BlockTokensShareOneReadyTimestampAndPrefillTimeIsNotInDecodeBlocks()
    {
        var (gen, _, _, blocks, _) = Run(8, 40, 50, true);
        Assert.True(blocks[0].Tokens.Length == 1);
        for (var i = 1; i < blocks.Count; i++) Assert.True(blocks[i].Ms > blocks[i - 1].Ms);
        Assert.True(gen.Stats.BootstrapMs > 0);
        Assert.Equal(blocks.Count, blocks.Select(b => b.Ms).Distinct().Count());
    }

    [Fact]
    public void BootstrapIngestsOnlyTheWindowedPromptTailOnceAndRollbackNeverReingests()
    {
        var (gen, target, draft, _, _) = Run(4, 24, 3000, true, Enumerable.Repeat(1, 50));
        Assert.Equal(3000 - 2047, draft.FirstIngested);
        // Contiguity is enforced by the drafter; rows ingested equal the covered span, so nothing is ingested twice.
        Assert.Equal(draft.ContextEnd - (3000 - 2047), draft.IngestedRows);
        Assert.True(draft.ContextEnd <= gen.Committed && gen.Committed - draft.ContextEnd <= 4);
        Assert.True(gen.Stats.Replays > 0);
        Assert.Equal(gen.Committed, target.History.Count);
    }

    [Fact]
    public void ShortPromptIngestsEveryRowFromZero()
    {
        var (_, _, draft, _, _) = Run(8, 10, 100, true);
        Assert.Equal(0, draft.FirstIngested);
    }

    [Fact]
    public void SingleOutputTokenNeverBootstrapsOrDrafts()
    {
        var (gen, target, draft, _, output) = Run(8, 1, 64, true);
        Assert.Single(output); Assert.Equal(0, draft.IngestCalls); Assert.Empty(draft.Blocks); Assert.Equal(0, target.VerifyCalls);
        Assert.Equal("length", gen.Finish);
    }

    [Fact]
    public void ProposalBlockIsAnchorThenMaskTokens()
    {
        var (_, _, draft, _, _) = Run(8, 20, 40, true);
        Assert.All(draft.Blocks, b => Assert.Equal(248077, b.Skip(1).Distinct().Single()));
        Assert.NotEmpty(draft.Blocks);
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void RemainingOutputClampsTheActualDraftedBlockAndReportsTrueProposedCount(int block)
    {
        // 20 outputs: anchor + full blocks, then a tail block drafted at exactly remaining+1 rows.
        var (gen, target, draft, _, output) = Run(block, 20, 40, true);
        Assert.Equal(20, output.Count);
        Assert.All(draft.Blocks, b => Assert.InRange(b.Length, 2, block));
        var emitted = 1; // the perfect fake drafter fully accepts, so every cycle emits one token per drafted row
        foreach (var b in draft.Blocks) { Assert.Equal(Math.Min(block, 20 - emitted), b.Length); emitted += b.Length; }
        Assert.Equal(20, emitted);
        Assert.Equal(draft.Blocks.Sum(b => b.Length - 1), gen.Stats.Proposed);
    }

    [Fact]
    public void CancellationStopsBeforeAnyTargetWorkAndDoesNotCorruptCommittedState()
    {
        var target = new DfTarget(); var draft = new DfDrafter(target); var clock = 0.0;
        var gen = new Dflash2Generation(target, draft, new Dflash2Options(4), 50, t => t == DfFakes.Eos, () => clock += 1);
        gen.Prefill(Prompt(40), default);
        Assert.True(gen.TryNext(default, out _)); // anchor
        using var cts = new CancellationTokenSource(); cts.Cancel();
        var verifies = target.VerifyCalls;
        Assert.Throws<OperationCanceledException>(() => gen.TryNext(cts.Token, out _));
        Assert.Equal(verifies, target.VerifyCalls);
        Assert.Equal(40, gen.Committed); Assert.Equal(40, draft.ContextEnd);
        Assert.True(gen.TryNext(default, out _));
    }

    [Fact]
    public void ResetAndFreshGenerationsAreIndependent()
    {
        var target = new DfTarget(); var draft = new DfDrafter(target); var clock = 0.0;
        List<int> Once(int length)
        {
            var gen = new Dflash2Generation(target, draft, new Dflash2Options(8), 30, t => t == DfFakes.Eos, () => clock += 1);
            gen.Prefill(Prompt(length), default); var o = new List<int>();
            while (gen.TryNext(default, out var b)) o.AddRange(b.Tokens);
            return o;
        }
        var first = Once(45); var other = Once(77); var again = Once(45);
        Assert.Equal(first, again);
        Assert.Equal(DfFakes.Reference(Prompt(77), 30, false).Output, other);
    }

    [Fact]
    public void ContextGrowthAcrossTheWindowKeepsContiguousIngest()
    {
        var (gen, _, draft, _, output) = Run(8, 300, 2040, true);
        Assert.InRange(draft.ContextEnd, 2040 + 299 - 8, 2040 + 299);
        Assert.Equal(300, output.Count);
        Assert.Equal(2040 + 299, gen.Committed);
    }

    [Fact]
    public void ContextMismatchBetweenDrafterAndCommittedPositionIsAHardError()
    {
        var target = new DfTarget(); var draft = new DfDrafter(target); var clock = 0.0;
        var gen = new Dflash2Generation(target, draft, new Dflash2Options(4), 50, t => false, () => clock += 1);
        gen.Prefill(Prompt(40), default); gen.TryNext(default, out _);
        draft.End -= 1;
        Assert.Throws<InvalidOperationException>(() => gen.TryNext(default, out _));
    }

    [Fact]
    public void InvalidBlockSizesAndLimitsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dflash2Options(2).Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new Dflash2Options(16).Validate());
        Assert.Null(Dflash2Limits.Check(Dflash2Limits.MaxContextTokens - 4096, 4096));
        Assert.NotNull(Dflash2Limits.Check(Dflash2Limits.MaxContextTokens - 4095, 4096));
        Assert.NotNull(Dflash2Limits.Check(10, 4097));
        Assert.NotNull(Dflash2Limits.Check(10, 0));
        Assert.Null(Dflash2Limits.CheckOutput(4096, false));
        Assert.NotNull(Dflash2Limits.CheckOutput(4097, false));
        Assert.NotNull(Dflash2Limits.CheckOutput(16, true));
    }

    [Fact]
    public void StatsLogLineIsStructured()
    {
        var (gen, _, _, _, _) = Run(8, 40, 50, true, Enumerable.Repeat(3, 20));
        var line = gen.Stats.ToLogLine(8);
        Assert.Contains("block=8", line); Assert.Contains("acceptedHistogram=", line); Assert.Contains("bootstrapMs=", line);
        Assert.Equal(gen.Stats.Cycles, gen.Stats.AcceptedHistogram.Sum());
        Assert.True(gen.Stats.Accepted <= gen.Stats.Proposed);
    }
}

public class Dflash8DirectCommitTests
{
    [Fact]
    public void EveryRejectionOffsetAndTailMatchesScalarWithoutReplayOrTentativeIngest()
    {
        foreach (var offset in Enumerable.Range(0, 8)) foreach (var max in Enumerable.Range(1, 25))
        {
            var prompt = Enumerable.Range(0, 17).Select(i => i % 40).ToArray();
            var target = new DfDirectTarget(); var draft = new DfDrafter(target);
            foreach (var i in Enumerable.Repeat(offset, 64)) draft.Script.Enqueue(i);
            var clock = 0d; var engine = new Dflash2Generation(target, draft, new(8, true, true), max, t => t == DfFakes.Eos, () => ++clock);
            engine.Prefill(prompt, default); var output = new List<int>();
            while (engine.TryNext(default, out var block)) output.AddRange(block.Tokens);
            var expected = DfFakes.Reference(prompt, max, true);
            Assert.Equal(expected.Output, output); Assert.Equal(expected.Consumed, target.History.Count);
            Assert.Equal(expected.Pending, engine.Anchor); Assert.False(target.Pending);
            Assert.All(target.VerifyRows, n => Assert.Equal(8, n));
            Assert.Equal(0, target.ReplayCalls); Assert.Equal(0, target.SnapshotCalls); Assert.Equal(0, engine.Stats.Replays);
        }
    }

    [Fact]
    public void NaturalEosAtAcceptedOrCorrectionPositionLeavesExactCommittedState()
    {
        foreach (var seed in Enumerable.Range(0, 50)) foreach (var offset in Enumerable.Range(0, 8))
        {
            var prompt = new[] { seed, 3, 11 }; var target = new DfDirectTarget(); var draft = new DfDrafter(target);
            foreach (var i in Enumerable.Repeat(offset, 128)) draft.Script.Enqueue(i);
            var clock = 0d; var engine = new Dflash2Generation(target, draft, new(8, false, true), 65, t => t == DfFakes.Eos, () => ++clock);
            engine.Prefill(prompt, default); var output = new List<int>();
            while (engine.TryNext(default, out var block)) output.AddRange(block.Tokens);
            var expected = DfFakes.Reference(prompt, 65, false);
            Assert.Equal(expected.Output, output); Assert.Equal(expected.Finish, engine.Finish);
            Assert.Equal(expected.Consumed, target.History.Count); Assert.False(target.Pending); Assert.Equal(0, target.ReplayCalls);
        }
    }
}
