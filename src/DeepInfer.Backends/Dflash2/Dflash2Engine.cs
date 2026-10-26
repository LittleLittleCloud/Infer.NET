using DeepInfer.Backends.Models;
namespace DeepInfer.Backends.Dflash2;

// Target-side operations the DFlash2 engine needs. Verify rows must equal single-token decode (argmax), the target feature
// taps stay device-resident for the drafter, and RestoreAndReplay must reproduce the committed single-token state.
public interface IDflash2Target
{
    // Prefill the prompt and return the first target anchor. The sink receives (offset, rows) after each chunk; the chunk's
    // feature rows are then available to the drafter's Ingest(tapRow, ...) until the next target call.
    int Prefill(int[] prompt, Action<int, int> chunkSink, CancellationToken ct);
    void EnsureCapacity(int tokens);
    // Single-token decode of `token` at `position` (the committed length); returns the target argmax.
    int Forward(int token, int position);
    // Capture the committed state at the current position (before a verification).
    void Snapshot();
    // Batched exact verification of [anchor, d1..dk]: argmax per row. Feature rows 0..k are tapped on the device.
    int[] Verify(int[] tokens);
    // Restore the last snapshot and re-run the committed prefix so the target holds exactly that prefix.
    void RestoreAndReplay(int[] committed);
    bool DirectCommit => false;
    void CommitPrefix(int[] candidates, int accepted)
    {
        if (accepted < candidates.Length) RestoreAndReplay(candidates.AsSpan(0, accepted).ToArray());
    }
}

public interface IDflash2Drafter
{
    // First absolute position not yet in the committed context.
    int ContextEnd { get; }
    void Reset();
    // Appends rows [tapRow, tapRow+rows) of the target's latest tap capture at absolute positions firstPosition...
    void Ingest(int tapRow, int rows, int firstPosition);
    // block[0] is the anchor, block[1..] the mask token; returns block.Length-1 proposals.
    int[] Propose(int[] block);
}

public sealed record Dflash2Options(int Block, bool IgnoreEos = false, bool FullBlocksOnly = false)
{
    public void Validate()
    {
        if (Block is not (4 or 8)) throw new ArgumentOutOfRangeException(nameof(Block), "DFlash2 block must be 4 or 8.");
    }
}

// Experimental envelope (32768 input + 4096 output; Pi context remains 32768). Larger requests are rejected, never truncated or sent down another path.
public static class Dflash2Limits
{
    public const int MaxContextTokens = 32768 + 4096;
    public const int MaxOutputTokens = 4096;
    public static string? CheckOutput(int maxTokens, bool useRemainingContext)
    {
        if (useRemainingContext) return $"DFlash2 mode does not support remaining-context outputs; request at most {MaxOutputTokens} output tokens.";
        return maxTokens > MaxOutputTokens ? $"DFlash2 mode supports at most {MaxOutputTokens} output tokens (requested {maxTokens})." : null;
    }
    public static string? Check(int promptTokens, int maxTokens)
    {
        if (maxTokens < 1) return "DFlash2 mode requires at least one output token.";
        if (maxTokens > MaxOutputTokens) return $"DFlash2 mode supports at most {MaxOutputTokens} output tokens (requested {maxTokens}).";
        if ((long)promptTokens + maxTokens > MaxContextTokens) return $"DFlash2 mode supports at most {MaxContextTokens} prompt+output tokens (requested {(long)promptTokens + maxTokens}).";
        return null;
    }
}

public sealed record Dflash2Block(int[] Tokens, double ReadyMs, bool Anchor);

public sealed class Dflash2Stats
{
    public long Cycles, Proposed, Accepted, RejectedCycles, FullAcceptCycles, ArSteps, Replays, ContextRows;
    // BootstrapMs is prefill/TTFT work. DraftMs covers embedding, six draft layers, shared-head logits, top-16 and the trained selector.
    public double BootstrapMs, DraftMs, VerifyMs, IngestMs, SnapshotMs, ReplayMs, CommitMs, ArMs;
    public long[] AcceptedHistogram = new long[Dflash2Geometry.MaxBlock];
    public string ToLogLine(int block) => FormattableString.Invariant(
        $"block={block} cycles={Cycles} proposed={Proposed} accepted={Accepted} rejectedCycles={RejectedCycles} fullAccept={FullAcceptCycles} arSteps={ArSteps} replays={Replays} contextRows={ContextRows} acceptedHistogram={string.Join('/', AcceptedHistogram.Take(block))} bootstrapMs={BootstrapMs:F3} draftMs={DraftMs:F3} verifyMs={VerifyMs:F3} ingestMs={IngestMs:F3} snapshotMs={SnapshotMs:F3} replayMs={ReplayMs:F3} commitMs={CommitMs:F3} arMs={ArMs:F3}");
}

// Greedy speculative generation over DFlash2 block drafts. One instance serves one request.
public sealed class Dflash2Generation
{
    private readonly IDflash2Target target; private readonly IDflash2Drafter drafter; private readonly Dflash2Options options; private readonly Func<double> clock;
    private readonly Func<int, bool> isEnd; private readonly int maxTokens;
    private int anchor = -1, committed, emitted; private readonly List<int> output = [];
    private bool started, finished, anchorEmitted;
    public Dflash2Stats Stats { get; } = new();
    public IReadOnlyList<int> Output => output;
    public string? Finish { get; private set; }
    // The pending target token: emitted (unless EOS) but not yet consumed by the target.
    public int Anchor => anchor;
    // Number of positions the target has consumed (and the drafter's committed context covers).
    public int Committed => committed;
    public int Block => options.Block;

    public Dflash2Generation(IDflash2Target target, IDflash2Drafter drafter, Dflash2Options options, int maxTokens, Func<int, bool> isEnd, Func<double> clock)
    {
        options.Validate();
        if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));
        (this.target, this.drafter, this.options, this.maxTokens, this.isEnd, this.clock) = (target, drafter, options, maxTokens, isEnd, clock);
    }

    private bool Ends(int token) => !options.IgnoreEos && isEnd(token);

    // Prefill plus bounded bootstrap of the drafter's context from the target feature rows. All of it is prefill (TTFT) time.
    public void Prefill(int[] ids, CancellationToken ct)
    {
        if (started) throw new InvalidOperationException("Generation already started.");
        if (ids.Length < 1) throw new ArgumentException("Prompt must not be empty.", nameof(ids));
        started = true; drafter.Reset();
        // The target consumes at most prompt + maxTokens - 1 positions (the last emitted token is never consumed), so no slack beyond the first block.
        target.EnsureCapacity(ids.Length + Math.Min(maxTokens - 1, Dflash2Geometry.MaxBlock));
        var bootstrap = maxTokens > 1;
        // Only the last Window-1 prompt rows can ever be visible to a draft query (sliding window), so earlier rows are skipped.
        var first = Math.Max(0, ids.Length - (Dflash2Geometry.Window - 1));
        anchor = target.Prefill(ids, (offset, rows) =>
        {
            if (!bootstrap) return;
            var begin = clock();
            var from = Math.Max(offset, first); var to = offset + rows;
            if (to > from) { drafter.Ingest(from - offset, to - from, from); Stats.ContextRows += to - from; }
            Stats.BootstrapMs += clock() - begin;
        }, ct);
        committed = ids.Length;
        if (bootstrap && drafter.ContextEnd != ids.Length) throw new InvalidOperationException("Bootstrap must leave the drafter context at the prompt end.");
    }

    // Produces the next block of newly ready outputs, or false when generation has finished with nothing more to emit.
    public bool TryNext(CancellationToken ct, out Dflash2Block block)
    {
        block = null!;
        if (!started) throw new InvalidOperationException("Call Prefill first.");
        if (finished) return false;
        ct.ThrowIfCancellationRequested();
        if (!anchorEmitted)
        {
            anchorEmitted = true;
            if (Ends(anchor)) { finished = true; Finish = "eos"; return false; }
            output.Add(anchor); emitted = 1;
            block = new([anchor], clock(), true);
            if (emitted >= maxTokens) { finished = true; Finish = "length"; }
            return true;
        }
        var k = Math.Min(options.Block - 1, maxTokens - emitted - 1);
        return k < 1 || options.FullBlocksOnly && k + 1 < options.Block ? ArStep(out block) : Cycle(k, out block);
    }

    private bool ArStep(out Dflash2Block block)
    {
        target.EnsureCapacity(committed + 1);
        var begin = clock();
        var next = target.Forward(anchor, committed);
        committed++;
        Stats.ArSteps++; Stats.ArMs += clock() - begin;
        anchor = next;
        if (Ends(next)) { finished = true; Finish = "eos"; block = null!; return false; }
        output.Add(next); emitted++;
        block = new([next], clock(), false);
        if (emitted >= maxTokens) { finished = true; Finish = "length"; }
        return true;
    }

    private bool Cycle(int k, out Dflash2Block block)
    {
        if (drafter.ContextEnd != committed) throw new InvalidOperationException($"Drafter context end {drafter.ContextEnd} differs from committed target position {committed}.");
        target.EnsureCapacity(committed + k + 1);
        var begin = clock();
        // The drafted block is clamped to the remaining output (k proposals), never drafted at full width and trimmed afterwards.
        var blockTokens = new int[k + 1]; blockTokens[0] = anchor; Array.Fill(blockTokens, Dflash2Geometry.MaskToken, 1, k);
        var proposals = drafter.Propose(blockTokens);
        Stats.DraftMs += clock() - begin;
        if (proposals.Length != k) throw new InvalidOperationException("Drafter returned the wrong proposal count.");
        begin = clock();
        target.Snapshot();
        Stats.SnapshotMs += clock() - begin;
        begin = clock();
        var tokens = new int[k + 1]; tokens[0] = anchor; Array.Copy(proposals, 0, tokens, 1, k);
        var argmax = target.Verify(tokens);
        Stats.VerifyMs += clock() - begin;
        if (argmax.Length != k + 1) throw new InvalidOperationException("Verifier returned the wrong row count.");
        var accepted = 0;
        while (accepted < k && tokens[accepted + 1] == argmax[accepted]) accepted++;
        Stats.Cycles++; Stats.Proposed += k; Stats.Accepted += accepted; Stats.AcceptedHistogram[accepted]++;
        if (accepted == k) Stats.FullAcceptCycles++; else Stats.RejectedCycles++;
        // Accepted proposals then the target's correction/bonus argmax[accepted].
        var produced = new int[accepted + 1];
        for (var i = 0; i < accepted; i++) produced[i] = tokens[i + 1];
        produced[accepted] = argmax[accepted];
        var cut = produced.Length; var eos = false;
        for (var i = 0; i < produced.Length; i++) if (Ends(produced[i])) { cut = i; eos = true; break; }
        // Target consumes the anchor plus every emitted token that precedes the new pending anchor.
        var commit = eos ? cut + 1 : accepted + 1;
        var emit = produced.AsSpan(0, cut).ToArray();
        var willFinish = eos || emitted + emit.Length >= maxTokens;
        // Native commit publishes the selected prefix before the drafter can
        // ingest it. Legacy replay overwrites taps, so its ingest must precede rollback.
        void Ingest()
        {
            if (willFinish) return;
            var start = clock(); drafter.Ingest(0, commit, committed);
            Stats.IngestMs += clock() - start; Stats.ContextRows += commit;
        }
        if (!target.DirectCommit) Ingest();
        begin = clock(); target.CommitPrefix(tokens, commit);
        var commitMs = clock() - begin; Stats.CommitMs += commitMs;
        if (!target.DirectCommit && commit < k + 1) { Stats.Replays++; Stats.ReplayMs += commitMs; }
        if (target.DirectCommit) Ingest();
        committed += commit;
        output.AddRange(emit);
        emitted += emit.Length;
        anchor = eos ? produced[cut] : produced[^1];
        // One timestamp for every validated token of the block, taken after all draft/verify/ingest/replay work.
        block = new(emit, clock(), false);
        if (eos) { finished = true; Finish = "eos"; return emit.Length > 0; }
        if (emitted >= maxTokens) { finished = true; Finish = "length"; return true; }
        return true;
    }
}
