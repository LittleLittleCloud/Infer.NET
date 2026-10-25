namespace DeepInfer.Backends.Mtp;

// Target-side operations the MTP engine needs. Implementations must be exact: Verify rows must equal
// single-token decode (argmax and hidden), and RestoreAndReplay must reproduce the committed single-token state.
public interface IMtpTarget
{
    // Prefill the prompt and return the first target anchor (argmax of the last row). The sink receives every chunk with
    // its post-final-norm hidden rows ([tokens.Length * 2048]).
    int Prefill(int[] prompt, Action<int[], float[]> sink, CancellationToken ct);
    void EnsureCapacity(int tokens);
    // Single-token decode of `token` at `position` (the committed length); returns the target argmax.
    int Forward(int token, int position);
    // Capture the committed state at the current position (before a verification).
    void Snapshot();
    // Batched exact verification of [anchor, d1..dk]: argmax per row and hidden rows. Advances the target by tokens.Length.
    (int[] Argmax, float[] Hidden) Verify(int[] tokens);
    // Restore the last snapshot and re-run the committed prefix so the target holds exactly that prefix.
    void RestoreAndReplay(int[] committed);
}

// Draft-side operations. ValidRows counts target-conditioned rows only.
public interface IMtpDrafter
{
    int ValidRows { get; }
    void Reset();
    void EnsureCapacity(int tokens);
    // Rows pair tokens[i] with hidden row i at position ValidRows + i; advances ValidRows. classify returns the draft token for the last row (else -1).
    int CatchUp(int[] tokens, float[] hidden, int rows, bool classify);
    // Provisional recurrent step from the previous draft output; never advances ValidRows.
    int Provisional(int token);
}

public sealed record MtpOptions(int Width, bool IgnoreEos = false)
{
    public void Validate()
    {
        if (Width is not (1 or 2)) throw new ArgumentOutOfRangeException(nameof(Width), "MTP width must be 1 or 2.");
    }
}

public sealed record MtpBlock(int[] Tokens, double ReadyMs, bool Anchor);

public sealed class MtpStats
{
    public long Cycles, Proposed, Accepted, RejectedCycles, ArSteps, FullAcceptCycles, Replays;
    public double BootstrapMs, FirstDraftMs, DraftMs, VerifyMs, CatchUpMs, SnapshotMs, ReplayMs, ArMs;
    public long[] AcceptedHistogram = new long[3];
    public string ToLogLine() => FormattableString.Invariant(
        $"cycles={Cycles} proposed={Proposed} accepted={Accepted} rejectedCycles={RejectedCycles} fullAccept={FullAcceptCycles} arSteps={ArSteps} replays={Replays} acceptedHistogram={string.Join('/', AcceptedHistogram)} bootstrapMs={BootstrapMs:F3} firstDraftMs={FirstDraftMs:F3} draftMs={DraftMs:F3} verifyMs={VerifyMs:F3} catchUpMs={CatchUpMs:F3} snapshotMs={SnapshotMs:F3} replayMs={ReplayMs:F3} arMs={ArMs:F3}");
}

// Greedy self-speculative generation over official MTP drafts. One instance serves one request.
public sealed class MtpGeneration
{
    private readonly IMtpTarget target; private readonly IMtpDrafter drafter; private readonly MtpOptions options; private readonly Func<double> clock;
    private readonly Func<int, bool> isEnd; private readonly int maxTokens;
    private int anchor = -1, committed, emitted; private readonly List<int> output = [];
    private int[] drafts = []; private bool started, finished, anchorEmitted, firstDraftPending, stale;
    // Hidden row of the previous prefill chunk's final position, carried across chunk boundaries.
    private float[]? carry;
    public MtpStats Stats { get; } = new();
    public IReadOnlyList<int> Output => output;
    public string? Finish { get; private set; }
    // The pending target token: emitted (unless EOS) but not yet consumed by the target.
    public int Anchor => anchor;
    // Number of positions the target has consumed.
    public int Committed => committed;

    public MtpGeneration(IMtpTarget target, IMtpDrafter drafter, MtpOptions options, int maxTokens, Func<int, bool> isEnd, Func<double> clock)
    {
        options.Validate();
        if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));
        (this.target, this.drafter, this.options, this.maxTokens, this.isEnd, this.clock) = (target, drafter, options, maxTokens, isEnd, clock);
    }

    private bool Ends(int token) => !options.IgnoreEos && isEnd(token);

    // Prefill plus bounded bootstrap of the draft cache from target hidden rows. All of it is prefill time.
    public void Prefill(int[] ids, CancellationToken ct)
    {
        if (started) throw new InvalidOperationException("Generation already started.");
        if (ids.Length < 1) throw new ArgumentException("Prompt must not be empty.", nameof(ids));
        started = true; drafter.Reset();
        var reserve = ids.Length + Math.Min(maxTokens, 8) + 2;
        target.EnsureCapacity(reserve); drafter.EnsureCapacity(reserve);
        var bootstrap = maxTokens > 1;
        anchor = target.Prefill(ids, (chunk, hidden) =>
        {
            if (!bootstrap) return;
            var begin = clock();
            Bootstrap(chunk, hidden);
            Stats.BootstrapMs += clock() - begin;
        }, ct);
        committed = ids.Length;
        // Bootstrap ends at draft cursor P-1 with the final target hidden row retained. The anchor-conditioned
        // first draft is decode work: it runs in the first cycle, after the initial anchor has been emitted.
        firstDraftPending = bootstrap && !Ends(anchor);
        if (!firstDraftPending) carry = null;
        else if (carry is null || drafter.ValidRows != ids.Length - 1) throw new InvalidOperationException("Bootstrap must leave the draft cursor at P-1 with the final hidden row.");
    }

    private void Bootstrap(int[] chunk, float[] hidden)
    {
        var rows = chunk.Length;
        // Row p pairs token p+1 with hidden p; the previous chunk's last hidden row pairs with this chunk's first token.
        var tokens = new List<int>(); var rowsHidden = new List<float>();
        if (carry is not null) { tokens.Add(chunk[0]); rowsHidden.AddRange(carry); }
        for (var i = 0; i + 1 < rows; i++) { tokens.Add(chunk[i + 1]); rowsHidden.AddRange(new ArraySegment<float>(hidden, i * 2048, 2048)); }
        carry = hidden.AsSpan((rows - 1) * 2048, 2048).ToArray();
        for (var offset = 0; offset < tokens.Count; offset += 32)
        {
            var count = Math.Min(32, tokens.Count - offset);
            drafter.CatchUp(tokens.GetRange(offset, count).ToArray(), rowsHidden.GetRange(offset * 2048, count * 2048).ToArray(), count, false);
        }
    }

    private void PlanDrafts(int remaining)
    {
        var width = Math.Min(options.Width, remaining - 1);
        if (width < 1) { drafts = []; return; }
        if (drafts.Length < 1) throw new InvalidOperationException("MTP draft missing after catch-up.");
        if (width == 2 && drafts.Length < 2)
        {
            var begin = clock();
            drafts = [drafts[0], drafter.Provisional(drafts[0])];
            Stats.DraftMs += clock() - begin;
        }
        else drafts = drafts[..width];
    }

    // Produces the next block of newly ready outputs, or false when generation has finished with nothing more to emit.
    public bool TryNext(CancellationToken ct, out MtpBlock block)
    {
        block = null!;
        if (!started) throw new InvalidOperationException("Call Prefill first.");
        if (finished) return false;
        if (stale) throw new InvalidOperationException("Generation was cancelled after a commit; the draft cache is stale. Start a new request (it resets draft state).");
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
        var k = Math.Min(options.Width, maxTokens - emitted - 1);
        return k < 1 ? ArStep(out block) : Cycle(k, ct, out block);
    }

    private bool ArStep(out MtpBlock block)
    {
        target.EnsureCapacity(committed + 2); drafter.EnsureCapacity(committed + 2);
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

    private void FirstDraft()
    {
        var begin = clock();
        // The final prompt row pairs with the verified anchor and yields the first draft.
        drafts = [drafter.CatchUp([anchor], carry!, 1, true)];
        carry = null; firstDraftPending = false;
        Stats.FirstDraftMs += clock() - begin;
        PlanDrafts(maxTokens - emitted);
    }

    private bool Cycle(int k, CancellationToken ct, out MtpBlock block)
    {
        if (firstDraftPending)
        {
            ct.ThrowIfCancellationRequested();
            FirstDraft();
        }
        if (drafts.Length != k) throw new InvalidOperationException($"Expected {k} drafts, have {drafts.Length}.");
        if (drafter.ValidRows != committed) throw new InvalidOperationException($"Draft cursor {drafter.ValidRows} differs from committed target position {committed}.");
        target.EnsureCapacity(committed + k + 2); drafter.EnsureCapacity(committed + k + 2);
        // Cancellation is honoured only where the target and draft state stay consistent: before the snapshot/verify,
        // and after the committed state has been restored/replayed.
        ct.ThrowIfCancellationRequested();
        var begin = clock();
        target.Snapshot();
        Stats.SnapshotMs += clock() - begin;
        begin = clock();
        var tokens = new int[k + 1]; tokens[0] = anchor; drafts.CopyTo(tokens, 1);
        var (argmax, hidden) = target.Verify(tokens);
        Stats.VerifyMs += clock() - begin;
        if (argmax.Length != k + 1 || hidden.Length != (k + 1) * 2048) throw new InvalidOperationException("Verifier returned the wrong row count.");
        var accepted = 0;
        while (accepted < k && drafts[accepted] == argmax[accepted]) accepted++;
        Stats.Cycles++; Stats.Proposed += k; Stats.Accepted += accepted; Stats.AcceptedHistogram[accepted]++;
        if (accepted == k) Stats.FullAcceptCycles++; else Stats.RejectedCycles++;
        // Accepted proposals then the target's correction/bonus argmax[accepted].
        var produced = new int[accepted + 1];
        for (var i = 0; i < accepted; i++) produced[i] = drafts[i];
        produced[accepted] = argmax[accepted];
        var cut = produced.Length; var eos = false;
        for (var i = 0; i < produced.Length; i++) if (Ends(produced[i])) { cut = i; eos = true; break; }
        // Target consumes the anchor plus every emitted token that precedes the new pending anchor.
        var commit = eos ? cut + 1 : accepted + 1;
        if (commit < k + 1)
        {
            begin = clock();
            target.RestoreAndReplay(tokens.AsSpan(0, commit).ToArray());
            Stats.Replays++; Stats.ReplayMs += clock() - begin;
        }
        committed += commit;
        var emit = produced.AsSpan(0, cut).ToArray();
        output.AddRange(emit);
        emitted += emit.Length;
        anchor = eos ? produced[cut] : produced[^1];
        block = new(emit, clock(), false);
        if (eos) { finished = true; Finish = "eos"; return emit.Length > 0; }
        if (emitted >= maxTokens) { finished = true; Finish = "length"; return true; }
        if (ct.IsCancellationRequested) { stale = true; ct.ThrowIfCancellationRequested(); }
        var left = maxTokens - emitted;
        if (left > 1)
        {
            // Recondition the draft cache from the accepted target hidden rows, never from draft-generated state.
            begin = clock();
            var rows = accepted + 1;
            var draft = drafter.CatchUp(produced, hidden.AsSpan(0, rows * 2048).ToArray(), rows, true);
            Stats.CatchUpMs += clock() - begin;
            drafts = [draft];
            PlanDrafts(left);
        }
        else drafts = [];
        return true;
    }
}
