using System.Diagnostics;
using DeepInfer.Core;
namespace DeepInfer.Backends.Speculative;

// A transactional greedy decode target: one forward consumes one token at Position and returns the
// target's argmax for the next position. Capture/Restore bracket exactly one open speculative cycle.
public interface ISpeculativeTarget
{
    int Position { get; }
    int Consume(int token);
    void EnsureCapacity(int tokens);
    double Capture();
    double Restore();
    // The target state is undefined after a failed or cancelled cycle and must be reset by its owner.
    void Abandon();
}

public sealed class SpeculativeStateException(string message) : InvalidOperationException(message);

// Synthetic proposal pattern. Offset k means proposal k is the first wrong one (0..7);
// an offset at or beyond the block length means every proposal is correct.
public sealed record SyntheticPattern(string Kind, int Offset = 0)
{
    public const int MaxBlockTokens = 8;
    public string Name => Kind == "offset" ? $"offset:{Offset}" : Kind;
    public static SyntheticPattern Parse(string text)
    {
        switch (text)
        {
            case "accept-all": case "reject-all": case "mixed": return new(text);
        }
        if (text.StartsWith("offset:", StringComparison.Ordinal) && int.TryParse(text.AsSpan(7), System.Globalization.NumberStyles.None,
            System.Globalization.CultureInfo.InvariantCulture, out var offset) && offset < MaxBlockTokens) return new("offset", offset);
        throw new ArgumentException("Speculative synthetic pattern must be accept-all, reject-all, offset:0..7 or mixed.");
    }
    // Returns the proposal index to corrupt, or count when the whole block is correct.
    public int RejectOffset(int cycle, int count) => Kind switch
    {
        "accept-all" => count,
        "reject-all" => 0,
        "offset" => Math.Min(Offset, count),
        _ => (cycle * 3 + 1) % (count + 1),
    };
}

public sealed record OracleTruth(int[] Tokens, double Milliseconds, int Forwards);

// Supplies the target's own greedy continuation: Tokens[i] is the prediction after consuming
// [pending, truth[0], ..] element i, so Tokens has count+1 entries and Tokens[i] is also proposal i.
public interface IProposalOracle
{
    string Name { get; }
    OracleTruth Truth(ISpeculativeTarget target, int pending, int emitted, int count);
}

// Truth from a saved baseline output (token ids). A mismatching pending token is a real divergence.
public sealed class ReferenceOracle(int[] baseline) : IProposalOracle
{
    public string Name => "saved-baseline-token-ids";
    public OracleTruth Truth(ISpeculativeTarget target, int pending, int emitted, int count)
    {
        if (emitted + count + 2 > baseline.Length)
            throw new SpeculativeStateException("Saved baseline is shorter than the requested block.");
        if (baseline[emitted] != pending)
            throw new SpeculativeStateException($"Pending token {pending} differs from saved baseline {baseline[emitted]} at output {emitted}.");
        return new(baseline.AsSpan(emitted + 1, count + 1).ToArray(), 0, 0);
    }
}

// Truth from the target itself: capture, serial rollout, restore. Its forwards are reported separately
// so they are never mistaken for the verification cost.
public sealed class RolloutOracle : IProposalOracle
{
    public string Name => "target-rollout";
    public OracleTruth Truth(ISpeculativeTarget target, int pending, int emitted, int count)
    {
        var timer = Stopwatch.StartNew();
        var truth = new int[count + 1];
        target.Capture();
        var token = pending;
        for (var i = 0; i <= count; i++) { truth[i] = target.Consume(token); token = truth[i]; }
        target.Restore();
        return new(truth, timer.Elapsed.TotalMilliseconds, count + 1);
    }
}

// TerminalPosition is the target position at return and FinalPending the target's still-unconsumed prediction.
// Natural end: every emitted token is consumed (position = prompt + output) and FinalPending is the end token.
// Length cap: the last emitted token is the unconsumed one (position = prompt + output - 1), like the standard loop.
public sealed record SpeculativeRun(int[] Output, string FinishReason, SpeculativeDiagnostics Diagnostics, int TerminalPosition, int FinalPending);

public static class SpeculativeEngine
{
    public const string Note = "Synthetic proposal correctness reference: proposals come from an oracle, not a draft model. Counters validate state rollback only and are not a performance or acceptance-quality measurement.";
    private const int Vocabulary = 248320;

    // Deterministic wrong token; never equal to the truth.
    public static int Corrupt(int truth, int cycle, int index) => (int)((truth + 1L + (cycle * 131L + index * 17L) % 1000) % Vocabulary);

    // `pending` is the target's prediction for the first output token and is not yet consumed. Position at entry is
    // the prompt length. `checkpoints` are output counts at which the engine guarantees a cycle boundary and calls
    // `checkpoint` with the target in baseline-equivalent state (consumed == prompt + emitted).
    public static async Task<SpeculativeRun> Run(ISpeculativeTarget target, int pending, int maxTokens, bool ignoreEos,
        Func<int, bool> isEnd, SyntheticPattern pattern, int blockTokens, IProposalOracle oracle, CancellationToken ct,
        Func<int, ValueTask>? emit = null, IReadOnlyList<int>? checkpoints = null, Action<int, int>? checkpoint = null,
        bool keepDetail = true)
    {
        if (blockTokens is < 1 or > SyntheticPattern.MaxBlockTokens) throw new ArgumentOutOfRangeException(nameof(blockTokens), "Block size must be 1..8.");
        if (maxTokens < 1) throw new ArgumentOutOfRangeException(nameof(maxTokens));
        var promptLength = target.Position;
        var output = new List<int>(maxTokens);
        var detail = new List<SpeculativeCycle>(); var cycles = 0;
        var points = new Queue<int>((checkpoints ?? []).Where(p => p >= 0 && p < maxTokens).Distinct().Order());
        var finish = "length";
        double snapshotMs = 0, restoreMs = 0, replayMs = 0, verifyMs = 0, oracleMs = 0, first = 0;
        long proposed = 0, verified = 0, matched = 0, accepted = 0, corrected = 0, bonus = 0, eosTruncations = 0, verifyForwards = 0, replayForwards = 0, oracleForwards = 0;
        var lifecycle = Stopwatch.StartNew();
        try
        {
            for (var cycle = 0; ; cycle++)
            {
                ct.ThrowIfCancellationRequested();
                var emitted = output.Count;
                if (target.Position != promptLength + emitted)
                    throw new SpeculativeStateException($"Target position {target.Position} differs from prompt {promptLength} plus emitted {emitted}.");
                while (points.Count > 0 && points.Peek() == emitted) { points.Dequeue(); checkpoint?.Invoke(emitted, pending); }
                var remaining = maxTokens - emitted;
                if (remaining <= 0) break;
                // The pending token is already the target's prediction: emit it before spending any forward.
                if (isEnd(pending) && !ignoreEos) { finish = "eos"; break; }
                output.Add(pending);
                if (emitted == 0) first = lifecycle.Elapsed.TotalMilliseconds;
                if (emit is not null) await emit(pending);
                if (remaining == 1) break;
                cycles++;

                var count = Math.Max(0, Math.Min(blockTokens, remaining - 2));
                if (points.Count > 0) count = Math.Min(count, points.Peek() - emitted - 1);
                target.EnsureCapacity(promptLength + emitted + count + 1);
                if (count == 0)
                {
                    var plain = Stopwatch.StartNew();
                    pending = target.Consume(output[^1]);
                    verifyMs += plain.Elapsed.TotalMilliseconds; verifyForwards++;
                    if (keepDetail) detail.Add(new(cycle, promptLength + emitted, 0, 0, 0, 0, 0, 0, 1, 1, false, -1, 1, 0, 0, 0, 0, plain.Elapsed.TotalMilliseconds, 0, 0));
                    continue;
                }

                var truth = oracle.Truth(target, pending, emitted, count);
                if (truth.Tokens.Length != count + 1 || target.Position != promptLength + emitted)
                    throw new SpeculativeStateException("Proposal oracle left the target in an unexpected state.");
                oracleMs += truth.Milliseconds; oracleForwards += truth.Forwards;
                var reject = pattern.RejectOffset(cycle, count);
                var drafts = new int[count];
                for (var i = 0; i < count; i++) drafts[i] = i < reject ? truth.Tokens[i] : Corrupt(truth.Tokens[i], cycle, i);
                var inputs = new int[count + 1]; inputs[0] = pending; drafts.CopyTo(inputs, 1);

                var snapshot = target.Capture(); snapshotMs += snapshot;
                var verify = Stopwatch.StartNew();
                var predictions = new int[count + 1];
                for (var i = 0; i <= count; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    predictions[i] = target.Consume(inputs[i]);
                }
                var verifyElapsed = verify.Elapsed.TotalMilliseconds; verifyMs += verifyElapsed; verifyForwards += count + 1;
                // Inputs up to and including the first wrong proposal are exact, so their predictions must equal the oracle.
                for (var i = 0; i <= Math.Min(reject, count); i++)
                    if (predictions[i] != truth.Tokens[i]) throw new SpeculativeStateException($"Verify prediction {i} ({predictions[i]}) differs from oracle truth ({truth.Tokens[i]}).");
                var accept = 0;
                while (accept < count && drafts[accept] == predictions[accept]) accept++;
                var rejected = accept < count;
                if (rejected != reject < count || rejected && accept != reject)
                    throw new SpeculativeStateException("Verification outcome differs from the synthetic pattern.");

                // An end token inside the matched run is never committed: the state must stop right after the input that
                // produced it, so the end token stays pending exactly as in the standard loop.
                var endAt = -1;
                if (!ignoreEos) for (var i = 0; i < accept; i++) if (isEnd(drafts[i])) { endAt = i; break; }
                var keep = (endAt >= 0 ? endAt : accept) + 1; // inputs that stay consumed

                double restore = 0, replay = 0; var replayCount = 0;
                if (keep != count + 1)
                {
                    restore = target.Restore(); restoreMs += restore;
                    if (target.Position != promptLength + emitted)
                        throw new SpeculativeStateException("Restore did not return to the cycle start position.");
                    var redo = Stopwatch.StartNew();
                    var last = -1;
                    for (var i = 0; i < keep; i++) { ct.ThrowIfCancellationRequested(); last = target.Consume(inputs[i]); replayCount++; }
                    replay = redo.Elapsed.TotalMilliseconds; replayMs += replay; replayForwards += replayCount;
                    if (last != predictions[keep - 1])
                        throw new SpeculativeStateException($"Replay prediction {last} differs from verified prediction {predictions[keep - 1]}.");
                }
                if (target.Position != promptLength + emitted + keep)
                    throw new SpeculativeStateException("Target position disagrees with committed tokens after the cycle.");
                pending = predictions[keep - 1];

                var committed = keep - 1;
                for (var i = 0; i < committed; i++)
                {
                    output.Add(drafts[i]);
                    if (emit is not null) await emit(drafts[i]);
                }
                var truncated = endAt >= 0;
                var fix = rejected && !truncated ? 1 : 0; var extra = !rejected && !truncated ? 1 : 0;
                proposed += count; verified += accept + (rejected ? 1 : 0); matched += accept; accepted += committed;
                corrected += fix; bonus += extra; if (truncated) eosTruncations++;
                if (keepDetail)
                    detail.Add(new(cycle, promptLength + emitted, count, accept + (rejected ? 1 : 0), accept, committed, fix, extra,
                        1 + committed, keep, truncated, rejected ? accept : -1, count + 1, replayCount, snapshot, restore, replay, verifyElapsed,
                        truth.Milliseconds, truth.Forwards));
                if (truncated) { finish = "eos"; break; }
            }
        }
        catch { target.Abandon(); throw; }
        var expectedPosition = promptLength + (finish == "eos" ? output.Count : Math.Max(0, output.Count - 1));
        if (target.Position != expectedPosition)
        {
            target.Abandon();
            throw new SpeculativeStateException($"Terminal target position {target.Position} differs from the standard loop's {expectedPosition} ({finish}).");
        }
        var diagnostics = new SpeculativeDiagnostics("synthetic-proposal-correctness-reference", pattern.Name, blockTokens, oracle.Name, cycles,
            proposed, verified, matched, accepted, corrected, bonus, output.Count, eosTruncations, verifyForwards, replayForwards, oracleForwards,
            snapshotMs, restoreMs, replayMs, verifyMs, oracleMs, first, lifecycle.Elapsed.TotalMilliseconds, finish, Note,
            keepDetail ? detail.ToArray() : null);
        return new(output.ToArray(), finish, diagnostics, target.Position, pending);
    }
}
