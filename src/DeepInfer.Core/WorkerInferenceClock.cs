namespace DeepInfer.Core;

// Generated-token clocks survive transport cancellation. They describe worker
// readiness, not client receipt or successful application completion.
public sealed record WorkerInferenceMetric(string RequestId, DateTimeOffset StartedAt, DateTimeOffset CompletedAt,
    string Build, int InputTokens, int OutputTokens, double? PrefillMs, int DecodeIntervals, double DecodeMs,
    double WorkerMs, string ExecutionPath, string Kv, string Outcome, string? FinishReason, bool TimingsComplete);

public sealed class WorkerInferenceClock(string requestId, string build, int inputTokens, string executionPath, Func<double> clock, string kv = "fp32")
{
    private readonly DateTimeOffset started = DateTimeOffset.UtcNow;
    private double? prefill;
    private double first, last;
    private int tokens;
    public void PrefillDone(double milliseconds)
    {
        if (!double.IsFinite(milliseconds) || milliseconds < 0) throw new ArgumentOutOfRangeException(nameof(milliseconds));
        prefill = milliseconds;
    }
    public void Ready(int count, double milliseconds)
    {
        if (count < 1 || !double.IsFinite(milliseconds) || milliseconds < 0 || tokens > 0 && milliseconds < last)
            throw new ArgumentOutOfRangeException(nameof(milliseconds));
        if (tokens == 0) first = milliseconds;
        tokens = checked(tokens + count); last = milliseconds;
    }
    public WorkerInferenceMetric Snapshot(string outcome, string? finishReason) => new(requestId, started, DateTimeOffset.UtcNow,
        build, inputTokens, tokens, prefill, Math.Max(0, tokens - 1), tokens > 1 ? last - first : 0,
        clock(), executionPath, kv, outcome, finishReason, prefill.HasValue);
}
