namespace DeepInfer.Core;

// Diagnostics of the synthetic speculative-state correctness reference (Issue #72, S0).
// Proposals come from an oracle, not a draft model. These counters validate state ownership,
// verification and rollback; they are not an acceleration or draft-quality measurement.
//
// Matched counts proposals equal to the target's own prediction before the first mismatch. Accepted counts only the
// matched proposals that were committed (emitted); an end token inside a matched run discards that token and everything
// after it, so Accepted can be smaller than Matched. Corrected/Bonus count the target token after the last committed
// input and are zero for a cycle truncated by an end token, because that token is the pending end token and is never emitted.
public sealed record SpeculativeCycle(int Cycle, int StartPosition, int Proposed, int Verified, int Matched, int Accepted, int Corrected,
    int Bonus, int Emitted, int CommittedInputs, bool EosTruncated, int RejectOffset, int VerifyForwards, int ReplayForwards,
    double SnapshotMs, double RestoreMs, double ReplayMs, double VerifyMs, double OracleMs, int OracleForwards);

// Resources the SOP timings object would otherwise carry. Enabled results have no N-1 decode timing, so these are
// reported separately and never combined into a throughput.
public sealed record SpeculativeResources(double PrefillMs, int ContextCapacity, long KvCacheBytes, long MetalAllocatedBytes,
    long SnapshotHostBytes, string DecodeAttention, int ScratchTokens, int PrefillChunkTokens);

public sealed record SpeculativeDiagnostics(string Mode, string Pattern, int BlockTokens, string Oracle, int Cycles,
    long Proposed, long Verified, long Matched, long Accepted, long Corrected, long Bonus, long Emitted, long EosTruncations,
    long VerifyForwards, long ReplayForwards, long OracleForwards, double SnapshotMs, double RestoreMs, double ReplayMs, double VerifyMs,
    double OracleMs, double FirstEmissionMs, double EmissionLifecycleMs, string FinishReason, string Note,
    SpeculativeCycle[]? Detail = null, SpeculativeResources? Resources = null);
