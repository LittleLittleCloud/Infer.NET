using System.Runtime.InteropServices;
using DeepInfer.Backends.Models;
namespace DeepInfer.Backends.Compute;

internal static unsafe partial class MtpNative
{
    [LibraryImport("deepinfer_metal", EntryPoint = "di_alloc_trace_enable")]
    internal static partial void EnableAllocationTrace(int enabled);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_create")]
    internal static partial nint Create(GraphHandle graph, int debug, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_destroy")]
    internal static partial void Destroy(nint mtp);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_tensor")]
    internal static partial int Tensor(MtpHandle mtp, int kind, int index, out nint pointer, out ulong bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_seal")]
    internal static partial int Seal(MtpHandle mtp);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_state")]
    internal static partial int State(MtpHandle mtp, ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_evidence")]
    internal static partial int Evidence(MtpHandle mtp, ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_alloc_trace")]
    internal static partial int AllocTrace(ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_reset")]
    internal static partial int Reset(MtpHandle mtp);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_truncate")]
    internal static partial int Truncate(MtpHandle mtp, int rows);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_reserve")]
    internal static partial int Reserve(MtpHandle mtp, int tokens, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_poison")]
    internal static partial int Poison(MtpHandle mtp, int from, int rows);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_export_kv")]
    internal static partial int ExportKv(MtpHandle mtp, int rows, float* keys, float* values);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_debug")]
    internal static partial int Debug(MtpHandle mtp, int index, void* output, ulong bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_mtp_run")]
    internal static partial int Run(MtpHandle mtp, int* tokens, float* hidden, int count, int baseRow, int commit, int hiddenFromPrevious, int classify,
        out int next, float* outputRow, float* logits, out int dispatches, out double ms, byte* error, int capacity);
}

public readonly record struct MtpResidencyEvidence(long Submissions, long ResidentSubmissions, long UnregisteredSubmissions)
{
    public bool Safe => Submissions == ResidentSubmissions && UnregisteredSubmissions == 0;
    public string ToLogLine() => FormattableString.Invariant($"MTP_RESIDENCY submissions={Submissions} weightsRegistered={ResidentSubmissions} unregistered={UnregisteredSubmissions} scope=weights-only");
}

/// <summary>Sampled allocation high-water (after every tracked allocation and every submission). Not continuous and not comparable to main's point-in-time allocatedBytes.</summary>
public readonly record struct MtpAllocatorTrace(bool Enabled, long PeakBytes, long LastBytes, long Samples, long Allocations, long NewPeakTransitions,
    long Submissions, long SubmissionsAboveWorkingSet, long SubmissionPeakBytes, long RecommendedWorkingSetBytes)
{
    public string ToLogLine() => FormattableString.Invariant(
        $"ALLOC_TRACE enabled={Enabled} sampledPeakBytes={PeakBytes} lastBytes={LastBytes} samples={Samples} allocations={Allocations} newPeaks={NewPeakTransitions} submissions={Submissions} submissionPeakBytes={SubmissionPeakBytes} aboveWorkingSetSubmissions={SubmissionsAboveWorkingSet} recommendedWorkingSetBytes={RecommendedWorkingSetBytes} kind=sampled-not-continuous comparableToMainStats=false");
}

// Device-side MTP drafter. Matrices stay BF16, norm scales FP32; the target embedding/head buffers are shared.
internal sealed unsafe class MtpDevice : IDisposable
{
    private readonly Qwen36Gpu owner; private readonly MtpHandle handle;
    internal MtpDevice(Qwen36Gpu owner, MtpHandle handle) { this.owner = owner; this.handle = handle; }
    public void Load(Qwen36MtpWeights weights)
    {
        for (var i = 0; i < Qwen36MtpNames.Matrices.Length; i++)
        {
            var name = Qwen36MtpNames.Matrices[i];
            if (MtpNative.Tensor(handle, 0, i, out var pointer, out var bytes) != 0) throw new InvalidOperationException("MTP matrix slot rejected.");
            if ((long)bytes != weights.TensorBytes(name)) throw new InvalidDataException($"MTP tensor {name} byte size differs from the native layout.");
            weights.ReadBytes(name, new Span<byte>((void*)pointer, checked((int)bytes)));
        }
        for (var i = 0; i < Qwen36MtpNames.Norms.Length; i++)
        {
            var scale = weights.ReadNorm(Qwen36MtpNames.Norms[i]);
            if (MtpNative.Tensor(handle, 1, i, out var pointer, out var bytes) != 0 || (long)bytes != scale.Length * 4L) throw new InvalidDataException("MTP norm slot rejected.");
            scale.CopyTo(new Span<float>((void*)pointer, scale.Length));
        }
        if (MtpNative.Seal(handle) != 0) throw new InvalidOperationException("MTP seal rejected.");
    }
    public (long ValidRows, long Capacity, long AllocatedBytes, long Runs) State()
    {
        var v = stackalloc ulong[5];
        if (MtpNative.State(handle, v, 5) != 0) throw new InvalidOperationException("Cannot read MTP state.");
        return ((long)v[0], (long)v[1], (long)v[2], (long)v[3]);
    }
    public int ValidRows => (int)State().ValidRows;
    /// <summary>Per-submission residency evidence: every MTP command buffer was committed with all weight buffers registered.</summary>
    public MtpResidencyEvidence Evidence()
    {
        var v = stackalloc ulong[3];
        if (MtpNative.Evidence(handle, v, 3) != 0) throw new InvalidOperationException("Cannot read MTP residency evidence.");
        return new MtpResidencyEvidence((long)v[0], (long)v[1], (long)v[2]);
    }
    /// <summary>Sampled candidate-only allocator trace; enabled by DEEPINFER_METAL_ALLOC_TRACE=1.</summary>
    public static MtpAllocatorTrace AllocatorTrace()
    {
        var v = stackalloc ulong[10];
        if (MtpNative.AllocTrace(v, 10) != 0) throw new InvalidOperationException("Cannot read allocator trace.");
        return new MtpAllocatorTrace(v[0] == 1, (long)v[1], (long)v[2], (long)v[3], (long)v[4], (long)v[5], (long)v[6], (long)v[7], (long)v[8], (long)v[9]);
    }
    public void Reset() { if (MtpNative.Reset(handle) != 0) throw new InvalidOperationException("MTP reset rejected."); }
    public void Truncate(int rows) { if (MtpNative.Truncate(handle, rows) != 0) throw new InvalidOperationException("MTP truncate rejected (cursor can only move down)."); }
    public void Reserve(int tokens)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (byte* e = error) if (MtpNative.Reserve(handle, tokens, e, error.Length) != 0) throw new InvalidOperationException(Qwen36Gpu.ErrorText(error));
    }
    public void Poison(int from, int rows) { if (MtpNative.Poison(handle, from, rows) != 0) throw new InvalidOperationException("MTP poison rejected."); }
    public (float[] Keys, float[] Values) ExportKv(int rows)
    {
        var k = new float[rows * 512]; var v = new float[rows * 512];
        fixed (float* kp = k) fixed (float* vp = v) if (MtpNative.ExportKv(handle, rows, kp, vp) != 0) throw new InvalidOperationException("MTP KV export rejected.");
        return (k, v);
    }
    public byte[] DebugTensor(int index, int bytes)
    {
        var data = new byte[bytes];
        fixed (byte* p = data) if (MtpNative.Debug(handle, index, p, (ulong)bytes) != 0) throw new InvalidOperationException("MTP debug tensor unavailable.");
        return data;
    }
    // baseRow must equal the target-conditioned cursor (native-enforced). hidden is null only for the recurrent provisional step.
    public (int Next, float[] Output, float[]? Logits) Run(int[] tokens, float[]? hidden, int baseRow, bool commit, bool classify, bool fullLogits = false)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        var output = new float[2048]; var logits = fullLogits ? new float[Qwen36Weights.Vocabulary] : null;
        int next;
        fixed (int* t = tokens) fixed (float* h = hidden) fixed (float* o = output) fixed (float* l = logits) fixed (byte* e = error)
        {
            if (MtpNative.Run(handle, t, h, tokens.Length, baseRow, commit ? 1 : 0, hidden is null ? 1 : 0, classify ? 1 : 0,
                out next, o, l, out var dispatches, out var ms, e, error.Length) != 0) throw new InvalidOperationException(Qwen36Gpu.ErrorText(error));
            owner.Device.Count(ms, dispatches);
        }
        return (next, output, logits);
    }
    public void Dispose() => handle.Dispose();
}
