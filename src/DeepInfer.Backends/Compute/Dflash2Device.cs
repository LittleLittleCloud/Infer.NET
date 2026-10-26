using System.Runtime.InteropServices;
using DeepInfer.Backends.Models;
using Microsoft.Win32.SafeHandles;
namespace DeepInfer.Backends.Compute;

internal sealed class Dflash2Handle : SafeHandleZeroOrMinusOneIsInvalid
{
    // The native child dereferences its graph on destroy, so the graph handle is ref-counted for the child's whole lifetime
    // and released only after di_dflash_destroy, whatever the finalizer order.
    private readonly GraphHandle graph;
    public Dflash2Handle(nint p, GraphHandle graph) : base(true)
    {
        var added = false; graph.DangerousAddRef(ref added);
        if (!added) throw new ObjectDisposedException(nameof(GraphHandle));
        this.graph = graph; SetHandle(p);
    }
    protected override bool ReleaseHandle() { Dflash2Native.Destroy(handle); graph.DangerousRelease(); return true; }
}

internal static unsafe partial class Dflash2Native
{
    [LibraryImport("deepinfer_metal", EntryPoint = "di_alloc_trace_enable")]
    internal static partial void EnableAllocationTrace(int enabled);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_alloc_trace")]
    internal static partial int AllocTrace(ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_create")]
    internal static partial nint Create(GraphHandle graph, int debug, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_destroy")]
    internal static partial void Destroy(nint dflash);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_tensor")]
    internal static partial int Tensor(Dflash2Handle d, int kind, int index, out nint pointer, out ulong bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_seal")]
    internal static partial int Seal(Dflash2Handle d);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_state")]
    internal static partial int State(Dflash2Handle d, ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_evidence")]
    internal static partial int Evidence(Dflash2Handle d, ulong* values, int count);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_reset")]
    internal static partial int Reset(Dflash2Handle d);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_poison_ring")]
    internal static partial int PoisonRing(Dflash2Handle d);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_export_context")]
    internal static partial int ExportContext(Dflash2Handle d, int layer, float* keys, float* values);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_debug")]
    internal static partial int Debug(Dflash2Handle d, int kind, int layer, void* output, ulong bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_ingest")]
    internal static partial int Ingest(Dflash2Handle d, int tapRow, int rows, int firstPosition, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_propose")]
    internal static partial int Propose(Dflash2Handle d, int* tokens, int block, int* proposals, int* candidates, float* scores, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_dflash_logits")]
    internal static partial int Logits(Dflash2Handle d, int rows, float* output);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_taps_export")]
    internal static partial int ExportTaps(GraphHandle graph, int rows, float* output);
}

public readonly record struct Dflash2ResidencyEvidence(long Submissions, long ResidentSubmissions, long UnregisteredSubmissions)
{
    public bool Safe => Submissions == ResidentSubmissions && UnregisteredSubmissions == 0;
    public string ToLogLine() => FormattableString.Invariant($"DFLASH2_RESIDENCY submissions={Submissions} weightsRegistered={ResidentSubmissions} unregistered={UnregisteredSubmissions} scope=weights-only");
}

/// <summary>Allocation high-water sampled after every tracked allocation and every submission. It is NOT a continuous driver peak and is not comparable to main's point-in-time allocatedBytes.</summary>
public readonly record struct MetalAllocatorTrace(bool Enabled, long PeakBytes, long LastBytes, long Samples, long Allocations, long NewPeakTransitions,
    long Submissions, long SubmissionsAboveWorkingSet, long SubmissionPeakBytes, long RecommendedWorkingSetBytes)
{
    public string ToLogLine() => FormattableString.Invariant(
        $"ALLOC_TRACE enabled={Enabled} sampledPeakBytes={PeakBytes} lastBytes={LastBytes} samples={Samples} allocations={Allocations} newPeaks={NewPeakTransitions} submissions={Submissions} submissionPeakBytes={SubmissionPeakBytes} aboveWorkingSetSubmissions={SubmissionsAboveWorkingSet} recommendedWorkingSetBytes={RecommendedWorkingSetBytes} kind=sampled-not-continuous comparableToMainStats=false");
}

public readonly record struct Dflash2DeviceState(long ContextBegin, long ContextEnd, long AllocatedBytes, long Ingests, long Proposals, long IngestedRows);

// Debug checkpoint kinds, matching DfDebug in gpu_dflash.mm. Layer-indexed except where noted.
public enum Dflash2Checkpoint { CtxRaw, CtxNorm, AttnDyn, AttnPrep, AttnOut, HAttn, MlpDyn, MlpPrep, MlpAct, HOut, Final, HProj, CandIds, CandVals, Scores, Embedding }

public readonly record struct Dflash2Proposal(int[] Tokens, int[] Candidates, float[] Scores, double Ms);

// Device-side DFlash2 drafter. Matrices stay BF16 (read as float), norm scales/base kernels FP32, everything else FP32.
// The target embedding and classifier buffers are shared, not copied.
internal sealed unsafe class Dflash2Device : IDisposable
{
    private readonly Qwen36Gpu owner; private readonly Dflash2Handle handle;
    internal Dflash2Device(Qwen36Gpu owner, Dflash2Handle handle) { this.owner = owner; this.handle = handle; }
    public void Load(Dflash2Weights weights)
    {
        var matrices = Dflash2Names.Matrices;
        for (var i = 0; i < matrices.Count; i++)
        {
            if (Dflash2Native.Tensor(handle, 0, i, out var pointer, out var bytes) != 0) throw new InvalidOperationException("DFlash2 matrix slot rejected.");
            if ((long)bytes != weights.TensorBytes(matrices[i].Name)) throw new InvalidDataException($"DFlash2 tensor {matrices[i].Name} byte size differs from the native layout.");
            weights.ReadBytes(matrices[i].Name, new Span<byte>((void*)pointer, checked((int)bytes)));
        }
        var vectors = Dflash2Names.Vectors;
        for (var i = 0; i < vectors.Count; i++)
        {
            var scale = weights.ReadVector(vectors[i].Name);
            if (Dflash2Native.Tensor(handle, 1, i, out var pointer, out var bytes) != 0 || (long)bytes != scale.Length * 4L) throw new InvalidDataException($"DFlash2 vector slot {vectors[i].Name} rejected.");
            scale.CopyTo(new Span<float>((void*)pointer, scale.Length));
        }
        if (Dflash2Native.Seal(handle) != 0) throw new InvalidOperationException("DFlash2 seal rejected.");
    }
    public static void ConfigureAllocationTrace(bool enabled) => Dflash2Native.EnableAllocationTrace(enabled ? 1 : 0);
    public static MetalAllocatorTrace AllocatorTrace()
    {
        var v = stackalloc ulong[10];
        if (Dflash2Native.AllocTrace(v, 10) != 0) throw new InvalidOperationException("Cannot read allocator trace.");
        return new MetalAllocatorTrace(v[0] == 1, (long)v[1], (long)v[2], (long)v[3], (long)v[4], (long)v[5], (long)v[6], (long)v[7], (long)v[8], (long)v[9]);
    }
    public Dflash2DeviceState State()
    {
        var v = stackalloc ulong[8];
        if (Dflash2Native.State(handle, v, 8) != 0) throw new InvalidOperationException("Cannot read DFlash2 state.");
        return new((long)v[0], (long)v[1], (long)v[2], (long)v[3], (long)v[4], (long)v[5]);
    }
    public Dflash2ResidencyEvidence Evidence()
    {
        var v = stackalloc ulong[3];
        if (Dflash2Native.Evidence(handle, v, 3) != 0) throw new InvalidOperationException("Cannot read DFlash2 residency evidence.");
        return new((long)v[0], (long)v[1], (long)v[2]);
    }
    public void Reset() { if (Dflash2Native.Reset(handle) != 0) throw new InvalidOperationException("DFlash2 reset rejected."); }
    public void PoisonRing() { if (Dflash2Native.PoisonRing(handle) != 0) throw new InvalidOperationException("DFlash2 ring poison rejected."); }
    // Ingests rows of the target's most recent tap capture as committed context starting at absolute position firstPosition.
    public double Ingest(int tapRow, int rows, int firstPosition)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        double ms;
        fixed (byte* e = error)
            if (Dflash2Native.Ingest(handle, tapRow, rows, firstPosition, out ms, e, error.Length) != 0) throw new InvalidOperationException(Qwen36Gpu.ErrorText(error));
        owner.Device.Count(ms, 0);
        return ms;
    }
    public Dflash2Proposal Propose(ReadOnlySpan<int> block)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        var rows = block.Length - 1; var tokens = new int[rows]; var candidates = new int[rows * 16]; var scores = new float[rows * 16];
        double ms;
        fixed (int* b = block) fixed (int* t = tokens) fixed (int* c = candidates) fixed (float* s = scores) fixed (byte* e = error)
            if (Dflash2Native.Propose(handle, b, block.Length, t, c, s, out ms, e, error.Length) != 0) throw new InvalidOperationException(Qwen36Gpu.ErrorText(error));
        owner.Device.Count(ms, 0);
        return new(tokens, candidates, scores, ms);
    }
    public (float[] Keys, float[] Values) ExportContext(int layer)
    {
        var s = State(); var rows = checked((int)(s.ContextEnd - s.ContextBegin));
        var k = new float[rows * 1024]; var v = new float[rows * 1024];
        fixed (float* kp = k) fixed (float* vp = v) if (Dflash2Native.ExportContext(handle, layer, kp, vp) != 0) throw new InvalidOperationException("DFlash2 context export rejected.");
        return (k, v);
    }
    public float[] Checkpoint(Dflash2Checkpoint kind, int layer, int floats)
    {
        var data = new float[floats];
        fixed (float* p = data) if (Dflash2Native.Debug(handle, (int)kind, layer, p, (ulong)floats * 4) != 0) throw new InvalidOperationException("DFlash2 checkpoint unavailable (debug device required).");
        return data;
    }
    public float[] Logits(int rows)
    {
        var data = new float[rows * Dflash2Geometry.Vocabulary];
        fixed (float* p = data) if (Dflash2Native.Logits(handle, rows, p) != 0) throw new InvalidOperationException("DFlash2 logits rejected.");
        return data;
    }
    public void Dispose() => handle.Dispose();
}
