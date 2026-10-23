using System.Runtime.InteropServices;
using DeepInfer.Backends.Models;
using Microsoft.Win32.SafeHandles;
namespace DeepInfer.Backends.Compute;

// Own Metal executor; the native graph retains references to all weight buffers.
internal sealed unsafe class Qwen36Gpu : IDisposable
{
    private readonly GraphHandle handle;
    private readonly QuantizedMetal device;
    private readonly int prefillChunk = Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_CHUNK") switch
    { "32" => 32, "64" => 64, "128" => 128, "256" => 256, "512" => 512, "1024" => 1024, _ => 2048 };
    public Qwen36Gpu(QuantizedMetal device, Qwen36Weights weights, int prefillScratchTokens = 0)
    {
        this.device = device;
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (float* norm = weights.FinalNorm) fixed (byte* e = error)
        {
            var p = GraphNative.Create(device.Context, weights.Embedding.Handle, weights.Classifier.Handle,
                norm, weights.Epsilon, weights.RopeTheta, e, error.Length);
            if (p == 0) throw new InvalidOperationException(Message(error));
            handle = new(p);
        }
        try
        {
            for (var index = 0; index < 40; index++)
            {
                var w = weights.Blocks[index];
                QuantMatrix?[] matrices = [w.Q, w.K, w.V, w.Z, w.A, w.B, w.O, w.Router, w.ExpertGate,
                    w.ExpertUp, w.ExpertDown, w.SharedGate, w.SharedUp, w.SharedDown, w.SharedRoute];
                var pointers = new nint[15]; var retained = new bool[15];
                var parameters = new float[37568];
                w.InputNorm.CopyTo(parameters, 0); w.PostNorm.CopyTo(parameters, 2048);
                w.Conv.CopyTo(parameters, 4096); w.ALog.CopyTo(parameters, 36864); w.DtBias.CopyTo(parameters, 36896);
                w.GatedNorm.CopyTo(parameters, 36928); w.QNorm.CopyTo(parameters, 37056); w.KNorm.CopyTo(parameters, 37312);
                try
                {
                    for (var i = 0; i < 15; i++)
                        if (matrices[i] is { } matrix)
                        { matrix.Handle.DangerousAddRef(ref retained[i]); pointers[i] = matrix.Handle.DangerousGetHandle(); }
                    fixed (nint* m = pointers) fixed (float* p = parameters) fixed (byte* e = error)
                        if (GraphNative.Layer(handle, index, m, p, parameters.Length, e, error.Length) != 0)
                            throw new InvalidOperationException(Message(error));
                }
                finally { for (var i = 14; i >= 0; i--) if (retained[i]) matrices[i]!.Handle.DangerousRelease(); }
            }
            if (Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL") != "0")
                fixed (byte* e = error)
                    if ((prefillScratchTokens == 0 ? GraphNative.PreparePrefill(handle, e, error.Length)
                        : GraphNative.PreparePrefillScratch(handle, prefillScratchTokens, e, error.Length)) != 0)
                        throw new InvalidOperationException(Message(error));
            Reset();
        }
        catch { handle.Dispose(); throw; }
    }
    private static string Message(ReadOnlySpan<byte> error) => System.Text.Encoding.UTF8.GetString(error).TrimEnd('\0');
    public void Reset()
    {
        if (GraphNative.Reset(handle) != 0) throw new InvalidOperationException("GPU graph state could not reset.");
    }
    public void Reserve(int tokens)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (byte* e = error)
            if (GraphNative.Reserve(handle, tokens, e, error.Length) != 0)
                throw new InvalidOperationException(Message(error));
    }
    public void Grow(int tokens)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (byte* e = error)
            if (GraphNative.Grow(handle, tokens, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
    }
    public (int Capacity, long KvBytes, long AllocatedBytes) Stats()
    {
        if (GraphNative.Stats(handle, out var capacity, out var kv, out var allocated) != 0)
            throw new InvalidOperationException("Cannot read GPU graph allocation statistics.");
        return (capacity, kv, allocated);
    }
    public int Forward(int token, int position, bool classifier, float[] logits, Action<int, int, float[]>? trace, bool fullLogits)
    {
        var snapshots = trace is null ? null : new float[40 * 2048];
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        int next;
        fixed (float* y = logits) fixed (float* t = snapshots) fixed (byte* e = error)
        {
            if (GraphNative.Forward(handle, token, position, classifier ? 1 : 0, fullLogits ? y : null, t, out next, out var dispatches, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(Message(error));
            device.Count(ms, dispatches);
        }
        if (classifier && next is < 0 or >= Qwen36Weights.Vocabulary) throw new InvalidOperationException("GPU classifier produced non-finite logits or invalid argmax.");
        if (trace is not null)
            for (var layer = 0; layer < 40; layer++) trace(position, layer, snapshots!.AsSpan(layer * 2048, 2048).ToArray());
        return next;
    }
    public int Prefill(ReadOnlySpan<int> tokens, CancellationToken ct, float[]? logits = null, Action<int, int, float[]>? trace = null,
        Action<int>? progress = null)
    {
        var next = -1;
        Span<byte> error = stackalloc byte[2048];
        for (var offset = 0; offset < tokens.Length; offset += prefillChunk)
        {
            ct.ThrowIfCancellationRequested();
            var chunk = tokens.Slice(offset, Math.Min(prefillChunk, tokens.Length - offset));
            var final = offset + chunk.Length == tokens.Length;
            var snapshots = trace is null ? null : new float[chunk.Length * 40 * 2048];
            error.Clear();
            fixed (int* ids = chunk) fixed (float* y = logits) fixed (float* t = snapshots) fixed (byte* e = error)
            {
                if (GraphNative.Prefill(handle, ids, chunk.Length, final ? 1 : 0, final ? y : null, t,
                    out next, out var dispatches, out var ms, e, error.Length) != 0)
                    throw new InvalidOperationException(Message(error));
                device.Count(ms, dispatches, GraphNative.PrefillSubmissions(handle));
            }
            progress?.Invoke(offset + chunk.Length);
            if (final && next is < 0 or >= Qwen36Weights.Vocabulary)
                throw new InvalidOperationException("GPU prefill produced non-finite logits or invalid argmax.");
            if (trace is not null)
                for (var p = 0; p < chunk.Length; p++) for (var layer = 0; layer < 40; layer++)
                    trace(offset + p, layer, snapshots!.AsSpan((p * 40 + layer) * 2048, 2048).ToArray());
        }
        return next;
    }
    public int Suffix(int[] tokens, float[] hidden, bool final, float[]? logits, float[]? trace = null)
    {
        if (tokens.Length is < 1 or > 512 || hidden.Length != tokens.Length * 2048 ||
            trace is not null && trace.Length != tokens.Length * 40 * 2048) throw new ArgumentException("Invalid split hidden geometry.");
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (int* ids = tokens) fixed (float* h = hidden) fixed (float* y = logits) fixed (float* t = trace) fixed (byte* e = error)
        {
            if (GraphNative.Range(handle, ids, tokens.Length, final ? 1 : 0, 20, 40, h, null, y, t,
                out var next, out var dispatches, out var ms, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
            device.Count(ms, dispatches, GraphNative.PrefillSubmissions(handle));
            return next;
        }
    }
    public void PrefixState(int layer, int kind, int position, byte[] data, bool import)
    {
        fixed (byte* p = data)
            if (GraphNative.PrefixState(handle, layer, kind, position, p, data.Length, import ? 1 : 0) != 0)
                throw new InvalidOperationException("State import/export rejected (identity, position, precision, size or duplicate).");
    }
    public void PrefixKvRange(int layer, int kind, int position, int count, int total, byte[] data)
    {
        fixed (byte* p = data)
            if (GraphNative.PrefixKvRange(handle, layer, kind, position, count, total, p, data.Length) != 0)
                throw new InvalidOperationException("KV range import rejected (identity, position, precision, size, target or duplicate).");
    }
    public void ConfigureFusedFp32Decode(bool enabled)
    {
        if (GraphNative.DecodeFusedFp32(handle, enabled ? 1 : 0) != 0)
            throw new InvalidOperationException("FP32 decode attention configuration rejected; reset and use FP32 KV/prepared scratch.");
    }
    public void Dispose() => handle.Dispose();
}
internal sealed class GraphHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public GraphHandle() : base(true) { }
    internal GraphHandle(nint p) : base(true) { SetHandle(p); }
    protected override bool ReleaseHandle() { GraphNative.Destroy(handle); return true; }
}
internal static unsafe partial class GraphNative
{
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prefill_range")]
    internal static partial int Range(GraphHandle graph, int* tokens, int count, int classifier, int begin, int end,
        float* hidden, float* output, float* logits, float* trace, out int nextToken, out int dispatches, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prefix_kv_range")]
    internal static partial int PrefixKvRange(GraphHandle graph, int layer, int kind, int position, int count, int total, byte* data, int bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_decode_fused_fp32")]
    internal static partial int DecodeFusedFp32(GraphHandle graph, int enabled);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prefix_state")]
    internal static partial int PrefixState(GraphHandle graph, int layer, int kind, int position, byte* data, int bytes, int importState);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_create")]
    internal static partial nint Create(QuantHandle context, QuantHandle embedding, QuantHandle classifier,
        float* norm, float epsilon, float theta, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_layer")]
    internal static partial int Layer(GraphHandle graph, int index, nint* matrices, float* parameters, int count, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_reset")]
    internal static partial int Reset(GraphHandle graph);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_reserve")]
    internal static partial int Reserve(GraphHandle graph, int tokens, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_grow")]
    internal static partial int Grow(GraphHandle graph, int tokens, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_stats")]
    internal static partial int Stats(GraphHandle graph, out int capacity, out long kvBytes, out long allocatedBytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_forward")]
    internal static partial int Forward(GraphHandle graph, int token, int position, int classifier,
        float* logits, float* trace, out int nextToken, out int dispatches, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prefill")]
    internal static partial int Prefill(GraphHandle graph, int* tokens, int count, int classifier,
        float* logits, float* trace, out int nextToken, out int dispatches, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prefill_submissions")]
    internal static partial int PrefillSubmissions(GraphHandle graph);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prepare_prefill")]
    internal static partial int PreparePrefill(GraphHandle graph, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prepare_prefill_scratch")]
    internal static partial int PreparePrefillScratch(GraphHandle graph, int tokens, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_destroy")]
    internal static partial void Destroy(nint graph);
}
