using System.Runtime.InteropServices;
namespace DeepInfer.Backends.Compute;

internal sealed unsafe partial class Qwen36Gpu
{
    public void PrepareTargetBlock(int rows)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (byte* e = error)
            if (GraphNative.PrepareTargetBlock(handle, rows, e, error.Length) != 0)
                throw new InvalidOperationException(Message(error));
    }
    public int[] TargetBlock(int[] tokens, float[]? logits, float[]? trace)
    {
        if (tokens.Length is not (1 or 4 or 8) || logits is not null && logits.Length != tokens.Length * 248320 ||
            trace is not null && trace.Length != tokens.Length * 40 * 2048)
            throw new ArgumentException("Invalid target block geometry.");
        Span<byte> error = stackalloc byte[2048]; error.Clear(); var next = new int[tokens.Length];
        fixed (int* t = tokens) fixed (int* n = next) fixed (float* l = logits) fixed (float* r = trace) fixed (byte* e = error)
        {
            if (GraphNative.TargetBlock(handle, t, tokens.Length, n, l, r, out var dispatches, out var submissions,
                out var ms, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
            device.Count(ms, dispatches, submissions);
        }
        return next;
    }
    public int[] ScalarVerify(int[] tokens)
    {
        if (tokens.Length is < 1 or > 8) throw new ArgumentException("Scalar verification requires 1..8 rows.");
        var next = new int[tokens.Length]; Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (int* t = tokens) fixed (int* n = next) fixed (byte* e = error)
        {
            if (GraphNative.ScalarVerify(handle, t, tokens.Length, n, out var dispatches, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(Message(error));
            device.Count(ms, dispatches, tokens.Length);
        }
        return next;
    }
    public void ScalarTargetTransaction(int[] tokens, int accepted)
    {
        if (tokens.Length is not (1 or 4 or 8) || accepted < 1 || accepted > tokens.Length)
            throw new ArgumentException("Invalid scalar target transaction.");
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (int* t = tokens) fixed (byte* e = error)
        {
            if (GraphNative.ScalarTargetTransaction(handle, t, tokens.Length, accepted, out var dispatches,
                out var submissions, out var ms, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
            device.Count(ms, dispatches, submissions);
        }
    }
    public void CommitTargetBlock(int accepted)
    {
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (byte* e = error)
        {
            if (GraphNative.CommitTargetBlock(handle, accepted, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(Message(error));
            if (accepted > 0) device.Count(ms, 30, 1);
        }
    }
    public (long Bytes, int Position, bool Pending) TargetBlockStats()
    {
        if (GraphNative.TargetBlockStats(handle, out var bytes, out var position, out var pending) != 0)
            throw new InvalidOperationException("Cannot read target block statistics.");
        return (bytes, position, pending != 0);
    }
    public byte[] TargetState(int layer, int kind, int prefix)
    {
        if (layer is < 0 or >= 40 || kind is < 0 or > 1 || prefix < -1)
            throw new ArgumentException("Invalid target state descriptor.");
        var linear = (layer + 1) % 4 != 0;
        var bytes = linear ? kind == 0 ? 32 * 128 * 128 * 4 : 3 * 8192 * 4 : checked(TargetBlockStats().Position * 512 * 4);
        var output = new byte[bytes];
        fixed (byte* o = output)
            if (GraphNative.TargetState(handle, layer, kind, prefix, o, (ulong)bytes) != 0)
                throw new InvalidOperationException("Target state is unavailable or descriptor does not match.");
        return output;
    }
}
internal static unsafe partial class GraphNative
{
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_scalar_verify")]
    internal static partial int ScalarVerify(GraphHandle graph, int* tokens, int count, int* next, out int dispatches, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_prepare_target_block")]
    internal static partial int PrepareTargetBlock(GraphHandle graph, int rows, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_target_block")]
    internal static partial int TargetBlock(GraphHandle graph, int* tokens, int rows, int* next, float* logits, float* trace,
        out int dispatches, out int submissions, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_scalar_target_transaction")]
    internal static partial int ScalarTargetTransaction(GraphHandle graph, int* tokens, int count, int accepted,
        out int dispatches, out int submissions, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_commit_target_block")]
    internal static partial int CommitTargetBlock(GraphHandle graph, int accepted, out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_target_state")]
    internal static partial int TargetState(GraphHandle graph, int layer, int kind, int prefix, byte* output, ulong bytes);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_graph_target_block_stats")]
    internal static partial int TargetBlockStats(GraphHandle graph, out long bytes, out int position, out int pending);
}
