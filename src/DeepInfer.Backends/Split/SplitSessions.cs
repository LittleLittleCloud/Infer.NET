using System.Runtime.InteropServices;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
namespace DeepInfer.Backends.Split;

// Dedicated, single-request sessions. A failed request is disposed, never reused.
public sealed class MetalSplitSession : IDisposable
{
    private readonly QuantizedMetal device;
    private readonly Qwen36Weights weights;
    private readonly Qwen36Gpu graph;
    private readonly QwenTokenizer tokenizer;
    public MetalSplitSession(string directory, int kvBytes = 2, bool? residency = null, int residencyRefreshMilliseconds = 1000)
    {
        if (kvBytes is not (2 or 4) || Environment.GetEnvironmentVariable("DEEPINFER_METAL_KV") != (kvBytes == 2 ? "bf16" : "fp32"))
            throw new InvalidOperationException("Split session KV precision must match the selected protocol profile.");
        device = new(residency: residency, residencyRefreshMilliseconds: residencyRefreshMilliseconds);
        try
        {
            weights = new(directory, device);
            try { if (weights.WeightBits != 8) throw new NotSupportedException("Split execution requires affine8 weights."); graph = new(device, weights); tokenizer = new(Path.Combine(directory, "tokenizer.json"), true); device.PrepareResidency(); }
            catch { graph?.Dispose(); weights.Dispose(); throw; }
        }
        catch { device.Dispose(); throw; }
    }
    public int[] Tokenize(string prompt, CancellationToken ct) => tokenizer.EncodePrompt(prompt, ct);
    public int[] Tokenize(DeepInfer.Core.ChatTurn[] messages, CancellationToken ct) => tokenizer.EncodeMessages(messages, ct);
    public string Decode(int[] tokens) => tokenizer.Decode(tokens);
    public bool IsEnd(int token) => tokenizer.IsEnd(token);
    public byte[] Piece(int token) => tokenizer.DecodePiece(0, token);
    public double GpuMilliseconds => device.GpuMilliseconds;
    public long Dispatches => device.Dispatches;
    public long Submissions => device.Submissions;
    public (int Capacity, long KvBytes, long AllocatedBytes) Stats() => graph.Stats();
    public void Reset(int capacity) { graph.Reset(); graph.Reserve(capacity); }
    public void EnsureCapacity(int tokens) { if (tokens > graph.Stats().Capacity) graph.Grow(tokens); }
    public int Suffix(int[] tokens, float[] hidden, bool final, float[]? logits = null) => graph.Suffix(tokens, hidden, final, logits);
    public int Forward(int token, int position, float[]? logits = null) => graph.Forward(token, position, true, logits ?? [], null, logits is not null);
    public void State(int layer, int tensor, int position, byte[] bytes, bool import) => graph.PrefixState(layer, tensor, position, bytes, import);
    public int Prefill(int[] tokens, CancellationToken ct) => graph.Prefill(tokens, ct);
    public int Reference(int[] tokens, CancellationToken ct, float[] logits, Action<int,int,float[]>? trace)
    {
        // Match the first CUDA correctness path: <=32-token chunks, FP32 affine GEMV.
        var next = -1;
        for (var offset = 0; offset < tokens.Length; offset += 32)
        {
            var count = Math.Min(32, tokens.Length - offset);
            next = graph.Prefill(tokens.AsSpan(offset, count), ct, offset + count == tokens.Length ? logits : null,
                trace is null ? null : (p,l,x) => trace(offset+p,l,x));
        }
        return next;
    }
    public void Dispose() { graph.Dispose(); weights.Dispose(); device.Dispose(); }
}
public sealed unsafe class CudaPrefixSession : IDisposable
{
    private nint handle;
    private readonly int kvBytes;
    public CudaPrefixSession(string bundle, int capacity, int kvBytes = 2, bool batchReadback = false, bool layerBatch = true)
    {
        if(kvBytes is not (2 or 4)) throw new ArgumentException("KV element width must be 2 or 4.");
        if(CudaPrefixNative.AbiVersion()!=5) throw new InvalidOperationException("CUDA prefix ABI mismatch; rebuild native and managed components.");
        this.kvBytes = kvBytes;
        var error = new byte[2048];
        fixed (byte* e = error) handle = CudaPrefixNative.Create(bundle, capacity, kvBytes, e, error.Length);
        if (handle == 0) throw new InvalidOperationException(Message(error));
        if (CudaPrefixNative.ExecutionMode(handle, layerBatch ? 1 : 0) != 0) { Dispose(); throw new InvalidOperationException("CUDA prefill configuration rejected."); }
        if (CudaPrefixNative.ReadbackMode(handle, batchReadback ? 1 : 0) != 0) { Dispose(); throw new InvalidOperationException("CUDA readback configuration rejected."); }
    }
    private static string Message(byte[] e) => System.Text.Encoding.UTF8.GetString(e).TrimEnd('\0');
    public float[] Run(int[] tokens, int position, float[]? trace)
    {
        ObjectDisposedException.ThrowIf(handle == 0, this);
        if (tokens.Length is < 1 or > 32 || (trace is not null && trace.Length != tokens.Length*20*2048)) throw new ArgumentException("Invalid CUDA prefix geometry.");
        var output = new float[tokens.Length*2048]; var error = new byte[2048];
        fixed (int* t = tokens) fixed (float* y = output) fixed (float* r = trace) fixed (byte* e = error)
            if (CudaPrefixNative.Run(handle, t, tokens.Length, position, y, r, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
        return output;
    }
    public void Reserve(int capacity)
    {
        ObjectDisposedException.ThrowIf(handle == 0, this);
        var error = new byte[2048];
        fixed (byte* e = error)
            if (CudaPrefixNative.Reserve(handle, capacity, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
    }
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(handle == 0, this);
        var error = new byte[2048];
        fixed (byte* e = error)
            if (CudaPrefixNative.Reset(handle, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
    }
    public byte[] State(int layer, int tensor, int position)
    {
        ObjectDisposedException.ThrowIf(handle == 0, this);
        var data = new byte[StateBytes(layer,tensor,position,kvBytes)]; var error = new byte[2048];
        fixed (byte* d = data) fixed (byte* e = error)
            if (CudaPrefixNative.State(handle, layer, tensor, position, d, data.Length, e, error.Length) != 0) throw new InvalidOperationException(Message(error));
        return data;
    }
    public static int StateBytes(int layer, int tensor, int position, int kvBytes = 2)
    {
        if (kvBytes is not (2 or 4)) throw new ArgumentException("Invalid KV width.");
        if (layer is < 0 or >= 20 || tensor is < 0 or > 1 || position is < 1 or > DeepInfer.Core.Protocol.Qwen36Context) throw new ArgumentException("Invalid state descriptor.");
        return (layer+1)%4 == 0 ? position*512*kvBytes : tensor == 0 ? 32*128*128*4 : 3*8192*4;
    }
    public void Dispose() { if(handle != 0) CudaPrefixNative.Destroy(handle); handle = 0; }
}
internal static unsafe partial class CudaPrefixNative
{
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_execution_mode")]
    internal static partial int ExecutionMode(nint graph, int mode);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_readback_mode")]
    internal static partial int ReadbackMode(nint graph, int batched);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_abi_version")]
    internal static partial int AbiVersion();
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_create", StringMarshalling=StringMarshalling.Utf8)]
    internal static partial nint Create(string path,int capacity,int kvBytes,byte* error,int errorCapacity);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_destroy")]
    internal static partial void Destroy(nint graph);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_reserve")]
    internal static partial int Reserve(nint graph, int tokens, byte* error, int capacity);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_reset")]
    internal static partial int Reset(nint graph, byte* error, int capacity);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_run")]
    internal static partial int Run(nint graph,int* tokens,int count,int position,float* hidden,float* trace,byte* error,int capacity);
    [LibraryImport("deepinfer_cuda", EntryPoint="di_cuda_prefix_state")]
    internal static partial int State(nint graph,int layer,int tensor,int position,byte* data,int bytes,byte* error,int capacity);
}
