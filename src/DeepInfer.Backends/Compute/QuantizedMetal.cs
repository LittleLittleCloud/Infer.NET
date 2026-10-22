using System.Runtime.InteropServices;
using DeepInfer.Backends.Models;
using Microsoft.Win32.SafeHandles;
namespace DeepInfer.Backends.Compute;

// Direct P/Invoke to our own Metal library. MLX is a checkpoint format/reference only.
public sealed unsafe class QuantizedMetal : IDisposable
{
    internal readonly QuantHandle Context;
    public long Dispatches { get; private set; }
    public long Submissions { get; private set; }
    public double GpuMilliseconds { get; private set; }
    public bool BatchSubmissions { get; }
    public QuantizedMetal(bool batchSubmissions = true)
    {
        BatchSubmissions = batchSubmissions;
        if (!OperatingSystem.IsMacOS()) throw new PlatformNotSupportedException("Quantized Metal requires macOS.");
        var error = new byte[2048];
        fixed (byte* e = error)
        {
            var p = QuantNative.Create(e, error.Length);
            if (p == 0) throw new InvalidOperationException(Error(error));
            Context = new(p, 0);
        }
    }
    public QuantMatrix Allocate(int experts, int rows, int columns, int bits, int group = 64)
    {
        ObjectDisposedException.ThrowIf(Context.IsClosed, this);
        if (experts is < 1 or > 256 || rows is < 1 or > 262144 || columns is < 64 or > 32768 ||
            columns % 64 != 0 || bits is not (5 or 8) || group != 64)
            throw new ArgumentException("Unsupported quantized shape/precision.");
        _ = checked(experts * rows * (columns * bits / 8));
        var error = new byte[2048];
        fixed (byte* e = error)
        {
            var p = QuantNative.MatrixCreate(Context, experts, rows, columns, bits, group, e, error.Length);
            if (p == 0) throw new InvalidOperationException(Error(error));
            return new(this, new QuantHandle(p, 1), experts, rows, columns, bits);
        }
    }
    public DeltaMemory NewDeltaMemory()
    {
        var p = QuantNative.DeltaCreate(Context);
        if (p == 0) throw new InvalidOperationException("Cannot allocate gated-delta state.");
        return new(this, new(p, 2));
    }
    // Only independent jobs may share a submission; outputs become visible on return.
    public void MultiplyBatch(ReadOnlySpan<QuantizedJob> jobs)
    {
        ObjectDisposedException.ThrowIf(Context.IsClosed, this);
        if (jobs.Length is < 1 or > 8) throw new ArgumentException("Expected 1..8 independent GEMV jobs.");
        ReadOnlySpan<int> zero = [0];
        for (var i = 0; i < jobs.Length; i++)
        {
            var job = jobs[i];
            if (job.Matrix is null || job.Input is null || job.Output is null) throw new ArgumentException("Incomplete GEMV job.");
            if (!ReferenceEquals(job.Matrix.Owner, this)) throw new ArgumentException("Batch matrices must belong to this device.");
            job.Matrix.Validate(job.Input.Length, job.Output.Length, job.Experts is null ? zero : job.Experts, job.Batched);
            for (var j = 0; j < i; j++)
                if (ReferenceEquals(job.Matrix, jobs[j].Matrix) || ReferenceEquals(job.Output, jobs[j].Output) ||
                    ReferenceEquals(job.Output, jobs[j].Input) || ReferenceEquals(job.Input, jobs[j].Output))
                    throw new ArgumentException("GEMV jobs must have distinct matrices/outputs and no cross-job dependencies.");
        }
        if (!BatchSubmissions)
        {
            foreach (var job in jobs) job.Matrix.MultiplyExperts(job.Input, job.Output, job.Experts is null ? zero : job.Experts, job.Batched);
            return;
        }
        Span<QuantNative.Job> native = stackalloc QuantNative.Job[jobs.Length];
        Span<GCHandle> pins = stackalloc GCHandle[jobs.Length * 3];
        var retained = 0; var pinned = 0; int defaultExpert = 0;
        try
        {
            for (var i = 0; i < jobs.Length; i++)
            {
                var job = jobs[i]; bool added = false;
                job.Matrix.Handle.DangerousAddRef(ref added);
                if (added) retained++;
                pins[pinned] = GCHandle.Alloc(job.Input, GCHandleType.Pinned);
                var input = (float*)pins[pinned++].AddrOfPinnedObject();
                pins[pinned] = GCHandle.Alloc(job.Output, GCHandleType.Pinned);
                var output = (float*)pins[pinned++].AddrOfPinnedObject();
                int* experts = &defaultExpert;
                if (job.Experts is not null)
                {
                    pins[pinned] = GCHandle.Alloc(job.Experts, GCHandleType.Pinned);
                    experts = (int*)pins[pinned++].AddrOfPinnedObject();
                }
                native[i] = new() { Matrix = job.Matrix.Handle.DangerousGetHandle(), Input = input, Output = output,
                    Experts = experts, Count = job.Experts?.Length ?? 1, Batched = job.Batched ? 1 : 0 };
            }
            Span<byte> error = stackalloc byte[2048]; error.Clear();
            fixed (QuantNative.Job* j = native) fixed (byte* e = error)
            {
                if (QuantNative.Batch(Context, j, native.Length, out var ms, e, error.Length) != 0)
                    throw new InvalidOperationException(System.Text.Encoding.UTF8.GetString(error).TrimEnd('\0'));
                Count(ms, jobs.Length);
            }
        }
        finally
        {
            for (var i = pinned - 1; i >= 0; i--) pins[i].Free();
            for (var i = retained - 1; i >= 0; i--) jobs[i].Matrix.Handle.DangerousRelease();
        }
    }
    // Gate/up + SwiGLU and dependent down stay in native buffers, sharing one submission
    // with the independent shared-expert down projection.
    public void ExpertMlp(QuantMatrix gate, QuantMatrix up, QuantMatrix down, QuantMatrix sharedDown,
        ReadOnlySpan<float> input, ReadOnlySpan<float> sharedInput, ReadOnlySpan<int> experts,
        Span<float> expertOutput, Span<float> sharedOutput)
    {
        ObjectDisposedException.ThrowIf(Context.IsClosed, this);
        ReadOnlySpan<QuantMatrix> matrices = [gate, up, down, sharedDown];
        for (var i = 0; i < matrices.Length; i++)
        {
            if (matrices[i] is null || !ReferenceEquals(matrices[i].Owner, this)) throw new ArgumentException("Expert matrices must belong to this device.");
            for (var j = 0; j < i; j++) if (ReferenceEquals(matrices[i], matrices[j])) throw new ArgumentException("Expert matrices must be distinct.");
        }
        if (gate.Rows != up.Rows || gate.Columns != up.Columns || gate.Experts != up.Experts || gate.Bits != up.Bits ||
            down.Columns != gate.Rows || down.Rows != gate.Columns || down.Experts != gate.Experts ||
            sharedDown.Experts != 1 || sharedDown.Rows != down.Rows || ((ReadOnlySpan<float>)expertOutput).Overlaps(sharedOutput))
            throw new ArgumentException("Incompatible expert shapes/precision or overlapping outputs.");
        gate.Validate(input.Length, gate.Rows * experts.Length, experts, false);
        up.Validate(input.Length, up.Rows * experts.Length, experts, false);
        down.Validate(gate.Rows * experts.Length, expertOutput.Length, experts, true);
        sharedDown.Validate(sharedInput.Length, sharedOutput.Length, [0], false);
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (float* x = input) fixed (float* sx = sharedInput) fixed (int* ids = experts)
        fixed (float* y = expertOutput) fixed (float* sy = sharedOutput) fixed (byte* e = error)
        {
            if (QuantNative.Experts(Context, gate.Handle, up.Handle, down.Handle, sharedDown.Handle,
                x, sx, ids, experts.Length, y, sy, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(System.Text.Encoding.UTF8.GetString(error).TrimEnd('\0'));
            Count(ms, 3);
        }
    }
    internal void Count(double ms, int dispatches = 1, int submissions = 1) { Dispatches += dispatches; Submissions += submissions; GpuMilliseconds += ms; }
    internal static string Error(byte[] bytes) => System.Text.Encoding.UTF8.GetString(bytes).TrimEnd('\0');
    public void Dispose() => Context.Dispose();
}

public readonly record struct QuantizedJob(QuantMatrix Matrix, float[] Input, float[] Output, int[]? Experts = null, bool Batched = false);

public sealed unsafe class QuantMatrix : IDisposable
{
    private readonly QuantizedMetal owner;
    private readonly QuantHandle handle;
    internal QuantizedMetal Owner => owner;
    internal QuantHandle Handle => handle;
    public int Experts { get; }
    public int Rows { get; }
    public int Columns { get; }
    public int Bits { get; }
    internal QuantMatrix(QuantizedMetal owner, QuantHandle handle, int experts, int rows, int columns, int bits)
    { this.owner = owner; this.handle = handle; Experts = experts; Rows = rows; Columns = columns; Bits = bits; }
    private Span<byte> Buffer(int index)
    {
        ObjectDisposedException.ThrowIf(handle.IsClosed || owner.Context.IsClosed, this);
        if (index is < 0 or > 2) throw new ArgumentOutOfRangeException(nameof(index));
        var bytes = checked(Experts * Rows * (index == 0 ? Columns * Bits / 8 : Columns / 64 * 2));
        var p = QuantNative.MatrixBuffer(handle, index);
        if (p == 0) throw new InvalidOperationException("Missing quantized buffer.");
        return new((void*)p, bytes);
    }
    public void Write(int index, ReadOnlySpan<byte> data)
    {
        var buffer = Buffer(index);
        if (buffer.Length != data.Length) throw new ArgumentException("Quantized buffer length mismatch.");
        data.CopyTo(buffer);
    }
    internal void ReadTensor(int index, SafeTensors file, SafeTensors.Tensor tensor) => file.ReadInto(tensor, Buffer(index));
    public void Multiply(ReadOnlySpan<float> input, Span<float> output) => MultiplyExperts(input, output, [0]);
    public void MultiplyExperts(ReadOnlySpan<float> input, Span<float> output, ReadOnlySpan<int> experts, bool batched = false)
    {
        Validate(input.Length, output.Length, experts, batched);
        Span<byte> error = stackalloc byte[2048]; error.Clear();
        fixed (float* x = input) fixed (float* y = output) fixed (int* ids = experts) fixed (byte* e = error)
        {
            if (QuantNative.MatVec(owner.Context, handle, x, y, ids, experts.Length, batched ? 1 : 0, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(System.Text.Encoding.UTF8.GetString(error).TrimEnd('\0'));
            owner.Count(ms);
        }
    }
    internal void Validate(int inputLength, int outputLength, ReadOnlySpan<int> experts, bool batched)
    {
        ObjectDisposedException.ThrowIf(handle.IsClosed || owner.Context.IsClosed, this);
        if (experts.Length is < 1 or > 8 || inputLength != Columns * (batched ? experts.Length : 1) || outputLength != Rows * experts.Length)
            throw new ArgumentException("Quantized GEMV input/output shape mismatch.");
        foreach (var id in experts) if (id < 0 || id >= Experts) throw new ArgumentOutOfRangeException(nameof(experts));
    }
    public void CopyRow(int token, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(handle.IsClosed || owner.Context.IsClosed, this);
        if (output.Length != Columns || token < 0 || token >= Rows || Experts != 1) throw new ArgumentException("Invalid embedding row.");
        fixed (float* y = output)
            if (QuantNative.Embedding(handle, token, y) != 0) throw new InvalidOperationException("Embedding lookup failed.");
    }
    public void Dispose() => handle.Dispose();
}

public sealed unsafe class DeltaMemory : IDisposable
{
    private readonly QuantizedMetal owner;
    private readonly QuantHandle handle;
    internal DeltaMemory(QuantizedMetal owner, QuantHandle handle) { this.owner = owner; this.handle = handle; }
    public void Step(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
        ReadOnlySpan<float> decay, ReadOnlySpan<float> beta, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(handle.IsClosed || owner.Context.IsClosed, this);
        if (q.Length != 2048 || k.Length != 2048 || v.Length != 4096 || decay.Length != 32 || beta.Length != 32 || output.Length != 4096)
            throw new ArgumentException("Unsupported gated-delta dimensions.");
        var error = new byte[2048];
        fixed (float* qp = q) fixed (float* kp = k) fixed (float* vp = v) fixed (float* gp = decay)
        fixed (float* bp = beta) fixed (float* yp = output) fixed (byte* e = error)
        {
            if (QuantNative.DeltaStep(owner.Context, handle, qp, kp, vp, gp, bp, yp, out var ms, e, error.Length) != 0)
                throw new InvalidOperationException(QuantizedMetal.Error(error));
            owner.Count(ms);
        }
    }
    public void Dispose() => handle.Dispose();
}
internal sealed class QuantHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private readonly int kind;
    public QuantHandle() : base(true) { }
    internal QuantHandle(nint p, int kind) : base(true) { SetHandle(p); this.kind = kind; }
    protected override bool ReleaseHandle()
    {
        if (kind == 0) QuantNative.Destroy(handle);
        else if (kind == 1) QuantNative.MatrixDestroy(handle);
        else QuantNative.DeltaDestroy(handle);
        return true;
    }
}
internal static unsafe partial class QuantNative
{
    [StructLayout(LayoutKind.Sequential)]
    internal struct Job
    {
        public nint Matrix;
        public float* Input;
        public float* Output;
        public int* Experts;
        public int Count, Batched;
    }
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_batch")] internal static partial int Batch(QuantHandle c, Job* jobs, int count, out double ms, byte* e, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_experts")] internal static partial int Experts(QuantHandle c,
        QuantHandle gate, QuantHandle up, QuantHandle down, QuantHandle sharedDown,
        float* input, float* sharedInput, int* experts, int count, float* output, float* sharedOutput,
        out double ms, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_create")] internal static partial nint Create(byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_destroy")] internal static partial void Destroy(nint p);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_matrix_create")] internal static partial nint MatrixCreate(QuantHandle c, int experts, int rows, int columns, int bits, int group, byte* e, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_matrix_buffer")] internal static partial nint MatrixBuffer(QuantHandle m, int index);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_matrix_destroy")] internal static partial void MatrixDestroy(nint m);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_matvec")] internal static partial int MatVec(QuantHandle c, QuantHandle m, float* x, float* y, int* experts, int count, int batched, out double ms, byte* e, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_quant_embedding")] internal static partial int Embedding(QuantHandle m, int token, float* y);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_delta_create")] internal static partial nint DeltaCreate(QuantHandle c);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_delta_destroy")] internal static partial void DeltaDestroy(nint p);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_delta_step")] internal static partial int DeltaStep(QuantHandle c, QuantHandle s, float* q, float* k, float* v, float* g, float* beta, float* y, out double ms, byte* e, int capacity);
}
