using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace DeepInfer.Backends.Compute;

public interface IMatrix : IDisposable
{
    int Rows { get; }
    int Columns { get; }
    void Multiply(ReadOnlySpan<float> input, Span<float> output);
}
public interface IMatVecDevice : IDisposable
{
    IMatrix Upload(float[] values, int rows, int columns);
    IMatrix UploadBFloat16(ushort[] values, int rows, int columns);
    long Dispatches { get; }
    double GpuMilliseconds { get; }
}

public sealed class CpuMatVecDevice : IMatVecDevice
{
    public long Dispatches => 0;
    public double GpuMilliseconds => 0;
    public IMatrix Upload(float[] values, int rows, int columns)
    {
        if (rows < 1 || columns < 1 || values.Length != checked(rows * columns)) throw new ArgumentException("Invalid matrix shape.");
        return new CpuMatrix(values, rows, columns);
    }
    public IMatrix UploadBFloat16(ushort[] values, int rows, int columns)
    {
        var floats = new float[values.Length];
        for (var i = 0; i < values.Length; i++) floats[i] = BitConverter.Int32BitsToSingle(values[i] << 16);
        return Upload(floats, rows, columns);
    }
    public void Dispose() { }
    private sealed class CpuMatrix(float[] values, int rows, int columns) : IMatrix
    {
        public int Rows => rows;
        public int Columns => columns;
        public void Multiply(ReadOnlySpan<float> input, Span<float> output)
        {
            if (input.Length != columns || output.Length != rows || values.Length != checked(rows * columns))
                throw new ArgumentException("GEMV shape mismatch.");
            for (var row = 0; row < rows; row++)
            {
                var weights = values.AsSpan(row * columns, columns);
                var sum = Vector<float>.Zero;
                var column = 0;
                for (; column <= columns - Vector<float>.Count; column += Vector<float>.Count)
                    sum += new Vector<float>(weights.Slice(column)) * new Vector<float>(input.Slice(column));
                var value = Vector.Sum(sum);
                for (; column < columns; column++) value += weights[column] * input[column];
                output[row] = value;
            }
        }
        public void Dispose() { }
    }
}

public sealed unsafe class MetalMatVecDevice : IMatVecDevice
{
    private readonly MetalContextHandle context;
    private bool disposed;
    public long Dispatches { get; private set; }
    public double GpuMilliseconds { get; private set; }
    public static bool BridgeAvailable
    {
        get
        {
            if (!OperatingSystem.IsMacOS()) return false;
            if (!NativeLibrary.TryLoad(Path.Combine(AppContext.BaseDirectory, "libdeepinfer_metal.dylib"), out var handle)) return false;
            NativeLibrary.Free(handle); return true;
        }
    }
    public MetalMatVecDevice()
    {
        if (!OperatingSystem.IsMacOS()) throw new PlatformNotSupportedException("Metal requires macOS.");
        var error = new byte[2048];
        fixed (byte* e = error)
        {
            var handle = MetalBridge.Create(e, error.Length);
            if (handle == 0) throw new InvalidOperationException(Error(error));
            context = new MetalContextHandle(handle);
        }
    }
    public IMatrix Upload(float[] values, int rows, int columns)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (rows < 1 || columns < 1 || values.Length != checked(rows * columns)) throw new ArgumentException("Invalid matrix shape.");
        var error = new byte[2048];
        fixed (float* v = values)
        fixed (byte* e = error)
        {
            var handle = MetalBridge.MatrixCreate(context, v, rows, columns, 0, e, error.Length);
            if (handle == 0) throw new InvalidOperationException(Error(error));
            return new MetalMatrix(this, new MetalMatrixHandle(handle), rows, columns);
        }
    }
    public IMatrix UploadBFloat16(ushort[] values, int rows, int columns)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (rows < 1 || columns < 1 || values.Length != checked(rows * columns)) throw new ArgumentException("Invalid BF16 matrix shape.");
        var error = new byte[2048];
        fixed (ushort* v = values)
        fixed (byte* e = error)
        {
            var handle = MetalBridge.MatrixCreate(context, v, rows, columns, 1, e, error.Length);
            if (handle == 0) throw new InvalidOperationException(Error(error));
            return new MetalMatrix(this, new MetalMatrixHandle(handle), rows, columns);
        }
    }
    public void Dispose() { disposed = true; context.Dispose(); }
    private static string Error(byte[] bytes) => System.Text.Encoding.UTF8.GetString(bytes, 0, Array.IndexOf(bytes, (byte)0) is var index && index >= 0 ? index : bytes.Length);
    private sealed class MetalMatrix(MetalMatVecDevice owner, MetalMatrixHandle handle, int rows, int columns) : IMatrix
    {
        public int Rows => rows;
        public int Columns => columns;
        public void Multiply(ReadOnlySpan<float> input, Span<float> output)
        {
            ObjectDisposedException.ThrowIf(owner.disposed || handle.IsClosed, this);
            if (input.Length != columns || output.Length != rows) throw new ArgumentException("GEMV shape mismatch.");
            var error = new byte[2048];
            fixed (float* x = input)
            fixed (float* y = output)
            fixed (byte* e = error)
            {
                if (MetalBridge.MatVec(owner.context, handle, x, y, out var ms, e, error.Length) != 0)
                    throw new InvalidOperationException(Error(error));
                owner.Dispatches++; owner.GpuMilliseconds += ms;
            }
        }
        public void Dispose() => handle.Dispose();
    }
}
internal sealed class MetalContextHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public MetalContextHandle() : base(true) { }
    internal MetalContextHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { MetalBridge.Destroy(handle); return true; }
}
internal sealed class MetalMatrixHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    public MetalMatrixHandle() : base(true) { }
    internal MetalMatrixHandle(nint value) : base(true) => SetHandle(value);
    protected override bool ReleaseHandle() { MetalBridge.MatrixDestroy(handle); return true; }
}
internal static unsafe partial class MetalBridge
{
    [LibraryImport("deepinfer_metal", EntryPoint = "di_metal_create")]
    internal static partial nint Create(byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_metal_destroy")]
    internal static partial void Destroy(nint context);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_metal_matrix_create")]
    internal static partial nint MatrixCreate(MetalContextHandle context, void* values, int rows, int columns, int bf16, byte* error, int capacity);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_metal_matrix_destroy")]
    internal static partial void MatrixDestroy(nint matrix);
    [LibraryImport("deepinfer_metal", EntryPoint = "di_metal_matvec")]
    internal static partial int MatVec(MetalContextHandle context, MetalMatrixHandle matrix, float* input, float* output,
        out double gpuMilliseconds, byte* error, int capacity);
}
