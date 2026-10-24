using System.Runtime.InteropServices;
using DeepInfer.Core;
namespace DeepInfer.Backends;

public sealed class MetalBackend(string? weightsPath = null, string? tokenizerPath = null) : IComputeBackend
{
    public string Kind => "metal";
    public IReadOnlyList<DeviceInfo> DiscoverDevices()
    {
        if (!OperatingSystem.IsMacOS()) return [Missing("Metal requires macOS.")];
        try
        {
            var device = MetalNative.MTLCreateSystemDefaultDevice();
            if (device == 0) return [Missing("Metal returned no default device.")];
            try
            {
                var name = MetalNative.SendPointer(device, MetalNative.sel_registerName("name"));
                var utf8 = MetalNative.SendPointer(name, MetalNative.sel_registerName("UTF8String"));
                var budget = MetalNative.SendUInt64(device, MetalNative.sel_registerName("recommendedMaxWorkingSetSize"));
                return [new("metal:0", Kind, Marshal.PtrToStringUTF8(utf8) ?? "Apple GPU", "Apple GPU (runtime detected)",
                    checked((long)budget), true, Compute.MetalMatVecDevice.BridgeAvailable,
                    Compute.MetalMatVecDevice.BridgeAvailable ? "Own Metal: Qwen3.6 full GPU executor, affine5/8 weights, FP32 state. Memory is the recommended working-set budget." : "Metal discovered; native compute bridge is not installed.")];
            }
            finally { MetalNative.SendPointer(device, MetalNative.sel_registerName("release")); }
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        { return [Missing(ex.Message)]; }
    }
    private DeviceInfo Missing(string status) => new("metal:unavailable", Kind, "Metal", "Apple GPU", null, false, false, status);
    public ValueTask<IInferenceModel> LoadModelAsync(ModelInfo model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model.Id != Protocol.Qwen36Model || model.Format != "mlx-qwen36-affine" || model.IsSimulation) throw new NotSupportedException("Unsupported Metal model format.");
        if (string.IsNullOrWhiteSpace(weightsPath) || string.IsNullOrWhiteSpace(tokenizerPath))
            throw new ArgumentException("Real Metal inference requires --weights and --tokenizer paths.");
        return ValueTask.FromResult<IInferenceModel>(new Models.Qwen36Model(model, weightsPath, tokenizerPath));
    }
}

internal static partial class MetalNative
{
    [LibraryImport("/System/Library/Frameworks/Metal.framework/Metal")]
    internal static partial nint MTLCreateSystemDefaultDevice();
    [LibraryImport("/usr/lib/libobjc.A.dylib", StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint sel_registerName(string name);
    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    internal static partial nint SendPointer(nint receiver, nint selector);
    [LibraryImport("/usr/lib/libobjc.A.dylib", EntryPoint = "objc_msgSend")]
    internal static partial ulong SendUInt64(nint receiver, nint selector);
}

// Inspired by dotLLM's separation of backend contracts and CUDA Driver API interop.
// Original implementation; no dotLLM source or kernels are copied or linked.
public sealed class CudaBackend : IComputeBackend
{
    public string Kind => "cuda";
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int Init(uint flags);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int GetCount(out int count);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int GetDevice(out int device, int ordinal);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int GetName(nint buffer, int length, int device);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int GetAttribute(out int value, int attribute, int device);
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate int GetMemory(out nuint bytes, int device);
    public IReadOnlyList<DeviceInfo> DiscoverDevices()
    {
        if (OperatingSystem.IsMacOS()) return [Missing("CUDA is unavailable on macOS; SM89 target requires an NVIDIA host.")];
        var library = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(library, out var handle)) return [Missing($"CUDA driver {library} was not found.")];
        try
        {
            T Export<T>(string name) where T : Delegate => Marshal.GetDelegateForFunctionPointer<T>(NativeLibrary.GetExport(handle, name));
            var result = Export<Init>("cuInit")(0);
            if (result != 0) return [Missing($"cuInit failed ({result}).")];
            Check(Export<GetCount>("cuDeviceGetCount")(out var count));
            var devices = new List<DeviceInfo>();
            for (var ordinal = 0; ordinal < count; ordinal++)
            {
                Check(Export<GetDevice>("cuDeviceGet")(out var device, ordinal));
                var buffer = Marshal.AllocHGlobal(256);
                try
                {
                    Check(Export<GetName>("cuDeviceGetName")(buffer, 256, device));
                    Check(Export<GetAttribute>("cuDeviceGetAttribute")(out var major, 75, device));
                    Check(Export<GetAttribute>("cuDeviceGetAttribute")(out var minor, 76, device));
                    Check(Export<GetMemory>("cuDeviceTotalMem_v2")(out var bytes, device));
                    devices.Add(new($"cuda:{ordinal}", Kind, Marshal.PtrToStringUTF8(buffer) ?? "NVIDIA GPU",
                        $"sm_{major}{minor}", checked((long)bytes), true, false,
                        "Detected; CUDA inference is NOT implemented. First planned kernel target: RTX 4090 SM89."));
                }
                finally { Marshal.FreeHGlobal(buffer); }
            }
            return devices.Count == 0 ? [Missing("CUDA driver found, no devices.")] : devices;
        }
        catch (Exception ex) when (ex is EntryPointNotFoundException or InvalidOperationException)
        { return [Missing(ex.Message)]; }
        finally { NativeLibrary.Free(handle); }
    }
    private static void Check(int code) { if (code != 0) throw new InvalidOperationException($"CUDA driver error {code}."); }
    private DeviceInfo Missing(string status) => new("cuda:unavailable", Kind, "CUDA", "SM89 target", null, false, false, status);
    public ValueTask<IInferenceModel> LoadModelAsync(ModelInfo model, CancellationToken cancellationToken)
        => throw new NotSupportedException("CUDA model loading and kernels are not implemented.");
}

public static class BackendCatalog
{
    public static IComputeBackend Create(string kind, string? weightsPath = null, string? tokenizerPath = null) => kind switch
    {
        "simulation" => new SimulationBackend(), "cuda" => new CudaBackend(), "metal" => new MetalBackend(weightsPath, tokenizerPath),
        _ => throw new ArgumentException($"Unknown backend '{kind}'. Use simulation, metal or cuda.")
    };
    public static DeviceInfo[] DiscoverAll() => new IComputeBackend[]
        { new SimulationBackend(), new MetalBackend(), new CudaBackend() }.SelectMany(x => x.DiscoverDevices()).ToArray();
}
