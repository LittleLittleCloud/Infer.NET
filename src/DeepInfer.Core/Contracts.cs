namespace DeepInfer.Core;

// Protocol v1: additive DTO changes only. Unknown protocol versions are rejected at registration.
public static class Protocol
{
    public const int Version = 1;
    public const string SyntheticModel = "synthetic-v1";
    public const string TinyStoriesModel = "stories15m";
    public const string Qwen3BModel = "qwen2.5-3b-instruct";
    public const string Qwen7BModel = "qwen2.5-7b-instruct";
    public const string Qwen36Model = "qwen3.6-35b-a3b";
    public const int MaxTokens = 32768;
    public const int MaxPromptChars = 32768;
    public const int MaxLongPromptChars = 8 * 1024 * 1024;
    public const int MaxRequestBytes = 64 * 1024 * 1024;
}

public sealed record DeviceInfo(string Id, string Backend, string Name, string Architecture,
    long? MemoryBytes, bool NativeAvailable, bool SupportsInference, string Status);
public sealed record ModelInfo(string Id, string Format, bool IsSimulation);
public sealed record InferenceRequest(string Prompt, int MaxTokens = 16,
    string Model = Protocol.SyntheticModel, string Backend = "simulation",
    bool Profile = false, bool BenchmarkIgnoreEos = false);
public sealed record InferenceResult(string RequestId, string WorkerId, string Model, string Backend,
    string Text, string[] Tokens, bool IsSimulation, double ComputeMs,
    int[]? TokenIds = null, int PromptTokens = 0, string? FinishReason = null,
    long GpuDispatches = 0, double GpuComputeMs = 0, InferenceTimings? Timings = null, long GpuSubmissions = 0);
public sealed record WorkerRegistration(int ProtocolVersion, string WorkerId, string Endpoint,
    DeviceInfo[] Devices, ModelInfo[] Models, int Capacity);
public sealed record WorkerSnapshot(WorkerRegistration Registration, DateTimeOffset LastSeenUtc, bool Healthy);
public sealed record ClusterSnapshot(int ProtocolVersion, WorkerSnapshot[] Workers);
public sealed record HealthInfo(string Role, string Status, int ProtocolVersion);
public sealed record RpcError(string Code, string Message);
public sealed record BenchmarkResult(string Mode, int Requests, int Warmup, int OutputUnits,
    double ElapsedMs, double P50Ms, double P95Ms, double RequestsPerSecond,
    double SyntheticUnitsPerSecond, string[] WorkerIds, bool IsSimulation = true,
    double TokensPerSecond = 0, long GpuDispatches = 0, double GpuComputeMs = 0,
    double MeanPrefillMs = 0, double P50FirstTokenMs = 0, int DecodeTokens = 0,
    double DecodeMs = 0, double DecodeTokensPerSecond = 0, long GpuSubmissions = 0);

public static class RequestValidation
{
    public static string? Error(InferenceRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt)) return "Prompt is required.";
        var moe = request.Model == Protocol.Qwen36Model;
        var chars = moe ? Protocol.MaxLongPromptChars : Protocol.MaxPromptChars;
        var tokens = moe ? Protocol.MaxTokens : 256;
        if (request.Prompt.Length > chars) return $"Prompt exceeds {chars} characters.";
        if (request.MaxTokens < 1 || request.MaxTokens > tokens) return $"MaxTokens must be 1..{tokens}.";
        if (request.BenchmarkIgnoreEos && !moe) return "BenchmarkIgnoreEos is only implemented for Qwen3.6.";
        if (string.IsNullOrWhiteSpace(request.Model) || string.IsNullOrWhiteSpace(request.Backend))
            return "Model and backend are required.";
        return null;
    }
}

// Device ownership and memory management remain local to the worker. Never send device pointers over RPC.
public interface IComputeBackend
{
    string Kind { get; }
    IReadOnlyList<DeviceInfo> DiscoverDevices();
    ValueTask<IInferenceModel> LoadModelAsync(ModelInfo model, CancellationToken cancellationToken);
}

public interface IInferenceModel : IAsyncDisposable
{
    ModelInfo Info { get; }
    ValueTask<ModelOutput> GenerateAsync(InferenceRequest request, CancellationToken cancellationToken);
}
public sealed record ModelOutput(string[] Tokens, bool IsSimulation, string? Text = null,
    int[]? TokenIds = null, int PromptTokens = 0, string? FinishReason = null,
    long GpuDispatches = 0, double GpuComputeMs = 0, InferenceTimings? Timings = null, long GpuSubmissions = 0);
public sealed class InferenceInputException(string message) : Exception(message);

// Worker-side timing. Decode spans the first through last emitted token and counts N-1 intervals.
// EOS detection after the final emitted token, model loading, prompt prefill and network time are excluded.
public sealed record DecodeWindow(int StartOutputToken, int EndOutputToken, int ContextTokens, double Milliseconds, double TokensPerSecond);
public sealed record InferenceTimings(double PrefillMs, double FirstTokenMs, double DecodeMs, int DecodeTokens, double DecodeTokensPerSecond,
    DecodeWindow[]? DecodeWindows = null, int ContextCapacity = 0, long KvCacheBytes = 0, long MetalAllocatedBytes = 0)
{
    public static InferenceTimings Create(double prefillMs, double firstTokenMs, double decodeMs, int outputTokens)
    {
        if (!double.IsFinite(prefillMs) || !double.IsFinite(firstTokenMs) || !double.IsFinite(decodeMs) ||
            prefillMs < 0 || firstTokenMs < 0 || decodeMs < 0 || outputTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(decodeMs));
        var intervals = Math.Max(0, outputTokens - 1);
        return new(prefillMs, firstTokenMs, intervals == 0 ? 0 : decodeMs, intervals,
            intervals == 0 || decodeMs == 0 ? 0 : intervals * 1000d / decodeMs);
    }
}
