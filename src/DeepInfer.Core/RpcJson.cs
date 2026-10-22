using System.Text.Json.Serialization;
namespace DeepInfer.Core;

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(int[]))]
[JsonSerializable(typeof(InferenceRequest))]
[JsonSerializable(typeof(InferenceResult))]
[JsonSerializable(typeof(WorkerRegistration))]
[JsonSerializable(typeof(ClusterSnapshot))]
[JsonSerializable(typeof(DeviceInfo[]))]
[JsonSerializable(typeof(HealthInfo))]
[JsonSerializable(typeof(RpcError))]
[JsonSerializable(typeof(BenchmarkResult))]
public partial class RpcJson : JsonSerializerContext;
