using System.Net.Http.Json;
using System.Text.Json.Serialization.Metadata;
using DeepInfer.Core;
namespace DeepInfer.Runtime;

public sealed class ClusterClient : IDisposable
{
    private readonly HttpClient http;
    public ClusterClient(string endpoint, string? token = null, TimeSpan? timeout = null)
    {
        http = new(new SocketsHttpHandler { AllowAutoRedirect = false })
        { BaseAddress = EndpointRules.Validate(endpoint, true), Timeout = timeout ?? TimeSpan.FromHours(24) };
        if (!string.IsNullOrEmpty(token)) http.DefaultRequestHeaders.Add("X-DeepInfer-Token", token);
    }
    public Task<ClusterSnapshot> StatusAsync(CancellationToken ct = default) => Get("/v1/workers", RpcJson.Default.ClusterSnapshot, ct);
    public Task<HealthInfo> HealthAsync(CancellationToken ct = default) => Get("/health", RpcJson.Default.HealthInfo, ct);
    public Task<InferenceResult> InferAsync(InferenceRequest request, CancellationToken ct = default)
        => Post("/v1/infer", request, RpcJson.Default.InferenceRequest, RpcJson.Default.InferenceResult, ct);
    public async Task RegisterAsync(WorkerRegistration request, CancellationToken ct)
    {
        using var response = await http.PostAsJsonAsync("/v1/workers/register", request, RpcJson.Default.WorkerRegistration, ct);
        await EnsureSuccess(response, ct);
    }
    private async Task<T> Get<T>(string path, JsonTypeInfo<T> type, CancellationToken ct)
    {
        using var response = await http.GetAsync(path, ct);
        await EnsureSuccess(response, ct);
        return await response.Content.ReadFromJsonAsync(type, ct) ?? throw new HttpRequestException("Empty RPC response.");
    }
    private async Task<TOut> Post<TIn,TOut>(string path, TIn value, JsonTypeInfo<TIn> input, JsonTypeInfo<TOut> output, CancellationToken ct)
    {
        using var response = await http.PostAsJsonAsync(path, value, input, ct);
        await EnsureSuccess(response, ct);
        return await response.Content.ReadFromJsonAsync(output, ct) ?? throw new HttpRequestException("Empty RPC response.");
    }
    private static async Task EnsureSuccess(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode) return;
        var body = await response.Content.ReadAsStringAsync(ct);
        throw new RpcException((int)response.StatusCode, body);
    }
    public void Dispose() => http.Dispose();
}
public sealed class RpcException(int statusCode, string message) : Exception(message)
{
    public int StatusCode { get; } = statusCode;
}
