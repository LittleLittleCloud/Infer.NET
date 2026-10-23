using System.Net.Http.Json;
using System.Text.Json;
using DeepInfer.Core;
using DeepInfer.Runtime;
namespace DeepInfer.Host;

internal static class ServiceStreaming
{
    public static async Task Proxy(InferenceRequest request, HttpContext context, WorkerRegistry registry, string? token)
    {
        var error = RequestValidation.Error(request);
        if (error != null) { context.Response.StatusCode = 400; await context.Response.WriteAsJsonAsync(new RpcError("invalid_request", error), RpcJson.Default.RpcError); return; }
        var worker = registry.Select(request);
        if (worker == null) { context.Response.StatusCode = 503; await context.Response.WriteAsJsonAsync(new RpcError("no_eligible_worker", "No ready worker."), RpcJson.Default.RpcError); return; }
        using var client = new HttpClient { Timeout = TimeSpan.FromHours(24) };
        using var message = new HttpRequestMessage(HttpMethod.Post, worker.Endpoint.TrimEnd('/') + "/v1/infer/stream")
        { Content = JsonContent.Create(request, RpcJson.Default.InferenceRequest) };
        if (token != null) message.Headers.Add("X-DeepInfer-Token", token);
        try
        {
            using var response = await client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, context.RequestAborted);
            context.Response.StatusCode = (int)response.StatusCode;
            context.Response.ContentType = response.Content.Headers.ContentType?.ToString() ?? "application/json";
            context.Response.Headers.CacheControl = "no-cache";
            await using var stream = await response.Content.ReadAsStreamAsync(context.RequestAborted);
            var buffer = new byte[8192]; int count;
            while ((count = await stream.ReadAsync(buffer, context.RequestAborted)) != 0)
            { await context.Response.Body.WriteAsync(buffer.AsMemory(0, count), context.RequestAborted); await context.Response.Body.FlushAsync(context.RequestAborted); }
        }
        catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested) { }
        catch (Exception ex) when (ex is HttpRequestException or IOException)
        {
            if (context.Response.HasStarted) context.Abort();
            else { context.Response.StatusCode = 502; await context.Response.WriteAsJsonAsync(new RpcError("worker_unreachable", "Worker stream failed; partial requests are never retried."), RpcJson.Default.RpcError); }
        }
    }
    public static async Task Client(Options o, CancellationToken ct)
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        using var request = new HttpRequestMessage(HttpMethod.Post, o.Get("endpoint", "http://127.0.0.1:5200").TrimEnd('/') + "/v1/infer/stream")
        { Content = JsonContent.Create(new InferenceRequest(o.Get("prompt", "你好"), o.Number("max-tokens", 128, 1, 256), o.Get("model", Protocol.Qwen36Model), "metal"), RpcJson.Default.InferenceRequest) };
        if (o.Token != null) request.Headers.Add("X-DeepInfer-Token", o.Token);
        using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct); response.EnsureSuccessStatusCode();
        using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(ct)); var done = false;
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
            var item = JsonSerializer.Deserialize(line[6..], ServiceJson.Default.ServiceEvent) ?? throw new InvalidDataException("Empty stream event.");
            if (item.Error != null) throw new IOException(item.Error.Code + ": " + item.Error.Message);
            if (item.Text != null) Console.Write(item.Text);
            if (item.Result != null) { done = true; Console.WriteLine(); Console.Error.WriteLine($"finish={item.Result.FinishReason} decode={item.Result.Timings?.DecodeTokensPerSecond:F2} tokens/s"); }
        }
        if (!done) throw new IOException("Response stream ended without completion; output may be partial.");
    }
}
