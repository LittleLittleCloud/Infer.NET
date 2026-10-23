using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using DeepInfer.Backends;
using DeepInfer.Core;
using DeepInfer.Runtime;
using Microsoft.AspNetCore.Http.Json;

namespace DeepInfer.Host;

public static class Server
{
    public static async Task RunAsync(Options options, bool worker, CancellationToken ct)
    {
        var endpoint = options.Get("listen", worker ? "http://127.0.0.1:5101" : "http://127.0.0.1:5100");
        var allowRemote = options.Flag("allow-remote");
        var uri = EndpointRules.Validate(endpoint, allowRemote);
        var token = options.Token;
        if ((allowRemote || !EndpointRules.IsLoopback(uri)) && string.IsNullOrWhiteSpace(token))
            throw new ArgumentException("Remote mode requires DEEPINFER_TOKEN or --token.");
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls(endpoint);
        builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = Protocol.MaxRequestBytes);
        builder.Logging.ClearProviders();
        builder.Logging.AddSimpleConsole(c => c.SingleLine = true);
        builder.Logging.SetMinimumLevel(LogLevel.Warning);
        builder.Services.Configure<JsonOptions>(o => o.SerializerOptions.TypeInfoResolverChain.Insert(0, RpcJson.Default));
        var app = builder.Build();
        app.Use(async (context, next) =>
        {
            var supplied = context.Request.Headers["X-DeepInfer-Token"].ToString();
            var compatible = !worker && CompatibleChat.IsPath(context.Request.Path);
            if (compatible)
            {
                var bearer = context.Request.Headers.Authorization.ToString();
                var apiKey = context.Request.Headers["x-api-key"].ToString();
                if (supplied.Length == 0) supplied = bearer.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? bearer[7..] : apiKey;
            }
            if (context.Request.Path != "/health" && !string.IsNullOrEmpty(token) &&
                !CryptographicOperations.FixedTimeEquals(
                    SHA256.HashData(Encoding.UTF8.GetBytes(supplied)),
                    SHA256.HashData(Encoding.UTF8.GetBytes(token))))
            {
                if (compatible) { await CompatibleChat.Error(context, context.Request.Path == "/v1/messages" ? ChatFormat.Messages : ChatFormat.Completions, 401, "A valid API key is required."); return; }
                context.Response.StatusCode = 401;
                await context.Response.WriteAsJsonAsync(new RpcError("unauthorized", "A valid cluster token is required."), RpcJson.Default.RpcError);
                return;
            }
            await next(context);
        });
        app.MapGet("/health", () => new HealthInfo(worker ? "worker" : "coordinator", "ok", Protocol.Version));
        IInferenceModel? model = null;
        Task? heartbeat = null;
        using var lifetime = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (worker)
        {
            var kind = options.Get("backend", "simulation");
            var weightsPath = options.Get("weights", ""); var tokenizerPath = options.Get("tokenizer", "");
            var qwen = Directory.Exists(weightsPath);
            var defaultModel = Protocol.TinyStoriesModel;
            var format = qwen ? "hf-qwen2" : "llama2c-f32";
            if (qwen)
            {
                using var config = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(weightsPath, "config.json")));
                if (config.RootElement.GetProperty("model_type").GetString() == "qwen3_5_moe")
                { defaultModel = Protocol.Qwen36Model; format = "mlx-qwen36-affine"; }
                else defaultModel = config.RootElement.GetProperty("hidden_size").GetInt32() == 3584 && config.RootElement.GetProperty("num_hidden_layers").GetInt32() == 28
                    ? Protocol.Qwen7BModel : Protocol.Qwen3BModel;
            }
            if (qwen && tokenizerPath.Length == 0) tokenizerPath = Path.Combine(weightsPath, "tokenizer.json");
            var backend = BackendCatalog.Create(kind, weightsPath, tokenizerPath);
            var devices = backend.DiscoverDevices().ToArray();
            if (kind == "simulation")
                model = await backend.LoadModelAsync(new(Protocol.SyntheticModel, "synthetic", true), ct);
            else if (weightsPath.Length != 0 || tokenizerPath.Length != 0)
                model = await backend.LoadModelAsync(new(options.Get("model", defaultModel), format, false), ct);
            var id = options.Get("id", $"{Environment.MachineName}-{Environment.ProcessId}");
            var capacity = options.Number("capacity", 1, 1, 64);
            if (model is not null && !model.Info.IsSimulation && capacity != 1)
                throw new ArgumentException("Real-model workers require --capacity 1; use independent workers for parallel requests.");
            using var slots = new SemaphoreSlim(capacity, capacity);
            var advertised = options.Get("advertise", endpoint);
            EndpointRules.Validate(advertised, allowRemote);
            var registration = new WorkerRegistration(Protocol.Version, id, advertised, devices,
                model is null ? [] : [model.Info], capacity);
            app.MapGet("/v1/capabilities", () => registration);
            app.MapPost("/v1/infer", async (InferenceRequest request, HttpContext context) =>
            {
                if (request.Messages != null) return Failure(501, "unsupported_chat", "Use a chat-v1 split worker for structured messages.");
                var error = RequestValidation.Error(request);
                if (error != null) return Failure(400, "invalid_request", error);
                if (model is null || request.Model != model.Info.Id || request.Backend != backend.Kind)
                    return Failure(501, "backend_not_implemented", "This worker has no implemented backend/model for this request.");
                if (!await slots.WaitAsync(0, context.RequestAborted)) return Failure(429, "worker_busy", "Worker capacity is exhausted.");
                try
                {
                    var timer = Stopwatch.StartNew();
                    var output = await model.GenerateAsync(request, context.RequestAborted);
                    return Results.Json(new InferenceResult(Guid.NewGuid().ToString("N"), id, model.Info.Id,
                        backend.Kind, output.Text ?? "[SIMULATION] " + string.Join(' ', output.Tokens), output.Tokens,
                        output.IsSimulation, timer.Elapsed.TotalMilliseconds, output.TokenIds, output.PromptTokens,
                        output.FinishReason, output.GpuDispatches, output.GpuComputeMs, output.Timings, output.GpuSubmissions), RpcJson.Default.InferenceResult);
                }
                catch (InferenceInputException ex) { return Failure(400, "invalid_model_input", ex.Message); }
                finally { slots.Release(); }
            });
            await app.StartAsync(ct);
            Console.WriteLine($"Worker {id} listening at {endpoint}; backend={backend.Kind}; inference={model is not null}");
            var coordinator = options.Get("coordinator", "http://127.0.0.1:5100");
            EndpointRules.Validate(coordinator, allowRemote);
            heartbeat = HeartbeatAsync(coordinator, token, registration, lifetime.Token);
            try { await app.WaitForShutdownAsync(ct); }
            finally
            {
                lifetime.Cancel();
                try { await heartbeat; } catch (OperationCanceledException) { }
                if (model is not null) await model.DisposeAsync();
                await app.DisposeAsync();
            }
        }
        else
        {
            var registry = new WorkerRegistry();
            CompatibleChat.Map(app, registry, token);
            app.MapGet("/v1/workers", () => registry.Snapshot());
            app.MapPost("/v1/infer/stream", (InferenceRequest request, HttpContext context) => ServiceStreaming.Proxy(request, context, registry, token));
            app.MapPost("/v1/workers/register", (WorkerRegistration registration) =>
            {
                try { registry.Register(registration, allowRemote); return Results.NoContent(); }
                catch (ArgumentException ex) { return Failure(400, "invalid_registration", ex.Message); }
            });
            app.MapPost("/v1/infer", async (InferenceRequest request, HttpContext context) =>
            {
                var error = RequestValidation.Error(request);
                if (error != null) return Failure(400, "invalid_request", error);
                var target = registry.Select(request);
                if (target is null) return Failure(503, "no_eligible_worker", "No live worker implements the requested model/backend. Check model/backend advertisements and loaded weights.");
                using var rpc = new ClusterClient(target.Endpoint, token, TimeSpan.FromHours(24));
                try { return Results.Json(await rpc.InferAsync(request, context.RequestAborted), RpcJson.Default.InferenceResult); }
                catch (RpcException ex) { return Failure(ex.StatusCode is 400 or 429 or 501 ? ex.StatusCode : 502, "worker_error", ex.Message); }
                catch (HttpRequestException) { return Failure(502, "worker_unreachable", $"Worker '{target.WorkerId}' is unreachable."); }
                catch (OperationCanceledException) when (!context.RequestAborted.IsCancellationRequested)
                { return Failure(504, "worker_timeout", $"Worker '{target.WorkerId}' exceeded the RPC deadline."); }
            });
            Console.WriteLine($"Coordinator listening at {endpoint}; protocol=v{Protocol.Version}; lease=8s");
            try { await app.RunAsync(ct); }
            finally { await app.DisposeAsync(); }
        }
    }
    private static IResult Failure(int status, string code, string message)
        => Results.Json(new RpcError(code, message), RpcJson.Default.RpcError, statusCode: status);

    private static async Task HeartbeatAsync(string endpoint, string? token, WorkerRegistration registration, CancellationToken ct)
    {
        using var rpc = new ClusterClient(endpoint, token, TimeSpan.FromSeconds(3));
        while (!ct.IsCancellationRequested)
        {
            try { await rpc.RegisterAsync(registration, ct); }
            catch (Exception ex) when (ex is HttpRequestException or RpcException or OperationCanceledException)
            {
                if (ct.IsCancellationRequested) break;
                Console.Error.WriteLine($"Registration pending: {ex.Message}");
            }
            await Task.Delay(TimeSpan.FromSeconds(2), ct);
        }
    }
}
