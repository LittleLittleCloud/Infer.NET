using System.Diagnostics;
using System.Net;
using System.Net.Security;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using DeepInfer.Backends.Split;
using DeepInfer.Core;
using DeepInfer.Runtime;
using Microsoft.AspNetCore.Http.Json;
namespace DeepInfer.Host;

// Persistent deployment path. One model per device, fresh request state and TLS session per handoff.
public static class SplitService
{
    private static readonly string Build = typeof(SplitService).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "unknown";
    private static string Token(Options o) => o.Token is { Length: >= 32 } s ? s : throw new ArgumentException("A token-file containing at least 32 characters is required.");
    private static X509Certificate2 Certificate(Options o) => X509CertificateLoader.LoadPkcs12FromFile(o.Get("tls-cert", ""), File.ReadAllText(o.Get("tls-password-file", "")).Trim(), OperatingSystem.IsWindows() ? X509KeyStorageFlags.MachineKeySet : X509KeyStorageFlags.DefaultKeySet);
    private static FileStream Lock(Options o, string role)
    {
        var path = o.Get("lock-file", Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Infer.NET", role + ".lock"));
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
    }
    private static void Profile(Options o)
    {
        if (o.Get("kv", "fp32") != "fp32") throw new ArgumentException("Deployment currently accepts only verified FP32 KV; BF16 remains experimental.");
        foreach (System.Collections.DictionaryEntry item in Environment.GetEnvironmentVariables())
            if (item.Key is string key && key.StartsWith("DEEPINFER_METAL_", StringComparison.Ordinal) || item.Key is string k && k.StartsWith("DEEPINFER_GRAPH_", StringComparison.Ordinal))
                Environment.SetEnvironmentVariable((string)item.Key, null);
        Environment.SetEnvironmentVariable("DEEPINFER_METAL_KV", "fp32");
        Environment.SetEnvironmentVariable("DEEPINFER_METAL_PREFILL_CHUNK", "32");
        Environment.SetEnvironmentVariable("DEEPINFER_METAL_PREFILL_GEMM", "fp32");
        Environment.SetEnvironmentVariable("DEEPINFER_METAL_PREFILL_MATRIX", "0");
        Environment.SetEnvironmentVariable("DEEPINFER_METAL_PREFILL_DELTA", "sequential");
    }
    public static async Task RunAgent(Options o, CancellationToken ct)
    {
        Profile(o); var token = Token(o); using var certificate = Certificate(o); using var instance = Lock(o, "cuda-8bit");
        var peer = o.Get("tls-peer-sha256", ""); var uri = new Uri(o.Get("endpoint", "tcp://127.0.0.1:5210"));
        if (uri.Scheme != "tcp") throw new ArgumentException("Agent endpoint must use tcp:// (mandatory TLS inside).");
        var bundle = o.Get("weights", "artifacts/split/prefix.bin");
        Console.WriteLine("AGENT Loading: verifying pinned 8bit prefix");
        await using (var f = File.OpenRead(bundle))
            if (Convert.ToHexStringLower(await SHA256.HashDataAsync(f, ct)) != SplitInference.PrefixHash) throw new InvalidDataException("Prefix bundle hash mismatch.");
        using var gpu = new CudaPrefixSession(bundle, 2048, 4);
        gpu.Run([1, 2, 3, 4], 0, null); gpu.Reset();
        Console.WriteLine($"AGENT Ready pid={Environment.ProcessId} build={Build} kv=fp32 capacity=2048");
        while (!ct.IsCancellationRequested)
        {
            try
            {
                using var client = new TcpClient { NoDelay = true };
                using var connect = CancellationTokenSource.CreateLinkedTokenSource(ct); connect.CancelAfter(TimeSpan.FromSeconds(20));
                await client.ConnectAsync(uri.Host, uri.Port, connect.Token);
                await using var tls = await SplitTls.Authenticate(client.GetStream(), certificate, peer, false, connect.Token);
                var wire = await SplitWire.ConnectAsync(tls, token, false, Guid.NewGuid(), connect.Token, 4);
                var start = await wire.ReceiveAsync(ct); // Idle connections are cancelled by service shutdown, not the request deadline.
                using var request = CancellationTokenSource.CreateLinkedTokenSource(ct);
                request.CancelAfter(TimeSpan.FromSeconds(o.Number("deadline-seconds", 300, 1, 1800)));
                gpu.Reset();
                await SplitInference.ServePrefix(gpu, wire, request.Token, start);
                var ack = await wire.ReceiveAsync(request.Token);
                if (ack.Kind != SplitWire.End || ack.Position != start.Count) throw new InvalidDataException("Missing state import acknowledgement.");
            }
            catch (Exception ex) when (ex is IOException or SocketException or System.Security.Authentication.AuthenticationException or OperationCanceledException)
            {
                if (ct.IsCancellationRequested) break;
                Console.Error.WriteLine($"AGENT reconnect: {ex.GetType().Name}");
                await Task.Delay(1000, ct);
            }
            // A native model error escapes and restarts the service, never reuses potentially corrupt GPU state.
        }
    }

    private sealed class Peer(TcpClient client, SslStream tls, SplitWire wire) : IDisposable
    {
        public SplitWire Wire => wire;
        public bool Connected => !(client.Client.Poll(0, SelectMode.SelectRead) && client.Available == 0);
        public void Dispose() { tls.Dispose(); client.Dispose(); }
    }
    public static async Task RunMac(Options o, CancellationToken ct)
    {
        Profile(o); var token = Token(o); using var certificate = Certificate(o); using var instance = Lock(o, "metal-8bit");
        var endpoint = o.Get("listen", "http://127.0.0.1:5201");
        if (!EndpointRules.IsLoopback(EndpointRules.Validate(endpoint, false))) throw new ArgumentException("Worker HTTP API must be loopback; use the local coordinator.");
        var dataUri = new Uri(o.Get("data-listen", "tcp://127.0.0.1:5210"));
        if (dataUri.Scheme != "tcp" || !IPAddress.TryParse(dataUri.Host, out var address)) throw new ArgumentException("data-listen requires a literal tcp:// address.");
        if (!IPAddress.IsLoopback(address) && !o.Flag("allow-remote")) throw new ArgumentException("Data listener requires --allow-remote.");
        using var life = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var peers = Channel.CreateBounded<Peer>(new BoundedChannelOptions(1) { SingleReader = false, SingleWriter = true });
        using var listener = new TcpListener(address, dataUri.Port);
        using var queue = new InferenceQueue(o.Number("queue-size", 2, 0, 16));
        MetalSplitSession? gpu = null; var state = "Loading"; var warmed = false; var completed = 0L;
        var builder = WebApplication.CreateSlimBuilder(); builder.WebHost.UseUrls(endpoint);
        builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = 1024 * 1024);
        builder.Services.Configure<JsonOptions>(j => { j.SerializerOptions.TypeInfoResolverChain.Insert(0, RpcJson.Default); j.SerializerOptions.TypeInfoResolverChain.Insert(0, ServiceJson.Default); });
        builder.Logging.ClearProviders(); var app = builder.Build();
        app.Use(async (context, next) =>
        {
            if (context.Request.Path != "/health" && !CryptographicOperations.FixedTimeEquals(SHA256.HashData(Encoding.UTF8.GetBytes(token)), SHA256.HashData(Encoding.UTF8.GetBytes(context.Request.Headers["X-DeepInfer-Token"].ToString()))))
            { context.Response.StatusCode = 401; return; }
            await next(context);
        });
        ServiceStatus Status() => new(warmed && queue.Busy ? "Busy" : state, warmed && peers.Reader.TryPeek(out var peer) && peer.Connected, Environment.ProcessId, Build, SplitWire.Revision, 8, "fp32", 2048, 256, Interlocked.Read(ref completed));
        app.MapGet("/health", () => Results.Json(Status(), ServiceJson.Default.ServiceStatus));
        app.MapGet("/ready", () => Results.Json(Status(), ServiceJson.Default.ServiceStatus, statusCode: Status().Ready ? 200 : 503));
        var registration = new WorkerRegistration(Protocol.Version, o.Get("id", "split-metal-8bit"), endpoint,
            [new("metal-split", "metal", "Metal + CUDA split prefill", "M5/SM89", null, true, true, "affine8, FP32 KV, 2K")],
            [new(Protocol.Qwen36Model, "mlx-qwen36-affine-chat-v1", false)], 1);
        app.MapGet("/v1/capabilities", () => Results.Json(registration, RpcJson.Default.WorkerRegistration));
        async Task Handle(InferenceRequest input, HttpContext context, bool streaming)
        {
            var error = RequestValidation.Error(input);
            if (error != null || input.Model != Protocol.Qwen36Model || input.Backend != "metal" || input.MaxTokens > 256)
            { await Fail(context, 400, "invalid_request", error ?? "Only Qwen3.6 affine8 / metal, maxTokens 1..256 is supported."); return; }
            if (!warmed || (!queue.Busy && !Status().Ready)) { await Fail(context, 503, "not_ready", "Model warming or CUDA peer unavailable."); return; }
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted, life.Token);
            timeout.CancelAfter(TimeSpan.FromSeconds(o.Number("deadline-seconds", 300, 1, 1800)));
            try
            {
                using var slot = await queue.Enter(timeout.Token);
                if (slot is null) { await Fail(context, 429, "queue_full", "Request queue is full."); return; }
                var ids = input.Messages is { } messages ? gpu!.Tokenize(messages, timeout.Token) : gpu!.Tokenize(input.Prompt, timeout.Token);
                if (ids.Length + input.MaxTokens > 2048) { await Fail(context, 400, "context_length", "Prompt plus output must fit 2048 tokens; no silent truncation."); return; }
                gpu.Reset(ids.Length + input.MaxTokens);
                Peer peer;
                do { peer = await peers.Reader.ReadAsync(timeout.Token); if (peer.Connected) break; peer.Dispose(); } while (true);
                using (peer)
                {
                    var timer = Stopwatch.StartNew(); var gpuStart = gpu.GpuMilliseconds; var dispatchStart = gpu.Dispatches; var submissionStart = gpu.Submissions;
                    await peer.Wire.SendAsync(new(SplitWire.Start, 20, 0, 0, ids.Length, MemoryMarshal.AsBytes(ids.AsSpan()).ToArray()), timeout.Token);
                    var next = -1;
                    for (var offset = 0; offset < ids.Length; offset += 32)
                    {
                        var count = Math.Min(32, ids.Length - offset); var frame = await peer.Wire.ReceiveAsync(timeout.Token);
                        if (frame.Kind != SplitWire.Hidden || frame.Position != offset || frame.Count != count) throw new InvalidDataException("Noncontiguous hidden stream.");
                        var hidden = MemoryMarshal.Cast<byte, float>(frame.Data).ToArray();
                        if (hidden.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Nonfinite hidden.");
                        next = gpu.Suffix(ids.AsSpan(offset, count).ToArray(), hidden, offset + count == ids.Length);
                    }
                    for (var layer = 0; layer < 20; layer++) for (var tensor = 0; tensor < 2; tensor++)
                    {
                        var frame = await peer.Wire.ReceiveAsync(timeout.Token);
                        if (frame.Kind != SplitWire.State || frame.Layer != layer || frame.Tensor != tensor || frame.Position != ids.Length) throw new InvalidDataException("Incomplete state handoff.");
                        gpu.State(layer, tensor, ids.Length, frame.Data, true);
                    }
                    var end = await peer.Wire.ReceiveAsync(timeout.Token);
                    if (end.Kind != SplitWire.End || end.Position != ids.Length) throw new InvalidDataException("Missing handoff completion.");
                    await peer.Wire.SendAsync(new(SplitWire.End, 20, 0, ids.Length, 0, []), timeout.Token);
                    var prefillMs = timer.Elapsed.TotalMilliseconds; var first = 0d; var last = 0d; var output = new List<int>();
                    var finish = "length"; var decoder = Encoding.UTF8.GetDecoder(); var windows = new List<DecodeWindow>();
                    if (streaming) { context.Response.ContentType = "text/event-stream"; context.Response.Headers.CacheControl = "no-cache"; }
                    for (var i = 0; i < input.MaxTokens; i++)
                    {
                        timeout.Token.ThrowIfCancellationRequested();
                        if (gpu.IsEnd(next) && !input.BenchmarkIgnoreEos) { finish = "eos"; break; }
                        output.Add(next); last = timer.Elapsed.TotalMilliseconds; if (output.Count == 1) first = last;
                        if (streaming)
                        {
                            var bytes = gpu.Piece(next); var chars = new char[bytes.Length + 4];
                            var length = decoder.GetChars(bytes, chars, false);
                            await Event(context, "token", new ServiceEvent(new string(chars, 0, length), next, null, null, ids.Length), timeout.Token);
                        }
                        if (i + 1 < input.MaxTokens)
                        {
                            var before = timer.Elapsed.TotalMilliseconds; var beforeGpu = gpu.GpuMilliseconds;
                            next = gpu.Forward(next, ids.Length + i);
                            var ms = timer.Elapsed.TotalMilliseconds - before;
                            if (input.Profile) windows.Add(new(i + 2, i + 2, ids.Length + i + 1, ms, ms > 0 ? 1000 / ms : 0));
                            Console.Error.WriteLine($"DECODE request={peer.Wire.Request} token={i+2} wallMs={ms:F3} gpuMs={gpu.GpuMilliseconds-beforeGpu:F3}");
                        }
                    }
                    if (streaming)
                    {
                        var chars = new char[4]; var length = decoder.GetChars([], chars, true);
                        if (length > 0) await Event(context, "token", new ServiceEvent(new string(chars, 0, length), null, null, null), timeout.Token);
                    }
                    var stats = gpu.Stats();
                    var timing = InferenceTimings.Create(prefillMs, first, output.Count > 1 ? last - first : 0, output.Count) with
                    { DecodeWindows = input.Profile ? windows.ToArray() : null, ContextCapacity = stats.Capacity, KvCacheBytes = stats.KvBytes, MetalAllocatedBytes = stats.AllocatedBytes };
                    var result = new InferenceResult(peer.Wire.Request.ToString(), registration.WorkerId, Protocol.Qwen36Model, "metal", gpu.Decode(output.ToArray()), [], false, timer.Elapsed.TotalMilliseconds,
                        output.ToArray(), ids.Length, finish, gpu.Dispatches - dispatchStart, gpu.GpuMilliseconds - gpuStart, timing, gpu.Submissions - submissionStart);
                    if (streaming) await Event(context, "done", new ServiceEvent(null, null, result, null), timeout.Token);
                    else await context.Response.WriteAsJsonAsync(result, RpcJson.Default.InferenceResult, cancellationToken: timeout.Token);
                    Interlocked.Increment(ref completed);
                }
            }
            catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested) { }
            catch (OperationCanceledException) { await Fail(context, 504, "deadline", "Request deadline exceeded."); }
            catch (Exception ex) when (ex is IOException or SocketException)
            { Console.Error.WriteLine("HANDOFF_ERROR " + ex); await Fail(context, 502, "handoff_failed", "State transfer failed; request state discarded."); }
            catch (InvalidOperationException)
            { state = "Faulted"; warmed = false; life.Cancel(); throw; }
        }
        app.MapPost("/v1/infer", (InferenceRequest r, HttpContext c) => Handle(r, c, false));
        app.MapPost("/v1/infer/stream", (InferenceRequest r, HttpContext c) => Handle(r, c, true));
        await app.StartAsync(ct);
        var background = Task.Run(async () =>
        {
            try
            {
                var directory = o.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx");
                await PinnedSplitModel.Verify(directory, life.Token);
                gpu = new MetalSplitSession(directory, 4);
                state = "Warming"; var warm = gpu.Tokenize("用一句话介绍你自己。", life.Token); gpu.Reset(2048);
                var next = gpu.Reference(warm, life.Token, new float[248320], null);
                for (var i = 0; i < 32; i++) { life.Token.ThrowIfCancellationRequested(); next = gpu.Forward(next, warm.Length + i); }
                gpu.Reset(2048); warmed = true; state = "Ready"; listener.Start(1);
                Console.WriteLine($"MAC Ready pid={Environment.ProcessId} build={Build} kv=fp32 capacity=2048");
                while (!life.IsCancellationRequested)
                {
                    var client = await listener.AcceptTcpClientAsync(life.Token); client.NoDelay = true;
                    SslStream? tls = null;
                    try
                    {
                        using var handshake = CancellationTokenSource.CreateLinkedTokenSource(life.Token); handshake.CancelAfter(TimeSpan.FromSeconds(15));
                        tls = await SplitTls.Authenticate(client.GetStream(), certificate, o.Get("tls-peer-sha256", ""), true, handshake.Token);
                        var wire = await SplitWire.ConnectAsync(tls, token, true, Guid.Empty, handshake.Token, 4);
                        var peer = new Peer(client, tls, wire);
                        if (peers.Reader.TryPeek(out var stale) && !stale.Connected && peers.Reader.TryRead(out stale)) stale.Dispose();
                        try { await peers.Writer.WriteAsync(peer, life.Token); } catch { peer.Dispose(); throw; }
                    }
                    catch (Exception ex) when (ex is IOException or System.Security.Authentication.AuthenticationException or OperationCanceledException)
                    { tls?.Dispose(); client.Dispose(); if (life.IsCancellationRequested) break; Console.Error.WriteLine("Rejected data connection: " + ex.GetType().Name); }
                }
            }
            catch (OperationCanceledException) when (life.IsCancellationRequested) { }
            catch { state = "Faulted"; life.Cancel(); throw; }
        }, CancellationToken.None);
        var heartbeat = Task.Run(async () =>
        {
            using var rpc = new ClusterClient(o.Get("coordinator", "http://127.0.0.1:5200"), token, TimeSpan.FromSeconds(3));
            while (!life.IsCancellationRequested)
            {
                try { if (Status().Ready || queue.Busy) await rpc.RegisterAsync(registration, life.Token); }
                catch (Exception ex) when (ex is HttpRequestException or RpcException or OperationCanceledException) { }
                await Task.Delay(2000, life.Token);
            }
        }, CancellationToken.None);
        try { await app.WaitForShutdownAsync(life.Token); }
        finally
        {
            life.Cancel(); listener.Stop(); await app.StopAsync(CancellationToken.None);
            try { await heartbeat; } catch (OperationCanceledException) { }
            try { await background; } finally { while (peers.Reader.TryRead(out var peer)) peer.Dispose(); gpu?.Dispose(); await app.DisposeAsync(); }
        }
    }
    private static async Task Event(HttpContext c, string name, ServiceEvent data, CancellationToken ct)
    {
        await c.Response.WriteAsync("event: " + name + "\ndata: " + JsonSerializer.Serialize(data, ServiceJson.Default.ServiceEvent) + "\n\n", ct);
        await c.Response.Body.FlushAsync(ct);
    }
    private static async Task Fail(HttpContext c, int status, string code, string message)
    {
        if (c.RequestAborted.IsCancellationRequested) return;
        if (c.Response.HasStarted) await Event(c, "error", new(null, null, null, new(code, message)), c.RequestAborted);
        else { c.Response.StatusCode = status; await c.Response.WriteAsJsonAsync(new RpcError(code, message), RpcJson.Default.RpcError, cancellationToken: c.RequestAborted); }
    }
}
public sealed record ServiceStatus(string State, bool Ready, int Pid, string Build, string Revision, int WeightBits, string Kv, int MaxContextTokens, int MaxOutputTokens, long CompletedRequests);
public sealed record ServiceEvent(string? Text, int? TokenId, InferenceResult? Result, RpcError? Error, int? PromptTokens = null);
[JsonSerializable(typeof(ServiceStatus))]
[JsonSerializable(typeof(ServiceEvent))]
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
internal partial class ServiceJson : JsonSerializerContext;
