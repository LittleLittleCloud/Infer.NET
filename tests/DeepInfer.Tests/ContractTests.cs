using System.Text.Json;
using DeepInfer.Backends;
using DeepInfer.Core;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public sealed class ContractTests
{
    private static WorkerRegistration Worker(string id, string backend = "simulation", bool capable = true) =>
        new(1, id, "http://127.0.0.1:5101", [new("device", backend, "test", "test", null, true, capable, "test")],
            [new(Protocol.SyntheticModel, "synthetic", true)], 1);

    [Fact] public void RegistryRoutesAcrossWorkersAndRejectsExpiredLease()
    {
        var clock = new ManualClock(); var registry = new WorkerRegistry(clock);
        registry.Register(Worker("one"), false); registry.Register(Worker("two"), false);
        var request = new InferenceRequest("test");
        Assert.Equal("one", registry.Select(request)?.WorkerId);
        Assert.Equal("two", registry.Select(request)?.WorkerId);
        clock.Advance(TimeSpan.FromSeconds(8));
        Assert.Null(registry.Select(request));
        Assert.All(registry.Snapshot().Workers, w => Assert.False(w.Healthy));
        registry.Register(Worker("one"), false);
        Assert.Equal("one", registry.Select(request)?.WorkerId);
    }
    [Fact] public void UnsupportedGpuAndModelNeverRoute()
    {
        var registry = new WorkerRegistry(); registry.Register(Worker("gpu", "metal", false), false);
        Assert.Null(registry.Select(new("hello", Backend: "metal")));
        registry.Register(Worker("sim"), false);
        Assert.Null(registry.Select(new("hello", Model: "DeepSeek")));
        Assert.Null(registry.Select(new("hello", Backend: "cuda")));
    }
    [Fact] public void RegistrationReplacesSameWorkerAndRejectsVersion()
    {
        var registry = new WorkerRegistry(); registry.Register(Worker("one"), false);
        registry.Register(Worker("one"), false);
        Assert.Single(registry.Snapshot().Workers);
        Assert.Throws<ArgumentException>(() => registry.Register(Worker("two") with { ProtocolVersion = 99 }, false));
        Assert.Throws<ArgumentException>(() => registry.Register(Worker("two") with { Capacity = 0 }, false));
    }
    [Fact] public void NullDeviceAndModelAdvertisementsAreRejected()
    {
        var registry = new WorkerRegistry();
        Assert.Throws<ArgumentException>(() => registry.Register(Worker("bad") with { Devices = [null!] }, false));
        Assert.Throws<ArgumentException>(() => registry.Register(Worker("bad") with { Models = [null!] }, false));
    }
    [Theory]
    [InlineData("http://example.com", false)] [InlineData("file:///tmp/test", true)]
    [InlineData("http://127.0.0.1/path", false)] [InlineData("http://user:pass@localhost", false)]
    [InlineData("http://127.0.0.1?x=1", true)]
    public void InvalidEndpointsAreRejected(string endpoint, bool remote)
        => Assert.Throws<ArgumentException>(() => EndpointRules.Validate(endpoint, remote));
    [Theory] [InlineData("http://127.0.0.1:5101")] [InlineData("http://[::1]:5101")] [InlineData("http://localhost:5101")]
    public void LoopbackEndpointsAreAccepted(string endpoint) => Assert.NotNull(EndpointRules.Validate(endpoint, false));
    [Theory] [InlineData("", 16)] [InlineData("hello", 0)] [InlineData("hello", 257)]
    public void InvalidRequestsAreRejected(string prompt, int tokens) => Assert.NotNull(RequestValidation.Error(new(prompt, tokens)));
    [Fact] public void OversizedPromptIsRejected() => Assert.NotNull(RequestValidation.Error(new(new string('a', 32769))));
    [Fact] public void LongRequestsAreScopedToMoeAndRoundTripProfileOptions()
    {
        var request = new InferenceRequest(new string('a', 40000), 16384, Protocol.Qwen36Model, "metal", true, true);
        Assert.Null(RequestValidation.Error(request));
        Assert.Equal(request, JsonSerializer.Deserialize(JsonSerializer.Serialize(request, RpcJson.Default.InferenceRequest), RpcJson.Default.InferenceRequest));
        Assert.NotNull(RequestValidation.Error(request with { MaxTokens = 32769 }));
        Assert.NotNull(RequestValidation.Error(request with { Model = Protocol.Qwen3BModel }));
        Assert.NotNull(RequestValidation.Error(new("hello", BenchmarkIgnoreEos: true)));
    }
    [Fact] public void ProfileWindowsAndAllocationStatisticsRoundTrip()
    {
        var timings = InferenceTimings.Create(100, 101, 2000, 513) with
        { DecodeWindows = [new(2, 257, 2304, 900, 256000d/900), new(258, 513, 2560, 1100, 256000d/1100)],
          ContextCapacity = 4096, KvCacheBytes = 4096L*40960, MetalAllocatedBytes = 24000000000 };
        var response = new InferenceResult("id", "worker", Protocol.Qwen36Model, "metal", "text", [], false, 2101, Timings: timings);
        var decoded = JsonSerializer.Deserialize(JsonSerializer.Serialize(response, RpcJson.Default.InferenceResult), RpcJson.Default.InferenceResult)!;
        Assert.Equal(timings.DecodeWindows, decoded.Timings!.DecodeWindows);
        Assert.Equal(timings.KvCacheBytes, decoded.Timings.KvCacheBytes);
        Assert.Equal(timings.MetalAllocatedBytes, decoded.Timings.MetalAllocatedBytes);
    }
    [Fact] public async Task SimulationIsDeterministicBoundedAndHonest()
    {
        await using var model = await new SimulationBackend().LoadModelAsync(new(Protocol.SyntheticModel, "synthetic", true), default);
        var a = await model.GenerateAsync(new("hello", 32), default);
        var b = await model.GenerateAsync(new("hello", 32), default);
        Assert.True(a.IsSimulation); Assert.Equal(32, a.Tokens.Length); Assert.Equal(a.Tokens, b.Tokens);
        using var cancelled = new CancellationTokenSource(); cancelled.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(async () => await model.GenerateAsync(new("hello"), cancelled.Token));
        await Assert.ThrowsAsync<NotSupportedException>(async () => await model.GenerateAsync(new("hello", Backend: "metal"), default));
    }
    [Fact] public async Task GpuBackendsFailExplicitly()
    {
        foreach (var backend in new IComputeBackend[] { new MetalBackend(), new CudaBackend() })
            await Assert.ThrowsAsync<NotSupportedException>(async () => await backend.LoadModelAsync(new("model", "gguf", false), default));
    }
    [Fact] public void SourceGeneratedJsonRoundTripsContracts()
    {
        var worker = Worker("one");
        var json = JsonSerializer.Serialize(worker, RpcJson.Default.WorkerRegistration);
        Assert.Contains("protocolVersion", json);
        Assert.Equal(worker.WorkerId, JsonSerializer.Deserialize(json, RpcJson.Default.WorkerRegistration)?.WorkerId);
        var request = new InferenceRequest("你好", 8);
        Assert.Equal(request, JsonSerializer.Deserialize(JsonSerializer.Serialize(request, RpcJson.Default.InferenceRequest), RpcJson.Default.InferenceRequest));
    }
    [Fact] public void SubmissionMetricsRoundTripAndOldResponsesDefaultToZero()
    {
        var response = new InferenceResult("request", "worker", Protocol.Qwen36Model, "metal", "4", ["4"], false, 1,
            GpuDispatches: 541, GpuSubmissions: 231);
        var json = JsonSerializer.Serialize(response, RpcJson.Default.InferenceResult);
        var decoded = JsonSerializer.Deserialize(json, RpcJson.Default.InferenceResult)!;
        Assert.Equal(541, decoded.GpuDispatches);
        Assert.Equal(231, decoded.GpuSubmissions);
        using var parsed = JsonDocument.Parse(json);
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            foreach (var property in parsed.RootElement.EnumerateObject())
                if (property.Name != "gpuSubmissions") property.WriteTo(writer);
            writer.WriteEndObject();
        }
        Assert.Equal(0, JsonSerializer.Deserialize(stream.ToArray(), RpcJson.Default.InferenceResult)!.GpuSubmissions);
    }
    private sealed class ManualClock : TimeProvider
    {
        private DateTimeOffset now = DateTimeOffset.UnixEpoch;
        public override DateTimeOffset GetUtcNow() => now;
        public void Advance(TimeSpan span) => now += span;
    }
}
