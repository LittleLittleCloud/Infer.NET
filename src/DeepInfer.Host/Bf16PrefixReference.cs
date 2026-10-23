using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using DeepInfer.Backends.Split;
using DeepInfer.Core;

namespace DeepInfer.Host;

// Metal control for CUDA's fixed synthetic prefix probe. This is a numerical
// diagnostic; it does not route inference over SSH or replace the full-model reference.
internal static class Bf16PrefixReference
{
    public static async Task Run(Options options, CancellationToken ct, bool fp32)
    {
        var kv = fp32 ? "fp32" : "bf16";
        var gemm = fp32 ? "fp32" : "bf16-nax";
        if (Environment.GetEnvironmentVariable("DEEPINFER_METAL_KV") != kv ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_GEMM") != gemm ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_CHUNK") != "128" ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_MATRIX") != (fp32 ? "0" : "1") ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_DELTA") != "sequential")
            throw new ArgumentException($"Set {kv} KV, {gemm} GEMM, 128-token chunk, matrix={(fp32 ? 0 : 1)} and sequential Delta explicitly.");
        var output = Path.GetFullPath(options.Get("output", fp32 ? "artifacts/fp32-metal-prefix" : "artifacts/bf16-metal-prefix"));
        if (Directory.Exists(output)) throw new ArgumentException("Use a fresh evidence directory.");
        Directory.CreateDirectory(output);
        var tokens = Enumerable.Range(0, 128).Select(i => 100 + (i * 137) % 247000).ToArray();
        using var gpu = new MetalSplitSession(options.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx"),
            fp32 ? 4 : 2, residency: true, prefillScratchTokens: 128);
        var profile = gpu.ExecutionProfile();
        if (profile.KvElementBytes != (fp32 ? 4 : 2) || profile.Bf16Gemm == fp32 ||
            profile.Matrix == fp32 || profile.ScratchTokens < 128)
            throw new InvalidOperationException($"Native Metal profile does not match {kv} control requirements.");
        var trace = new float[tokens.Length * 20 * 2048];
        var logits = new float[248320];
        gpu.Reset(tokens.Length + 8);
        var next = gpu.PrefillTrace(tokens, ct, logits, (position, layer, values) =>
        {
            if (layer < 20) values.CopyTo(trace, (position * 20 + layer) * 2048);
        });
        var bytes = MemoryMarshal.AsBytes(trace.AsSpan()).ToArray();
        await File.WriteAllBytesAsync(Path.Combine(output, "trace.f32"), bytes, ct);
        for (var layer = 0; layer < 20; layer++) for (var kind = 0; kind < 2; kind++)
        {
            var state = new byte[CudaPrefixSession.StateBytes(layer, kind, tokens.Length, fp32 ? 4 : 2)];
            gpu.State(layer, kind, tokens.Length, state, false);
            await File.WriteAllBytesAsync(Path.Combine(output, $"state-{layer}-{kind}.bin"), state, ct);
        }
        await File.WriteAllTextAsync(Path.Combine(output, "tokens.json"),
            JsonSerializer.Serialize(tokens, RpcJson.Default.Int32Array), ct);
        using var file = File.Create(Path.Combine(output, "result.json"));
        using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteString("kind", fp32 ? "metal-fp32-prefix-control-not-performance" : "independent-metal-bf16-prefix-reference-not-performance");
        json.WriteNumber("tokens", tokens.Length);
        json.WriteNumber("weightBits", 8);
        json.WriteString("kv", kv);
        json.WriteString("gemm", gemm);
        json.WriteString("routerProjection", fp32 ? "fp32" : Environment.GetEnvironmentVariable("DEEPINFER_METAL_ROUTER_FP32") == "1" ? "fp32-diagnostic" : "bf16");
        json.WriteNumber("chunkTokens", 128);
        json.WriteNumber("nextToken", next);
        json.WriteString("traceSha256", Convert.ToHexStringLower(SHA256.HashData(bytes)));
        json.WriteNumber("metalAllocatedBytes", gpu.Stats().AllocatedBytes);
        json.WriteEndObject();
        await json.FlushAsync(ct);
        Console.WriteLine($"Metal {kv} prefix control: {tokens.Length} tokens, next {next}, trace {output}");
    }
}
