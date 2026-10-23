using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using DeepInfer.Backends.Split;
using DeepInfer.Core;

namespace DeepInfer.Host;

// Independent Metal reference for CUDA's fixed synthetic prefix probe.
// This is a numerical diagnostic; it does not route inference over SSH.
internal static class Bf16PrefixReference
{
    public static async Task Run(Options options, CancellationToken ct)
    {
        if (Environment.GetEnvironmentVariable("DEEPINFER_METAL_KV") != "bf16" ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_GEMM") != "bf16-nax" ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_CHUNK") != "128" ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_MATRIX") != "1" ||
            Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_DELTA") != "sequential")
            throw new ArgumentException("Set BF16 KV/GEMM, 128-token chunk, matrix=1 and sequential Delta explicitly.");
        var output = Path.GetFullPath(options.Get("output", "artifacts/bf16-metal-prefix"));
        if (Directory.Exists(output)) throw new ArgumentException("Use a fresh evidence directory.");
        Directory.CreateDirectory(output);
        var tokens = Enumerable.Range(0, 128).Select(i => 100 + (i * 137) % 247000).ToArray();
        using var gpu = new MetalSplitSession(options.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx"),
            2, residency: true, prefillScratchTokens: 128);
        var profile = gpu.ExecutionProfile();
        if (profile.KvElementBytes != 2 || !profile.Bf16Gemm || !profile.Matrix || profile.ScratchTokens < 128)
            throw new InvalidOperationException("Native Metal profile does not match BF16 reference requirements.");
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
            var state = new byte[CudaPrefixSession.StateBytes(layer, kind, tokens.Length, 2)];
            gpu.State(layer, kind, tokens.Length, state, false);
            await File.WriteAllBytesAsync(Path.Combine(output, $"state-{layer}-{kind}.bin"), state, ct);
        }
        await File.WriteAllTextAsync(Path.Combine(output, "tokens.json"),
            JsonSerializer.Serialize(tokens, RpcJson.Default.Int32Array), ct);
        using var file = File.Create(Path.Combine(output, "result.json"));
        using var json = new Utf8JsonWriter(file, new JsonWriterOptions { Indented = true });
        json.WriteStartObject();
        json.WriteString("kind", "independent-metal-bf16-prefix-reference-not-performance");
        json.WriteNumber("tokens", tokens.Length);
        json.WriteNumber("weightBits", 8);
        json.WriteString("kv", "bf16");
        json.WriteNumber("chunkTokens", 128);
        json.WriteNumber("nextToken", next);
        json.WriteString("traceSha256", Convert.ToHexStringLower(SHA256.HashData(bytes)));
        json.WriteNumber("metalAllocatedBytes", gpu.Stats().AllocatedBytes);
        json.WriteEndObject();
        await json.FlushAsync(ct);
        Console.WriteLine($"Metal BF16 prefix reference: {tokens.Length} tokens, next {next}, trace {output}");
    }
}
