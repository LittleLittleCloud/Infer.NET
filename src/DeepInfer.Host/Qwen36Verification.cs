using System.Runtime.InteropServices;
using System.Text.Json;
using DeepInfer.Backends.Models;
using DeepInfer.Core;
namespace DeepInfer.Host;

public static class Qwen36Verification
{
    public static async Task RunAsync(Options options, CancellationToken ct)
    {
        var directory = options.Get("weights", "models/qwen3.6-35b-a3b-5bit-mlx");
        var referencePath = Path.GetFullPath(options.Get("reference", "artifacts/reference/qwen36/reference.json"));
        var output = Path.GetFullPath(options.Get("output", "artifacts/reference/qwen36/own"));
        Directory.CreateDirectory(output);
        using var reference = JsonDocument.Parse(File.ReadAllBytes(referencePath)); var r = reference.RootElement;
        var tokenizer = new QwenTokenizer(Path.Combine(directory, "tokenizer.json"));
        var prompt = tokenizer.EncodePrompt(r.GetProperty("prompt").GetString()!, ct);
        if (!prompt.SequenceEqual(r.GetProperty("promptTokenIds").EnumerateArray().Select(x => x.GetInt32())))
            throw new InvalidOperationException("Qwen3.6 chat token IDs differ from the independent reference.");
        await using var model = new Qwen36Model(new(Protocol.Qwen36Model, "mlx-qwen36-affine", false), directory, Path.Combine(directory, "tokenizer.json"), allowDiagnosticRouting: true);
        var inputs = r.GetProperty("inputTokenIds").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var positions = r.GetProperty("tracePositions").EnumerateArray().Select(x => x.GetInt32()).ToHashSet();
        using var allTrace = options.Flag("trace-all") ? File.Create(Path.Combine(output, "all-states.f32")) : null;
        var routePath = options.Get("route-trace", "");
        using var routeTrace = routePath.Length == 0 ? null : File.Create(routePath);
        var logits = model.EvaluateTokens(inputs, ct, (p, l, x) =>
        {
            allTrace?.Write(MemoryMarshal.AsBytes(x.AsSpan()));
            if (positions.Contains(p)) File.WriteAllBytes(Path.Combine(output, $"p{p}-layer{l}.f32"), MemoryMarshal.AsBytes(x.AsSpan()).ToArray());
        }, routeTrace is null ? null : (p, l, ids) => routeTrace.Write(MemoryMarshal.AsBytes(ids.AsSpan())), prefill: options.Flag("prefill"));
        File.WriteAllBytes(Path.Combine(output, "logits.f32"), MemoryMarshal.AsBytes(logits.AsSpan()).ToArray());
        Console.WriteLine($"Own .NET/Metal evaluation completed. tokens={inputs.Length}; argmax={Qwen36Model.ArgMax(logits)}; trace={output}");
        var expected = r.GetProperty("generatedTokenIds").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var generated = await model.GenerateAsync(new(r.GetProperty("prompt").GetString()!, expected.Length, Protocol.Qwen36Model, "metal"), ct);
        if (generated.TokenIds is null || !generated.TokenIds.SequenceEqual(expected))
            throw new InvalidOperationException("Independent greedy generation token IDs differ.");
        Console.WriteLine($"Independent greedy token verification passed: {generated.Text}");
    }
}
