using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using DeepInfer.Core;
namespace DeepInfer.Host;

public static class ModelVerification
{
    public static async Task RunAsync(Options options, CancellationToken ct)
    {
        using var gpu = new MetalMatVecDevice(); using var cpu = new CpuMatVecDevice();
        var random = new Random(123);
        foreach (var (rows, columns) in new[] { (1, 1), (53, 77), (288, 288), (32000, 288) })
        {
            var values = Enumerable.Range(0, rows * columns).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            var input = Enumerable.Range(0, columns).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            using var a = cpu.Upload(values, rows, columns); using var b = gpu.Upload(values, rows, columns);
            var expected = new float[rows]; var actual = new float[rows]; a.Multiply(input, expected); b.Multiply(input, actual);
            Compare(expected, actual, 0.0001f, $"GEMV {rows}x{columns}");
        }
        var info = new ModelInfo(options.Get("model", Protocol.TinyStoriesModel), "llama2c-f32", false);
        var weights = options.Get("weights", "models/stories15M.bin"); var tokenizer = options.Get("tokenizer", "models/tokenizer.bin");
        await using var reference = new LlamaModel(info, weights, tokenizer, "cpu");
        await using var metal = new LlamaModel(info, weights, tokenizer, "metal");
        foreach (var prompt in new[] { "Once upon a time", "The little girl", "Hello 世界" })
        {
            var expected = reference.EvaluatePrompt(prompt, ct); var actual = metal.EvaluatePrompt(prompt, ct);
            Compare(expected, actual, 0.002f, $"complete {expected.Length}-logit prefill: {prompt}");
            if (LlamaModel.ArgMax(expected) != LlamaModel.ArgMax(actual)) throw new InvalidOperationException("CPU/Metal top token differs.");
        }
        var cpuResult = await reference.GenerateAsync(new("Once upon a time", 48, info.Id, "cpu"), ct);
        var metalResult = await metal.GenerateAsync(new("Once upon a time", 48, info.Id, "metal"), ct);
        if (!cpuResult.TokenIds!.SequenceEqual(metalResult.TokenIds!)) throw new InvalidOperationException("CPU/Metal 48-token greedy generation differs.");
        if (metalResult.IsSimulation || metalResult.GpuDispatches == 0 || string.IsNullOrWhiteSpace(metalResult.Text)) throw new InvalidOperationException("Real Metal output was not produced.");
        Console.WriteLine($"PASS: CPU/Metal greedy token IDs match ({metalResult.TokenIds!.Length} tokens); {metalResult.GpuDispatches} actual GPU dispatches; GPU time={metalResult.GpuComputeMs:F3}ms");
        Console.WriteLine(metalResult.Text);
    }
    private static void Compare(float[] expected, float[] actual, float tolerance, string label)
    {
        var maximum = 0f;
        for (var i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(actual[i])) throw new InvalidOperationException($"{label}: non-finite GPU output.");
            maximum = MathF.Max(maximum, MathF.Abs(expected[i] - actual[i]));
        }
        if (maximum > tolerance) throw new InvalidOperationException($"{label}: max absolute error {maximum} exceeds {tolerance}.");
        Console.WriteLine($"PASS: {label}; max absolute error={maximum:G6}, tolerance={tolerance}");
    }
}
