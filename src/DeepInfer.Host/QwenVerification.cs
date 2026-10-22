using System.Text.Json;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using DeepInfer.Core;
namespace DeepInfer.Host;

public static class QwenVerification
{
    public static async Task RunAsync(Options options, CancellationToken ct)
    {
        using (var metal = new MetalMatVecDevice())
        using (var cpu = new CpuMatVecDevice())
        {
            var random = new Random(42);
            var bf = Enumerable.Range(0, 53 * 77).Select(_ => (ushort)(BitConverter.SingleToInt32Bits((float)(random.NextDouble()*2-1)) >> 16)).ToArray();
            var input = Enumerable.Range(0,77).Select(_ => (float)random.NextDouble()).ToArray();
            using var a = cpu.UploadBFloat16(bf,53,77); using var b = metal.UploadBFloat16(bf,53,77);
            var expected = new float[53]; var actual = new float[53]; a.Multiply(input,expected);b.Multiply(input,actual);
            var error = expected.Zip(actual,(x,y) => MathF.Abs(x-y)).Max();
            if (!float.IsFinite(error) || error > 0.0001f) throw new InvalidOperationException($"BF16 kernel mismatch: {error}.");
            Console.WriteLine($"PASS: BF16 weights / FP32 accumulation GEMV; max absolute error={error}");
        }
        var directory = options.Get("weights","models/qwen2.5-3b-instruct");
        var tokenizer = new QwenTokenizer(Path.Combine(directory,"tokenizer.json"));
        var referencePath = options.Get("reference", "");
        using var reference = referencePath.Length > 0 ? JsonDocument.Parse(File.ReadAllBytes(referencePath)) : null;
        if (reference is not null)
            foreach (var item in reference.RootElement.GetProperty("tokenizerCases").EnumerateArray())
            {
                var prompt = item.GetProperty("prompt").GetString()!;
                if (!tokenizer.EncodePrompt(prompt, ct).SequenceEqual(item.GetProperty("tokenIds").EnumerateArray().Select(x => x.GetInt32())))
                    throw new InvalidOperationException($"Qwen tokenizer/chat template mismatch: {prompt}");
            }
        Console.WriteLine("PASS: Qwen tokenizer initialized; reference tokenization matches when provided.");
        var modelId = options.Get("model", Path.GetFileName(Path.GetFullPath(directory)).Contains("7b",StringComparison.OrdinalIgnoreCase) ? Protocol.Qwen7BModel : Protocol.Qwen3BModel);
        Console.WriteLine($"Loading {modelId} BF16 checkpoint into Metal buffers…");
        await using var model = new LlamaModel(new(modelId,"hf-qwen2",false),directory,Path.Combine(directory,"tokenizer.json"),"metal");
        var promptText = options.Get("prompt","用一句话介绍你自己。");
        if (reference is not null && reference.RootElement.TryGetProperty("logits",out var expectedLogits))
        {
            promptText = reference.RootElement.GetProperty("prompt").GetString()!;
            var actual = model.EvaluatePrompt(promptText,ct); var expected = expectedLogits.EnumerateArray().Select(x => x.GetSingle()).ToArray();
            if (actual.Length != expected.Length) throw new InvalidOperationException("Logit count mismatch.");
            var maximum = actual.Zip(expected,(x,y) => MathF.Abs(x-y)).Max();
            if (!float.IsFinite(maximum) || maximum > 0.01f || LlamaModel.ArgMax(actual) != LlamaModel.ArgMax(expected))
                throw new InvalidOperationException($"Qwen full-logit reference mismatch; max absolute error={maximum}.");
            Console.WriteLine($"PASS: {actual.Length} logits versus independent Transformers FP32 reference; max absolute error={maximum}");
        }
        var count = options.Number("max-tokens",48,1,Protocol.MaxTokens);
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var result = await model.GenerateAsync(new(promptText,count,modelId,"metal"),ct);
        if (result.IsSimulation || result.GpuDispatches < 1 || string.IsNullOrWhiteSpace(result.Text)) throw new InvalidOperationException("Qwen did not produce real GPU-backed text.");
        if (reference is not null && reference.RootElement.TryGetProperty("generatedTokenIds",out var expectedIds) &&
            !result.TokenIds!.SequenceEqual(expectedIds.EnumerateArray().Select(x => x.GetInt32())))
            throw new InvalidOperationException("Qwen greedy output token IDs differ from independent reference.");
        Console.WriteLine($"PASS: {result.TokenIds!.Length} real Qwen tokens; {result.GpuDispatches} GPU dispatches; elapsed={timer.Elapsed.TotalMilliseconds:F1}ms; finish={result.FinishReason}");
        Console.WriteLine(result.Text);
    }
}
