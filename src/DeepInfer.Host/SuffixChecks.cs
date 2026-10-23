using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using DeepInfer.Backends.Split;
namespace DeepInfer.Host;

// Isolate suffix reblocking from CUDA/network differences with the resident pinned model.
// Run with the deployed Metal model stopped. All data stays in the supplied evidence folder.
internal static class SuffixChecks
{
    public static async Task Run(Options o, CancellationToken ct)
    {
        if (Environment.GetEnvironmentVariable("DEEPINFER_METAL_KV") != "fp32")
            throw new ArgumentException("Start this diagnostic with DEEPINFER_METAL_KV=fp32 in the process environment.");
        var directory = Path.GetFullPath(o.Get("output", "artifacts/suffix-checks"));
        if (Directory.Exists(directory)) throw new ArgumentException("Use a fresh evidence directory.");
        Directory.CreateDirectory(directory);
        using var gpu = new MetalSplitSession(o.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx"), 4, residency: true, prefillScratchTokens: 512);
        var prompt = o.Get("prompt-file", "") is { Length: > 0 } file ? await File.ReadAllTextAsync(file, ct) : o.Get("prompt", "Explain causal attention.");
        var tokens = gpu.Tokenize(prompt, ct);
        if (tokens.Length is < 1 or > 1024) throw new ArgumentException("Suffix correctness input is limited to 1024 tokens including template.");
        const int generated = 8;
        var reference = new float[tokens.Length * 40 * 2048];
        var expectedIds = new int[generated]; var expectedLogits = new float[generated][];
        var prefix = new byte[40][];
        gpu.Reset(tokens.Length + generated);
        var logits = new float[248320];
        var next = gpu.Reference(tokens, ct, logits, (p, l, values) => values.CopyTo(reference, (p * 40 + l) * 2048));
        Save("reference-trace.f32", reference);
        for (var l = 0; l < 20; l++) for (var t = 0; t < 2; t++)
        {
            var state = new byte[CudaPrefixSession.StateBytes(l, t, tokens.Length, 4)];
            gpu.State(l, t, tokens.Length, state, false); prefix[l * 2 + t] = state;
            await File.WriteAllBytesAsync(Path.Combine(directory, $"reference-state-{l}-{t}.bin"), state, ct);
        }
        for (var i = 0; i < generated; i++)
        {
            expectedIds[i] = next; expectedLogits[i] = (float[])logits.Clone(); Save($"reference-logits-{i}.f32", logits);
            if (i + 1 < generated) next = gpu.Forward(next, tokens.Length + i, logits);
        }
        using var output = File.Create(Path.Combine(directory, "result.json"));
        using var json = new Utf8JsonWriter(output, new() { Indented = true });
        json.WriteStartObject(); json.WriteNumber("promptTokens", tokens.Length); json.WriteNumber("weightBits", 8);
        json.WriteString("kv", "fp32"); json.WriteString("reference", "full Metal model, 32-token blocks, same FP32 profile");
        json.WriteStartArray("configurations"); var failures = 0;
        foreach (var chunk in new[] { 32, 128, 512 })
        {
            ct.ThrowIfCancellationRequested(); gpu.Reset(tokens.Length + generated);
            var timer = Stopwatch.StartNew(); var before = failures;
            json.WriteStartObject(); json.WriteNumber("suffixChunk", chunk); json.WriteStartArray("checks");
            for (var offset = 0; offset < tokens.Length; offset += chunk)
            {
                ct.ThrowIfCancellationRequested(); var count = Math.Min(chunk, tokens.Length - offset);
                var hidden = new float[count * 2048]; var trace = new float[count * 40 * 2048];
                for (var t = 0; t < count; t++) reference.AsSpan(((offset + t) * 40 + 19) * 2048, 2048).CopyTo(hidden.AsSpan(t * 2048));
                next = gpu.Suffix(tokens.AsSpan(offset, count).ToArray(), hidden, offset + count == tokens.Length, logits, trace);
                Save($"chunk-{chunk}-trace-{offset}.f32", trace);
                for (var t = 0; t < count; t++) for (var l = 20; l < 40; l++)
                    Compare($"p{offset + t}/l{l}", reference.AsSpan(((offset + t) * 40 + l) * 2048, 2048), trace.AsSpan((t * 40 + l) * 2048, 2048));
            }
            for (var l = 0; l < 20; l++) for (var t = 0; t < 2; t++) gpu.State(l, t, tokens.Length, prefix[l * 2 + t], true);
            var actualIds = new int[generated];
            for (var i = 0; i < generated; i++)
            {
                actualIds[i] = next; Compare($"logits/{i}", expectedLogits[i], logits); Save($"chunk-{chunk}-logits-{i}.f32", logits);
                if (i + 1 < generated) next = gpu.Forward(next, tokens.Length + i, logits);
            }
            json.WriteEndArray(); var equal = actualIds.SequenceEqual(expectedIds); if (!equal) failures++;
            json.WriteBoolean("greedyEqual", equal); json.WriteNumber("failedChecks", failures - before);
            json.WriteNumber("diagnosticSecondsIncludingTraceAndDisk", timer.Elapsed.TotalSeconds);
            json.WriteNumber("metalAllocatedBytes", gpu.Stats().AllocatedBytes); json.WriteEndObject(); json.Flush();
            Console.WriteLine($"SUFFIX_CHECK chunk={chunk} tokens={tokens.Length} failed={failures - before} greedyEqual={equal}");
        }
        json.WriteEndArray(); json.WriteNumber("failedChecks", failures); json.WriteEndObject(); json.Flush();
        if (failures != 0) throw new InvalidOperationException("Suffix reblocking numeric gate failed; retain evidence, do not deploy that policy.");
        void Save(string name, float[] values) => File.WriteAllBytes(Path.Combine(directory, name), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        void Compare(string name, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
        {
            double error = 0, norm = 0, max = 0;
            for (var i = 0; i < expected.Length; i++) { var d = (double)expected[i] - actual[i]; error += d * d; norm += (double)expected[i] * expected[i]; max = Math.Max(max, Math.Abs(d)); }
            var relative = Math.Sqrt(error / Math.Max(norm, 1e-30)); var finite = double.IsFinite(relative) && double.IsFinite(max);
            var pass = finite && relative < 1e-4 && max < .01; if (!pass) failures++;
            json.WriteStartObject(); json.WriteString("name", name); json.WriteBoolean("pass", pass); json.WriteBoolean("finite", finite);
            if (finite) { json.WriteNumber("relativeL2", relative); json.WriteNumber("maxAbs", max); }
            json.WriteEndObject();
        }
    }
}
