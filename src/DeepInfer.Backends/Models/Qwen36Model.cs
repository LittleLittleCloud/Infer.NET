using System.Diagnostics;
using System.Numerics;
using DeepInfer.Backends.Compute;
using DeepInfer.Core;
namespace DeepInfer.Backends.Models;

// Original single-token engine. The default executor keeps all model math on
// Metal; the C# layer executor remains available for differential verification.
public sealed class Qwen36Model : IInferenceModel
{
    private readonly QuantizedMetal device;
    private readonly Qwen36Weights weights;
    private readonly QwenTokenizer tokenizer;
    private readonly bool fuseExperts;
    private readonly Qwen36Gpu? gpu;
    private readonly int debugPosition = int.TryParse(Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_DEBUG_POSITION"), out var p) ? p : 511;
    private Action<int, int, int[]>? routeTrace;
    private readonly SemaphoreSlim gate = new(1, 1);
    private bool disposed;
    public ModelInfo Info { get; }
    public Qwen36Model(ModelInfo info, string directory, string tokenizerPath, bool allowDiagnosticRouting = false)
    {
        if (!allowDiagnosticRouting && Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_FORCE_ROUTES") is not null)
            throw new NotSupportedException("Forced routing is for explicit numerical verification only; unavailable in normal inference workers.");
        if (info.Id != Protocol.Qwen36Model || info.IsSimulation || info.Format != "mlx-qwen36-affine") throw new NotSupportedException("Expected quantized Qwen3.6 text weights.");
        if (Qwen36Gpu.ReadLmHeadMode() != 0 &&
            (Environment.GetEnvironmentVariable("DEEPINFER_METAL_SERIAL") == "1" ||
             Environment.GetEnvironmentVariable("DEEPINFER_METAL_EXPERT_UNFUSED") == "1" ||
             Environment.GetEnvironmentVariable("DEEPINFER_METAL_GENERIC") == "1" ||
             Environment.GetEnvironmentVariable("DEEPINFER_METAL_HOST_EXECUTOR") == "1"))
            throw new NotSupportedException("tile4 LM head requires the native GPU graph executor.");
        Info = info; device = new(batchSubmissions: Environment.GetEnvironmentVariable("DEEPINFER_METAL_SERIAL") != "1");
        fuseExperts = device.BatchSubmissions && Environment.GetEnvironmentVariable("DEEPINFER_METAL_EXPERT_UNFUSED") != "1";
        try
        {
            weights = new(directory, device);
            try
            {
                tokenizer = new(tokenizerPath);
                if (fuseExperts && Environment.GetEnvironmentVariable("DEEPINFER_METAL_GENERIC") != "1" &&
                    Environment.GetEnvironmentVariable("DEEPINFER_METAL_HOST_EXECUTOR") != "1") gpu = new(device, weights);
            }
            catch { weights.Dispose(); throw; }
        }
        catch { device.Dispose(); throw; }
    }
    public async ValueTask<ModelOutput> GenerateAsync(InferenceRequest request, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var error = RequestValidation.Error(request);
        if (error is not null || request.Model != Info.Id || request.Backend != "metal")
            throw new InferenceInputException(error ?? "Model/backend mismatch.");
        await gate.WaitAsync(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return await Task.Run(() => Generate(request, cancellationToken), cancellationToken);
        }
        finally { gate.Release(); }
    }
    private ModelOutput Generate(InferenceRequest request, CancellationToken ct)
    {
        var started = Stopwatch.StartNew(); var prompt = tokenizer.EncodePrompt(request.Prompt, ct);
        if (prompt.Length >= Qwen36Weights.Context) throw new InferenceInputException($"Qwen3.6 context is capped at {Qwen36Weights.Context} tokens, including generation; unscaled text RoPE only.");
        var capacity = Math.Min(Qwen36Weights.Context, prompt.Length + request.MaxTokens);
        using var state = new State(weights, device, gpu is not null, capacity);
        gpu?.Reset();
        gpu?.Reserve(capacity);
        var dispatches = device.Dispatches; var submissions = device.Submissions; var gpuMs = device.GpuMilliseconds;
        var prefill = Stopwatch.StartNew();
        if (gpu is not null && Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL") != "0" &&
            Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_FORCE_ROUTES") is null &&
            Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_DEBUG_DIR") is null)
            state.NextToken = gpu.Prefill(prompt, ct, progress: request.Profile ? count =>
            {
                if (count % 1024 == 0 || count == prompt.Length)
                    Console.Error.WriteLine($"PREFILL_PROGRESS tokens={count}/{prompt.Length} elapsedMs={prefill.Elapsed.TotalMilliseconds:F1}");
            } : null);
        else for (var p = 0; p < prompt.Length; p++) Forward(state, prompt[p], p, ct, p == prompt.Length - 1);
        prefill.Stop();
        var maximum = Math.Min(request.MaxTokens, Qwen36Weights.Context - prompt.Length);
        var ids = new List<int>(); var pieces = new List<string>(); double first = 0, last = 0;
        var windows = new List<DecodeWindow>(); int windowStart = 2; double windowTime = 0;
        void RecordWindow()
        {
            var intervals = ids.Count - windowStart + 1;
            if (!request.Profile || intervals <= 0) return;
            var ms = last - windowTime;
            windows.Add(new(windowStart, ids.Count, prompt.Length + ids.Count - 1, ms, ms > 0 ? intervals * 1000d / ms : 0));
            Console.Error.WriteLine($"DECODE_PROGRESS outputTokens={ids.Count} contextTokens={prompt.Length + ids.Count - 1} tokensPerSecond={windows[^1].TokensPerSecond:F2}");
            windowStart = ids.Count + 1; windowTime = last;
        }
        var finish = maximum < request.MaxTokens ? "context_length" : "length";
        for (var n = 0; n < maximum; n++)
        {
            ct.ThrowIfCancellationRequested(); var next = gpu is null ? ArgMax(state.Logits) : state.NextToken;
            if (tokenizer.IsEnd(next) && !request.BenchmarkIgnoreEos) { finish = "eos"; break; }
            ids.Add(next); pieces.Add(System.Text.Encoding.UTF8.GetString(tokenizer.DecodePiece(0, next)));
            last = started.Elapsed.TotalMilliseconds; if (ids.Count == 1) { first = last; windowTime = first; }
            if (ids.Count - windowStart + 1 == 256) RecordWindow();
            if (n + 1 < maximum) Forward(state, next, prompt.Length + n, ct, true);
        }
        RecordWindow();
        var stats = gpu?.Stats() ?? (capacity, 40960L * capacity, 0L);
        return new(pieces.ToArray(), false, tokenizer.Decode(ids.ToArray()), ids.ToArray(), prompt.Length, finish,
            device.Dispatches - dispatches, device.GpuMilliseconds - gpuMs,
            InferenceTimings.Create(prefill.Elapsed.TotalMilliseconds, first, last - first, ids.Count) with
            { DecodeWindows = request.Profile ? windows.ToArray() : null,
                ContextCapacity = stats.Item1, KvCacheBytes = stats.Item2, MetalAllocatedBytes = stats.Item3 }, device.Submissions - submissions);
    }
    public float[] EvaluateTokens(int[] tokens, CancellationToken ct = default, Action<int, int, float[]>? trace = null, Action<int, int, int[]>? routeTrace = null, bool prefill = false)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (tokens.Length is < 1 or > Qwen36Weights.Context || tokens.Any(x => x < 0 || x >= Qwen36Weights.Vocabulary))
            throw new ArgumentException("Invalid evaluation token sequence.");
        gate.Wait(ct);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (gpu is not null && routeTrace is not null) throw new NotSupportedException("Route export requires DEEPINFER_METAL_HOST_EXECUTOR=1.");
            this.routeTrace = routeTrace;
            using var state = new State(weights, device, gpu is not null, tokens.Length);
            gpu?.Reset();
            gpu?.Reserve(tokens.Length);
            if (prefill)
            {
                if (gpu is null) throw new NotSupportedException("Batched prefill verification requires the GPU executor.");
                state.NextToken = gpu.Prefill(tokens, ct, state.Logits, trace);
            }
            else for (var p = 0; p < tokens.Length; p++) Forward(state, tokens[p], p, ct, true, trace, true);
            return state.Logits.ToArray();
        }
        finally { this.routeTrace = null; gate.Release(); }
    }
    private void Forward(State s, int token, int position, CancellationToken ct, bool logits, Action<int, int, float[]>? trace = null, bool fullLogits = false)
    {
        ct.ThrowIfCancellationRequested();
        if (gpu is not null) { s.NextToken = gpu.Forward(token, position, logits, s.Logits, trace, fullLogits); return; }
        weights.Embedding.CopyRow(token, s.X);
        for (var l = 0; l < Qwen36Weights.Layers; l++)
        {
            ct.ThrowIfCancellationRequested(); var w = weights.Blocks[l]; var cache = s.Caches[l];
            s.Position = position; s.Layer = l;
            Norm(s.X, w.InputNorm, s.Input, weights.Epsilon);
            if (w.Linear) Linear(s, cache, w); else Attention(s, cache, w, position);
            w.O.Multiply(s.Attention, s.Output);
            for (var d = 0; d < 2048; d++) s.X[d] += s.Output[d];
            Norm(s.X, w.PostNorm, s.Input, weights.Epsilon);
            Moe(s, w);
            for (var d = 0; d < 2048; d++) s.X[d] += s.Output[d];
            trace?.Invoke(position, l, s.X);
        }
        if (logits)
        {
            Norm(s.X, weights.FinalNorm, s.Input, weights.Epsilon);
            weights.Classifier.Multiply(s.Input, s.Logits);
        }
    }
    public static int ArgMax(ReadOnlySpan<float> values)
    {
        var index = 0;
        for (var i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i])) throw new InvalidOperationException("Model produced non-finite logits.");
            if (values[i] > values[index]) index = i;
        }
        return index;
    }
    private void Linear(State s, LayerState cache, Qwen36Weights.Layer w)
    {
        device.MultiplyBatch([new(w.Q, s.Input, s.Qkv), new(w.Z, s.Input, s.Z), new(w.A, s.Input, s.A), new(w.B, s.Input, s.B)]);
        for (var d = 0; d < 8192; d++)
        {
            var t = cache.Conv[d] * w.Conv[d * 4] + cache.Conv[8192 + d] * w.Conv[d * 4 + 1] +
                cache.Conv[16384 + d] * w.Conv[d * 4 + 2] + s.Qkv[d] * w.Conv[d * 4 + 3];
            s.Conv[d] = Silu(t);
        }
        Array.Copy(cache.Conv, 8192, cache.Conv, 0, 16384); s.Qkv.CopyTo(cache.Conv, 16384);
        for (var h = 0; h < 16; h++)
        {
            var q = s.Conv.AsSpan(h * 128, 128); var k = s.Conv.AsSpan(2048 + h * 128, 128);
            var qs = (1f / MathF.Sqrt(128)) / MathF.Sqrt(Dot(q, q) + 1e-6f);
            var ks = 1f / MathF.Sqrt(Dot(k, k) + 1e-6f);
            for (var d = 0; d < 128; d++) { q[d] *= qs; k[d] *= ks; }
        }
        for (var h = 0; h < 32; h++)
        {
            var a = s.A[h] + w.DtBias[h];
            var softplus = a > 20 ? a : MathF.Log(1 + MathF.Exp(a));
            s.A[h] = MathF.Exp(-MathF.Exp(w.ALog[h]) * softplus); s.B[h] = Sigmoid(s.B[h]);
        }
        cache.Delta!.Step(s.Conv.AsSpan(0, 2048), s.Conv.AsSpan(2048, 2048), s.Conv.AsSpan(4096, 4096), s.A, s.B, s.Attention);
        for (var h = 0; h < 32; h++)
        {
            var output = s.Attention.AsSpan(h * 128, 128);
            Norm(output, w.GatedNorm, output, weights.Epsilon);
            for (var d = 0; d < 128; d++) output[d] *= Silu(s.Z[h * 128 + d]);
        }
    }
    private void Attention(State s, LayerState cache, Qwen36Weights.Layer w, int position)
    {
        device.MultiplyBatch([new(w.Q, s.Input, s.Qkv), new(w.K, s.Input, s.K), new(w.V, s.Input, s.V)]);
        for (var h = 0; h < 16; h++)
        {
            Norm(s.Qkv.AsSpan(h * 512, 256), w.QNorm, s.Q.AsSpan(h * 256, 256), weights.Epsilon);
            Rope(s.Q.AsSpan(h * 256, 256), position);
        }
        for (var h = 0; h < 2; h++)
        {
            var k = s.K.AsSpan(h * 256, 256); Norm(k, w.KNorm, k, weights.Epsilon); Rope(k, position);
        }
        s.K.CopyTo(cache.Keys, position * 512); s.V.CopyTo(cache.Values, position * 512);
        Array.Clear(s.Attention);
        for (var h = 0; h < 16; h++)
        {
            var scores = s.Scores.AsSpan(0, position + 1); var q = s.Q.AsSpan(h * 256, 256); var kh = h / 8;
            for (var t = 0; t <= position; t++) scores[t] = Dot(q, cache.Keys.AsSpan(t * 512 + kh * 256, 256)) / 16;
            Softmax(scores);
            var result = s.Attention.AsSpan(h * 256, 256);
            for (var t = 0; t <= position; t++) AddScaled(result, cache.Values.AsSpan(t * 512 + kh * 256, 256), scores[t]);
            for (var d = 0; d < 256; d++) result[d] *= Sigmoid(s.Qkv[h * 512 + 256 + d]);
        }
    }
    private void Moe(State s, Qwen36Weights.Layer w)
    {
        // Stage 1: routing and shared projections depend only on the same layer input.
        device.MultiplyBatch([new(w.Router, s.Input, s.Router), new(w.SharedGate, s.Input, s.SharedGate),
            new(w.SharedUp, s.Input, s.SharedUp), new(w.SharedRoute, s.Input, s.SharedRoute)]);
        SelectExperts(s.Router, s.Selected, s.Probabilities);
        routeTrace?.Invoke(s.Position, s.Layer, s.Selected);
        if (s.Position == debugPosition && Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_DEBUG_DIR") is { } debug)
        {
            File.WriteAllBytes(Path.Combine(debug, $"host-router-{s.Layer}.f32"), System.Runtime.InteropServices.MemoryMarshal.AsBytes(s.Router.AsSpan()).ToArray());
            File.WriteAllText(Path.Combine(debug, $"host-ids-{s.Layer}.json"), "[" + string.Join(',', s.Selected) + "]");
        }
        for (var i = 0; i < 512; i++) s.SharedGate[i] = Silu(s.SharedGate[i]) * s.SharedUp[i];
        if (fuseExperts)
            device.ExpertMlp(w.ExpertGate, w.ExpertUp, w.ExpertDown, w.SharedDown,
                s.Input, s.SharedGate, s.Selected, s.ExpertOut, s.Output);
        else
        {
            // Diagnostic path retains the preceding version's CPU expert activation.
            device.MultiplyBatch([new(w.ExpertGate, s.Input, s.ExpertGate, s.Selected),
                new(w.ExpertUp, s.Input, s.ExpertUp, s.Selected), new(w.SharedDown, s.SharedGate, s.Output)]);
            for (var i = 0; i < s.ExpertGate.Length; i++) s.ExpertGate[i] = Silu(s.ExpertGate[i]) * s.ExpertUp[i];
            w.ExpertDown.MultiplyExperts(s.ExpertGate, s.ExpertOut, s.Selected, batched: true);
        }
        var scale = Sigmoid(s.SharedRoute[0]);
        for (var d = 0; d < 2048; d++) s.Output[d] *= scale;
        for (var i = 0; i < 8; i++) AddScaled(s.Output, s.ExpertOut.AsSpan(i * 2048, 2048), s.Probabilities[i]);
    }
    public static void SelectExperts(ReadOnlySpan<float> logits, Span<int> ids, Span<float> probabilities)
    {
        if (logits.Length != 256 || ids.Length != 8 || probabilities.Length != 8) throw new ArgumentException("Expected 256 experts/top8.");
        ids.Fill(-1);
        for (var e = 0; e < logits.Length; e++)
        {
            if (!float.IsFinite(logits[e])) throw new InvalidOperationException("Non-finite router logits.");
            for (var j = 0; j < 8; j++) if (ids[j] < 0 || logits[e] > logits[ids[j]])
            {
                for (var k = 7; k > j; k--) ids[k] = ids[k - 1];
                ids[j] = e; break;
            }
        }
        // Softmax over the selected logits equals full softmax followed by top-k renormalization.
        for (var i = 0; i < 8; i++) probabilities[i] = logits[ids[i]];
        Softmax(probabilities);
    }
    private void Rope(Span<float> head, int position)
    {
        for (var d = 0; d < 32; d++)
        {
            var angle = position / MathF.Pow(weights.RopeTheta, d / 32f);
            var sin = MathF.Sin(angle); var cos = MathF.Cos(angle); var a = head[d]; var b = head[d + 32];
            head[d] = a * cos - b * sin; head[d + 32] = b * cos + a * sin;
        }
    }
    private static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));
    private static float Silu(float x) => x * Sigmoid(x);
    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        var sum = Vector<float>.Zero; var i = 0;
        for (; i <= a.Length - Vector<float>.Count; i += Vector<float>.Count)
            sum += new Vector<float>(a.Slice(i)) * new Vector<float>(b.Slice(i));
        var result = Vector.Sum(sum); for (; i < a.Length; i++) result += a[i] * b[i]; return result;
    }
    private static void AddScaled(Span<float> y, ReadOnlySpan<float> x, float scale)
    {
        var factor = new Vector<float>(scale); var i = 0;
        for (; i <= y.Length - Vector<float>.Count; i += Vector<float>.Count)
            (new Vector<float>(y.Slice(i)) + factor * new Vector<float>(x.Slice(i))).CopyTo(y.Slice(i));
        for (; i < y.Length; i++) y[i] += scale * x[i];
    }
    private static void Norm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> y, float epsilon)
    {
        var scale = 1 / MathF.Sqrt(Dot(x, x) / x.Length + epsilon);
        for (var i = 0; i < x.Length; i++) y[i] = x[i] * scale * weight[i];
    }
    private static void Softmax(Span<float> x)
    {
        var max = float.NegativeInfinity; foreach (var v in x) max = MathF.Max(max, v);
        var sum = 0f; for (var i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        for (var i = 0; i < x.Length; i++) x[i] /= sum;
    }
    public async ValueTask DisposeAsync()
    {
        await gate.WaitAsync();
        try { if (!disposed) { disposed = true; gpu?.Dispose(); weights.Dispose(); device.Dispose(); } }
        finally { gate.Release(); }
    }
    private sealed class LayerState : IDisposable
    {
        public float[] Conv = [], Keys = [], Values = [];
        public DeltaMemory? Delta;
        public void Dispose() => Delta?.Dispose();
    }
    private sealed class State : IDisposable
    {
        public int Position, Layer, NextToken;
        public readonly float[] X = new float[2048], Input = new float[2048], Output = new float[2048];
        public readonly float[] Qkv = new float[8192], Conv = new float[8192], Z = new float[4096], A = new float[32], B = new float[32];
        public readonly float[] Attention = new float[4096], Q = new float[4096], K = new float[512], V = new float[512];
        public readonly float[] Router = new float[256], Probabilities = new float[8], ExpertGate = new float[4096], ExpertUp = new float[4096], ExpertOut = new float[16384];
        public readonly int[] Selected = new int[8];
        public readonly float[] SharedGate = new float[512], SharedUp = new float[512], SharedRoute = new float[1];
        public readonly float[] Scores, Logits = new float[Qwen36Weights.Vocabulary];
        public readonly LayerState[] Caches = new LayerState[40];
        public State(Qwen36Weights weights, QuantizedMetal device, bool gpu = false, int capacity = 2048)
        {
            Scores = gpu ? [] : new float[capacity];
            if (gpu) return;
            try
            {
                for (var l = 0; l < 40; l++) Caches[l] = weights.Blocks[l].Linear
                    ? new() { Conv = new float[3 * 8192], Delta = device.NewDeltaMemory() }
                    : new() { Keys = new float[capacity * 512], Values = new float[capacity * 512] };
            }
            catch { Dispose(); throw; }
        }
        public void Dispose() { foreach (var cache in Caches) cache?.Dispose(); }
    }
}
