using System.Diagnostics;
using DeepInfer.Backends.Compute;
using DeepInfer.Core;
namespace DeepInfer.Backends.Models;

public sealed class LlamaModel : IInferenceModel
{
    private readonly IMatVecDevice device;
    private readonly ITransformerWeights weights;
    private readonly IModelTokenizer tokenizer;
    private readonly string backend;
    private readonly SemaphoreSlim gate = new(1, 1);
    private bool disposed;
    public ModelInfo Info { get; }
    public LlamaConfig Config => weights.Config;
    public LlamaModel(ModelInfo info, string checkpointPath, string tokenizerPath, string backend)
    {
        if (info.IsSimulation || info.Format is not ("llama2c-f32" or "hf-qwen2")) throw new NotSupportedException("Unsupported transformer model format.");
        Info = info; this.backend = backend;
        device = backend switch { "cpu" => new CpuMatVecDevice(), "metal" => new MetalMatVecDevice(), _ => throw new NotSupportedException(backend) };
        try
        {
            weights = info.Format == "hf-qwen2" ? new QwenCheckpoint(checkpointPath, device) : new LlamaCheckpoint(checkpointPath, device);
            try { tokenizer = info.Format == "hf-qwen2" ? new QwenTokenizer(tokenizerPath) : new LlamaTokenizer(tokenizerPath, weights.Config.Vocabulary); }
            catch { weights.Dispose(); throw; }
        }
        catch { device.Dispose(); throw; }
    }
    public async ValueTask<ModelOutput> GenerateAsync(InferenceRequest request, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        var error = RequestValidation.Error(request);
        if (error is not null) throw new InferenceInputException(error);
        if (request.Model != Info.Id || request.Backend != backend) throw new InferenceInputException("Model/backend mismatch.");
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
        var started = Stopwatch.StartNew();
        var prompt = tokenizer.EncodePrompt(request.Prompt, ct);
        if (prompt.Length >= Config.Context) throw new InferenceInputException($"Prompt uses {prompt.Length} tokens; this model's context is {Config.Context}. Leave room for generation.");
        var dispatches = device.Dispatches; var gpuMs = device.GpuMilliseconds;
        var state = new State(Config);
        var prefill = Stopwatch.StartNew();
        for (var position = 0; position < prompt.Length; position++) Forward(state, prompt[position], position, ct);
        prefill.Stop();
        double firstTokenMs = 0, lastTokenMs = 0;
        var maximum = Math.Min(request.MaxTokens, Config.Context - prompt.Length);
        var ids = new List<int>(); var pieces = new List<string>();
        var previous = prompt[^1];
        var finish = maximum < request.MaxTokens ? "context_length" : "length";
        for (var i = 0; i < maximum; i++)
        {
            ct.ThrowIfCancellationRequested();
            var next = ArgMax(state.Logits);
            if (tokenizer.IsEnd(next)) { finish = "eos"; break; }
            ids.Add(next); pieces.Add(System.Text.Encoding.UTF8.GetString(tokenizer.DecodePiece(previous, next))); previous = next;
            lastTokenMs = started.Elapsed.TotalMilliseconds;
            if (ids.Count == 1) firstTokenMs = lastTokenMs;
            if (i + 1 < maximum) Forward(state, next, prompt.Length + i, ct);
        }
        var outputIds = ids.ToArray();
        return new(pieces.ToArray(), false, tokenizer.Decode(outputIds, prompt[^1]), outputIds, prompt.Length,
            finish, device.Dispatches - dispatches, device.GpuMilliseconds - gpuMs,
            InferenceTimings.Create(prefill.Elapsed.TotalMilliseconds, firstTokenMs, lastTokenMs - firstTokenMs, outputIds.Length));
    }
    // Useful for numerical acceptance: prefill the same prompt and compare the complete vocabulary logits.
    public float[] EvaluatePrompt(string prompt, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        gate.Wait(ct);
        try
        {
            var ids = tokenizer.EncodePrompt(prompt, ct);
            if (ids.Length > Config.Context) throw new InferenceInputException("Prompt exceeds model context.");
            var state = new State(Config);
            for (var position = 0; position < ids.Length; position++) Forward(state, ids[position], position, ct);
            return state.Logits;
        }
        finally { gate.Release(); }
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
    private void Forward(State s, int token, int position, CancellationToken ct)
    {
        var c = Config; var dim = c.Dimension; var kv = c.KvDimension; var head = c.HeadSize;
        weights.CopyEmbedding(token, s.X);
        for (var layer = 0; layer < c.Layers; layer++)
        {
            ct.ThrowIfCancellationRequested();
            Norm(s.X, weights.AttentionNorm[layer], s.Branch);
            var offset = (layer * c.Context + position) * kv;
            var key = s.Keys.AsSpan(offset, kv); var value = s.Values.AsSpan(offset, kv);
            weights.Query[layer].Multiply(s.Branch, s.Query);
            weights.Key[layer].Multiply(s.Branch, key);
            weights.Value[layer].Multiply(s.Branch, value);
            AddBias(s.Query, weights.QueryBias?[layer]); AddBias(key, weights.KeyBias?[layer]); AddBias(value, weights.ValueBias?[layer]);
            Rotate(s.Query, position, head); Rotate(key, position, head);
            Array.Clear(s.Branch);
            for (var h = 0; h < c.Heads; h++)
            {
                var query = s.Query.AsSpan(h * head, head);
                var scores = s.Scores.AsSpan(0, position + 1);
                var kvHead = h / (c.Heads / c.KvHeads);
                for (var t = 0; t <= position; t++)
                {
                    var past = s.Keys.AsSpan((layer * c.Context + t) * kv + kvHead * head, head);
                    var dot = 0f;
                    for (var d = 0; d < head; d++) dot += query[d] * past[d];
                    scores[t] = dot / MathF.Sqrt(head);
                }
                Softmax(scores);
                for (var t = 0; t <= position; t++)
                {
                    var past = s.Values.AsSpan((layer * c.Context + t) * kv + kvHead * head, head);
                    for (var d = 0; d < head; d++) s.Branch[h * head + d] += scores[t] * past[d];
                }
            }
            weights.AttentionOutput[layer].Multiply(s.Branch, s.Residual);
            for (var d = 0; d < dim; d++) s.X[d] += s.Residual[d];
            Norm(s.X, weights.FfnNorm[layer], s.Branch);
            weights.Gate[layer].Multiply(s.Branch, s.Hidden);
            weights.Up[layer].Multiply(s.Branch, s.HiddenGate);
            for (var d = 0; d < c.HiddenDimension; d++) s.Hidden[d] = s.Hidden[d] / (1f + MathF.Exp(-s.Hidden[d])) * s.HiddenGate[d];
            weights.Down[layer].Multiply(s.Hidden, s.Residual);
            for (var d = 0; d < dim; d++) s.X[d] += s.Residual[d];
        }
        Norm(s.X, weights.FinalNorm, s.Branch);
        weights.Classifier.Multiply(s.Branch, s.Logits);
    }
    private void Norm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, Span<float> output)
    {
        var square = 0f; foreach (var value in input) square += value * value;
        var scale = 1f / MathF.Sqrt(square / input.Length + weights.Epsilon);
        for (var i = 0; i < input.Length; i++) output[i] = weight[i] * (input[i] * scale);
    }
    private static void AddBias(Span<float> values, float[]? bias)
    {
        if (bias is null) return;
        for (var i = 0; i < values.Length; i++) values[i] += bias[i];
    }
    private void Rotate(Span<float> vector, int position, int headSize)
    {
        if (weights.SplitHalfRope)
        {
            var half = headSize / 2;
            for (var h = 0; h < vector.Length; h += headSize)
                for (var d = 0; d < half; d++)
                {
                    var angle = position / MathF.Pow(weights.RopeTheta, 2f * d / headSize);
                    var sin = MathF.Sin(angle); var cos = MathF.Cos(angle);
                    var a = vector[h + d]; var b = vector[h + d + half];
                    vector[h + d] = a * cos - b * sin; vector[h + d + half] = b * cos + a * sin;
                }
            return;
        }
        for (var i = 0; i < vector.Length; i += 2)
        {
            var angle = position / MathF.Pow(weights.RopeTheta, (float)(i % headSize) / headSize);
            var sin = MathF.Sin(angle); var cos = MathF.Cos(angle);
            var a = vector[i]; var b = vector[i + 1];
            vector[i] = a * cos - b * sin; vector[i + 1] = a * sin + b * cos;
        }
    }
    private static void Softmax(Span<float> values)
    {
        var max = float.NegativeInfinity; foreach (var v in values) max = MathF.Max(max, v);
        var sum = 0f;
        for (var i = 0; i < values.Length; i++) { values[i] = MathF.Exp(values[i] - max); sum += values[i]; }
        for (var i = 0; i < values.Length; i++) values[i] /= sum;
    }
    public async ValueTask DisposeAsync()
    {
        await gate.WaitAsync();
        try { if (!disposed) { disposed = true; weights.Dispose(); device.Dispose(); } }
        finally { gate.Release(); }
    }
    private sealed class State(LlamaConfig c)
    {
        public readonly float[] X = new float[c.Dimension], Branch = new float[c.Dimension], Residual = new float[c.Dimension], Query = new float[c.Dimension];
        public readonly float[] Hidden = new float[c.HiddenDimension], HiddenGate = new float[c.HiddenDimension];
        public readonly float[] Keys = new float[checked(c.Layers * c.Context * c.KvDimension)], Values = new float[checked(c.Layers * c.Context * c.KvDimension)];
        public readonly float[] Scores = new float[c.Context], Logits = new float[c.Vocabulary];
    }
}
