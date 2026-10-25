namespace DeepInfer.Backends.Models;

// Independent CPU oracle of one official MTP step (G0/G2). FP32 activations and accumulation over the exact
// BF16 matrix values and FP32 raw+1 norm scales, mirroring the native operator order:
//   e = norm(embed(token)); h = norm(target hidden); x = fc([e, h]); decoder layer = gated full attention
//   (per-head [query256, gate256], Q/K RMSNorm, partial RoPE, FP32 KV) + MoE (top-8, softmax over the 8,
//   shared expert scaled by sigmoid gate); y = final norm. The class owns the draft KV and its valid-row cursor.
public sealed class Qwen36MtpReference
{
    private readonly Qwen36MtpGeometry g;
    private readonly IMtpTensorSource source;
    private readonly List<float[]> keys = [], values = [];
    private readonly Dictionary<string, float[]> norms = new();
    public int ValidRows { get; private set; }
    public int[] LastExperts { get; private set; } = [];
    public float[] LastRouterLogits { get; private set; } = [];
    public float[] LastFc { get; private set; } = [];
    public float[] LastAfterAttention { get; private set; } = [];

    public Qwen36MtpReference(IMtpTensorSource source)
    { this.source = source; g = source.Geometry; foreach (var n in Qwen36MtpNames.Norms) norms[n] = source.ReadNorm(n); }

    public float[] KeyRow(int row) => keys[row];
    public float[] ValueRow(int row) => values[row];
    // Rows at or above the cursor are never read; truncation invalidates them.
    public void Truncate(int rows)
    {
        if (rows < 0 || rows > ValidRows) throw new ArgumentOutOfRangeException(nameof(rows));
        ValidRows = rows;
    }

    public float[] RmsNorm(ReadOnlySpan<float> x, string scaleName)
    {
        var w = norms[scaleName]; var o = new float[x.Length]; float sum = 0;
        foreach (var v in x) sum += v * v;
        var scale = 1f / MathF.Sqrt(sum / x.Length + g.Epsilon);
        for (var i = 0; i < x.Length; i++) o[i] = x[i] * scale * w[i];
        return o;
    }

    private float[] Gemv(string name, ReadOnlySpan<float> x, int rows, int columns, long baseElement = 0)
    {
        var y = new float[rows]; var row = new float[columns];
        for (var r = 0; r < rows; r++)
        {
            source.ReadBf16Range(name, baseElement + (long)r * columns, row);
            float sum = 0;
            for (var c = 0; c < columns; c++) sum += row[c] * x[c];
            y[r] = sum;
        }
        return y;
    }
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    public float[] Step(ReadOnlySpan<float> embedding, ReadOnlySpan<float> hidden, int position) =>
        StepNormalized(RmsNorm(embedding, Qwen36MtpNames.PreEmbeddingNorm), RmsNorm(hidden, Qwen36MtpNames.PreHiddenNorm), position);

    // Same step from already-normalized inputs (used to compare against native debug tensors when raw embeddings are unavailable on the CPU).
    public float[] StepNormalized(ReadOnlySpan<float> embeddingNormalized, ReadOnlySpan<float> hiddenNormalized, int position)
    {
        if (position < 0 || position > ValidRows) throw new InvalidOperationException($"Draft cache has {ValidRows} valid rows; cannot write position {position}.");
        var h = g.Hidden;
        var cat = new float[2 * h];
        embeddingNormalized.CopyTo(cat); hiddenNormalized.CopyTo(cat.AsSpan(h));
        var x = Gemv(Qwen36MtpNames.Fc, cat, h, 2 * h); LastFc = x;
        var n = RmsNorm(x, Qwen36MtpNames.InputNorm);
        var qRaw = Gemv(Qwen36MtpNames.Q, n, g.QProjRows, h);
        var k = Gemv(Qwen36MtpNames.K, n, g.KvRows, h);
        var v = Gemv(Qwen36MtpNames.V, n, g.KvRows, h);
        var d = g.HeadDim;
        var q = new float[g.Heads * d]; var kRoped = new float[g.KvRows];
        for (var head = 0; head < g.Heads; head++) Array.Copy(HeadNorm(qRaw.AsSpan(head * 2 * d, d), Qwen36MtpNames.QNorm, position), 0, q, head * d, d);
        for (var head = 0; head < g.KvHeads; head++) Array.Copy(HeadNorm(k.AsSpan(head * d, d), Qwen36MtpNames.KNorm, position), 0, kRoped, head * d, d);
        if (position == keys.Count) { keys.Add(kRoped); values.Add(v); } else { keys[position] = kRoped; values[position] = v; }
        ValidRows = position + 1;
        var attention = new float[g.Heads * d];
        var group = g.Heads / g.KvHeads;
        for (var head = 0; head < g.Heads; head++)
        {
            var kv = head / group; var scores = new float[position + 1]; float max = float.NegativeInfinity;
            for (var t = 0; t <= position; t++)
            {
                float s = 0;
                for (var i = 0; i < d; i++) s += q[head * d + i] * keys[t][kv * d + i];
                scores[t] = s / MathF.Sqrt(d); max = MathF.Max(max, scores[t]);
            }
            float total = 0;
            for (var t = 0; t <= position; t++) { scores[t] = MathF.Exp(scores[t] - max); total += scores[t]; }
            for (var i = 0; i < d; i++)
            {
                float acc = 0;
                for (var t = 0; t <= position; t++) acc += scores[t] / total * values[t][kv * d + i];
                attention[head * d + i] = acc * Sigmoid(qRaw[head * 2 * d + d + i]);
            }
        }
        var o = Gemv(Qwen36MtpNames.O, attention, h, g.AttentionWidth);
        var r = new float[h]; for (var i = 0; i < h; i++) r[i] = x[i] + o[i];
        LastAfterAttention = r;
        var m = RmsNorm(r, Qwen36MtpNames.PostNorm);
        var router = Gemv(Qwen36MtpNames.Router, m, g.Experts, h); LastRouterLogits = router;
        var ids = TopK(router, g.TopK); LastExperts = ids;
        var probs = new float[g.TopK]; float sum = 0;
        for (var i = 0; i < ids.Length; i++) { probs[i] = MathF.Exp(router[ids[i]] - router[ids[0]]); sum += probs[i]; }
        var ei = g.ExpertIntermediate;
        var y = new float[h];
        for (var slot = 0; slot < ids.Length; slot++)
        {
            var gate = Gemv(Qwen36MtpNames.ExpertGate, m, ei, h, (long)ids[slot] * ei * h);
            var up = Gemv(Qwen36MtpNames.ExpertUp, m, ei, h, (long)ids[slot] * ei * h);
            var act = new float[ei]; for (var i = 0; i < ei; i++) act[i] = gate[i] * Sigmoid(gate[i]) * up[i];
            var down = Gemv(Qwen36MtpNames.ExpertDown, act, h, ei, (long)ids[slot] * h * ei);
            var p = probs[slot] / sum;
            for (var i = 0; i < h; i++) y[i] += down[i] * p;
        }
        var sg = Gemv(Qwen36MtpNames.SharedGate, m, g.SharedIntermediate, h);
        var su = Gemv(Qwen36MtpNames.SharedUp, m, g.SharedIntermediate, h);
        var sa = new float[g.SharedIntermediate]; for (var i = 0; i < sa.Length; i++) sa[i] = sg[i] * Sigmoid(sg[i]) * su[i];
        var sd = Gemv(Qwen36MtpNames.SharedDown, sa, h, g.SharedIntermediate);
        var gateValue = Sigmoid(Gemv(Qwen36MtpNames.SharedRoute, m, 1, h)[0]);
        var outX = new float[h];
        for (var i = 0; i < h; i++) outX[i] = r[i] + y[i] + sd[i] * gateValue;
        return RmsNorm(outX, Qwen36MtpNames.FinalNorm);
    }

    private float[] HeadNorm(ReadOnlySpan<float> input, string scaleName, int position)
    {
        var d = g.HeadDim; var normed = RmsNorm(input, scaleName); var o = (float[])normed.Clone();
        var half = g.RopeDims / 2;
        for (var t = 0; t < g.RopeDims; t++)
        {
            var f = t % half; var partner = t < half ? t + half : t - half;
            var angle = position / MathF.Pow(g.RopeTheta, (float)f / half);
            o[t] = normed[t] * MathF.Cos(angle) + (t < half ? -normed[partner] : normed[partner]) * MathF.Sin(angle);
        }
        return o;
    }

    // Highest logits first; ties choose the lower expert index, like the native routing kernel.
    public static int[] TopK(ReadOnlySpan<float> logits, int count)
    {
        var chosen = new int[count]; var used = new bool[logits.Length];
        for (var n = 0; n < count; n++)
        {
            var best = -1;
            for (var i = 0; i < logits.Length; i++) if (!used[i] && (best < 0 || logits[i] > logits[best])) best = i;
            chosen[n] = best; used[best] = true;
        }
        return chosen;
    }
}
