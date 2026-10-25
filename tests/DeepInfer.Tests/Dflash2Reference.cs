using System.Text.Json;
using DeepInfer.Backends.Models;
namespace DeepInfer.Tests;

// Independent FP32 CPU implementation of the DFlash2 draft equations, written from the pinned upstream Python
// (artifacts/dual-host/issue80-dflash2/dflash-model_mlx.py). It shares no code with the native Metal path and only reads
// the same BF16 draft tensors and the affine8 target embedding/head. Norm scales are direct BF16 values. Arithmetic is
// FP32 end to end (BF16 weights are exact in FP32; activations/accumulators are float, sequential summation), so it is the
// same-arithmetic oracle up to reduction order; tolerances stay relL2<1e-4 / maxabs<0.01. RoPE table convention: angles in
// double, cos/sin rounded once to FP32 (identical to the native table). Full 2047-row windows are impractical here; use the
// parent NumPy oracle with exported taps for those.
internal sealed class Dflash2CpuReference(Dflash2Weights weights)
{
    public const int H = 2048, Heads = 32, KvHeads = 8, Hd = 128, Window = 2048, K = 2, GroupSize = 16, Groups = 128, TopK = 16;
    public readonly SortedDictionary<long, (float[] Key, float[] Value)>[] Context = Enumerable.Range(0, 6).Select(_ => new SortedDictionary<long, (float[], float[])>()).ToArray();

    public static float[] Rms(ReadOnlySpan<float> x, float[] scale, int offset = 0, int count = -1)
    {
        count = count < 0 ? x.Length : count;
        float s = 0; for (var i = 0; i < count; i++) s += x[offset + i] * x[offset + i];
        var inv = 1f / MathF.Sqrt(s / count + 1e-6f);
        var r = new float[count]; for (var i = 0; i < count; i++) r[i] = x[offset + i] * inv * scale[i];
        return r;
    }

    private readonly Dictionary<string, float[]> matrices = [];
    public float[] Matrix(string name)
    {
        if (matrices.TryGetValue(name, out var cached)) return cached;
        var n = (int)(weights.TensorBytes(name) / 2); var m = new float[n]; weights.ReadBf16Range(name, 0, m);
        if (matrices.Count > 8) matrices.Clear(); // bound memory; fc is 134 MB
        return matrices[name] = m;
    }

    // y[r] = W x[r], W row-major [outDim, inDim].
    public static float[][] MatMul(float[] w, int outDim, int inDim, float[][] x)
    {
        var y = new float[x.Length][];
        for (var r = 0; r < x.Length; r++)
        {
            y[r] = new float[outDim];
            Parallel.For(0, outDim, o => { float s = 0; var b = (long)o * inDim; for (var i = 0; i < inDim; i++) s += w[b + i] * x[r][i]; y[r][o] = s; });
        }
        return y;
    }

    public static void Rope(float[] v, int offset, long position)
    {
        for (var i = 0; i < Hd / 2; i++)
        {
            var angle = position * Math.Pow(1e7, -2.0 * i / Hd);
            var (sd, cd) = Math.SinCos(angle); var sin = (float)sd; var cos = (float)cd;
            var a = v[offset + i]; var b = v[offset + i + Hd / 2];
            v[offset + i] = a * cos - b * sin; v[offset + i + Hd / 2] = a * sin + b * cos;
        }
    }

    public static float Silu(float x) => x / (1 + MathF.Exp(-x));

    // Causal grouped dynamic convolution of one block. dyn[t] is [which, K, Groups] flattened (which*K*Groups + k*Groups + g).
    public static float[][] DynamicConv(float[][] hidden, float[][] dyn, int which, float[] baseKernel)
    {
        var length = hidden.Length; var output = new float[length][];
        for (var t = 0; t < length; t++)
        {
            output[t] = new float[H];
            for (var offset = 0; offset < K; offset++)
            {
                if (t < offset) continue;
                for (var j = 0; j < H; j++)
                {
                    var g = j / GroupSize;
                    var kernel = baseKernel[(which * K + offset) * H + j];
                    // Upstream accumulates the base and dynamic contributions separately in FP32.
                    output[t][j] += kernel * hidden[t - offset][j];
                    output[t][j] += dyn[t][which * K * Groups + offset * Groups + g] * hidden[t - offset][j];
                }
            }
        }
        return output;
    }

    public (float[][] CtxRaw, float[][] CtxNorm) Ingest(float[] taps, int rows, long firstPosition)
    {
        var x = new float[rows][]; for (var r = 0; r < rows; r++) { x[r] = new float[16384]; for (var i = 0; i < 16384; i++) x[r][i] = taps[r * 16384 + i]; }
        var raw = MatMul(Matrix(Dflash2Names.Fc), H, 16384, x);
        var hn = weights.ReadVector(Dflash2Names.HiddenNorm);
        var norm = raw.Select(v => Rms(v, hn)).ToArray();
        var kNorm = Enumerable.Range(0, 6).Select(l => weights.ReadVector(Dflash2Names.Layer(l, "self_attn.k_norm.weight"))).ToArray();
        for (var l = 0; l < 6; l++)
        {
            var k = MatMul(Matrix(Dflash2Names.Layer(l, "self_attn.k_proj.weight")), 1024, H, norm);
            var v = MatMul(Matrix(Dflash2Names.Layer(l, "self_attn.v_proj.weight")), 1024, H, norm);
            for (var r = 0; r < rows; r++)
            {
                var key = new float[1024];
                for (var h = 0; h < KvHeads; h++) { Rms(k[r], kNorm[l], h * Hd, Hd).CopyTo(key, h * Hd); Rope(key, h * Hd, firstPosition + r); }
                Context[l][firstPosition + r] = (key.Select(d => (float)d).ToArray(), v[r].Select(d => (float)d).ToArray());
            }
        }
        return (raw, norm);
    }

    public sealed record Draft(Dictionary<(string Kind, int Layer), float[][]> Checkpoints, int[] Proposals, int[][] Candidates, float[][] Scores);

    // embeddings: [B][H] rows of the shared target embedding; headLogits maps a normalized hidden row to full-vocabulary logits.
    public Draft Propose(int anchor, float[][] embeddings, Func<float[], float[]> headLogits, long contextEnd)
    {
        var block = embeddings.Length; var cp = new Dictionary<(string, int), float[][]>();
        var h = embeddings.Select(e => (float[])e.Clone()).ToArray();
        for (var l = 0; l < 6; l++)
        {
            string N(string tail) => Dflash2Names.Layer(l, tail);
            var inputNorm = weights.ReadVector(N("input_layernorm.weight")); var postNorm = weights.ReadVector(N("post_attention_layernorm.weight"));
            var qn = weights.ReadVector(N("self_attn.q_norm.weight")); var kn = weights.ReadVector(N("self_attn.k_norm.weight"));
            var attnBase = weights.ReadVector(N("attention_conv.base_kernel")); var mlpBase = weights.ReadVector(N("mlp_conv.base_kernel"));
            var norm = h.Select(v => Rms(v, inputNorm)).ToArray();
            var dyn = MatMul(Matrix(N("attention_conv.kernel_projection.weight")), 512, H, norm); cp[("AttnDyn", l)] = dyn;
            var prep = DynamicConv(norm, dyn, 0, attnBase); cp[("AttnPrep", l)] = prep;
            var q = MatMul(Matrix(N("self_attn.q_proj.weight")), 4096, H, prep);
            var k = MatMul(Matrix(N("self_attn.k_proj.weight")), 1024, H, prep);
            var v = MatMul(Matrix(N("self_attn.v_proj.weight")), 1024, H, prep);
            for (var i = 0; i < block; i++)
            {
                for (var hd = 0; hd < Heads; hd++) { Rms(q[i], qn, hd * Hd, Hd).CopyTo(q[i], hd * Hd); Rope(q[i], hd * Hd, contextEnd + i); }
                for (var hd = 0; hd < KvHeads; hd++) { Rms(k[i], kn, hd * Hd, Hd).CopyTo(k[i], hd * Hd); Rope(k[i], hd * Hd, contextEnd + i); }
            }
            var attn = new float[block][];
            for (var i = 0; i < block; i++)
            {
                attn[i] = new float[4096]; var qpos = contextEnd + i;
                var visible = Context[l].Where(p => qpos - p.Key < Window).ToList();
                for (var hd = 0; hd < Heads; hd++)
                {
                    var kv = hd / (Heads / KvHeads); var scores = new List<float>(); var values = new List<Func<int, float>>();
                    foreach (var (_, (ck, cv)) in visible)
                    {
                        float s = 0; for (var d = 0; d < Hd; d++) s += q[i][hd * Hd + d] * ck[kv * Hd + d];
                        scores.Add(s / MathF.Sqrt(Hd)); var local = cv; values.Add(d => local[kv * Hd + d]);
                    }
                    // Proposed keys are bidirectional inside the block.
                    for (var j = 0; j < block; j++)
                    {
                        float s = 0; for (var d = 0; d < Hd; d++) s += q[i][hd * Hd + d] * k[j][kv * Hd + d];
                        scores.Add(s / MathF.Sqrt(Hd)); var jj = j; values.Add(d => v[jj][kv * Hd + d]);
                    }
                    var max = scores.Max(); var exps = scores.Select(s => MathF.Exp(s - max)).ToArray(); var sum = exps.Sum();
                    for (var d = 0; d < Hd; d++) { float acc = 0; for (var s = 0; s < exps.Length; s++) acc += exps[s] / sum * values[s](d); attn[i][hd * Hd + d] = acc; }
                }
            }
            cp[("AttnOut", l)] = attn;
            var o = MatMul(Matrix(N("self_attn.o_proj.weight")), H, 4096, attn);
            var oc = DynamicConv(o, dyn, 1, attnBase);
            for (var i = 0; i < block; i++) for (var j = 0; j < H; j++) h[i][j] += oc[i][j];
            cp[("HAttn", l)] = h.Select(r => (float[])r.Clone()).ToArray();
            norm = h.Select(r => Rms(r, postNorm)).ToArray();
            dyn = MatMul(Matrix(N("mlp_conv.kernel_projection.weight")), 512, H, norm); cp[("MlpDyn", l)] = dyn;
            prep = DynamicConv(norm, dyn, 0, mlpBase); cp[("MlpPrep", l)] = prep;
            var gate = MatMul(Matrix(N("mlp.gate_proj.weight")), 6144, H, prep); var up = MatMul(Matrix(N("mlp.up_proj.weight")), 6144, H, prep);
            var act = new float[block][];
            for (var i = 0; i < block; i++) { act[i] = new float[6144]; for (var j = 0; j < 6144; j++) act[i][j] = Silu(gate[i][j]) * up[i][j]; }
            cp[("MlpAct", l)] = act;
            var down = MatMul(Matrix(N("mlp.down_proj.weight")), H, 6144, act);
            var dc = DynamicConv(down, dyn, 1, mlpBase);
            for (var i = 0; i < block; i++) for (var j = 0; j < H; j++) h[i][j] += dc[i][j];
            cp[("HOut", l)] = h.Select(r => (float[])r.Clone()).ToArray();
        }
        var finalNorm = weights.ReadVector(Dflash2Names.FinalNorm);
        var final = h.Skip(1).Select(r => Rms(r, finalNorm)).ToArray(); cp[("Final", 0)] = final;
        var hproj = MatMul(Matrix(Dflash2Names.HiddenProjection), 256, H, final); cp[("HProj", 0)] = hproj;
        var pred = new float[Dflash2Geometry.Vocabulary * 256]; var succ = new float[pred.Length];
        weights.ReadBf16Range(Dflash2Names.PredecessorCodebook, 0, pred); weights.ReadBf16Range(Dflash2Names.SuccessorCodebook, 0, succ);
        var proposals = new int[block - 1]; var candidates = new int[block - 1][]; var allScores = new float[block - 1][];
        var predecessor = anchor;
        for (var i = 0; i < block - 1; i++)
        {
            var logits = headLogits(final[i]);
            var top = Enumerable.Range(0, logits.Length).OrderByDescending(t => logits[t]).ThenBy(t => t).Take(TopK).ToArray();
            candidates[i] = top; allScores[i] = new float[TopK];
            for (var c = 0; c < TopK; c++)
            {
                float edge = 0; for (var r = 0; r < 256; r++) edge += (float)pred[(long)predecessor * 256 + r] * hproj[i][r] * succ[(long)top[c] * 256 + r];
                allScores[i][c] = logits[top[c]] + edge;
            }
            var best = 0; for (var c = 1; c < TopK; c++) if (allScores[i][c] > allScores[i][best]) best = c;
            proposals[i] = top[best]; predecessor = top[best];
        }
        return new(cp, proposals, candidates, allScores);
    }

    public static (double RelL2, double MaxAbs) Compare(ReadOnlySpan<float> actual, float[][] expected)
    {
        double num = 0, den = 0, max = 0; var i = 0;
        foreach (var row in expected) foreach (var e in row) { double d = (double)actual[i++] - e; num += d * d; den += (double)e * e; max = Math.Max(max, Math.Abs(d)); }
        return (Math.Sqrt(num / Math.Max(den, 1e-30)), max);
    }
}

// Affine8 (group 64) CPU reader for the target's shared embedding and vocabulary head.
internal sealed class Affine8Reference : IDisposable
{
    private readonly Dictionary<string, SafeTensors> files = [];
    private readonly Dictionary<string, string> map = [];
    private readonly string directory;
    public Affine8Reference(string targetDirectory)
    {
        directory = targetDirectory;
        using var doc = JsonDocument.Parse(System.IO.File.ReadAllText(Path.Combine(directory, "model.safetensors.index.json")));
        foreach (var p in doc.RootElement.GetProperty("weight_map").EnumerateObject()) map[p.Name] = p.Value.GetString()!;
    }
    private SafeTensors Open(string tensor)
    {
        var name = map[tensor];
        if (!files.TryGetValue(name, out var f)) files[name] = f = new SafeTensors(Path.Combine(directory, name));
        return f;
    }
    // Dequantized row range [first, first+count) of an affine8 [rows, 2048] matrix.
    public float[] Rows(string prefix, int first, int count)
    {
        const int cols = 2048;
        var f = Open(prefix + ".weight"); var w = f.Find(prefix + ".weight");
        var sf = Open(prefix + ".scales"); var s = sf.Find(prefix + ".scales");
        var bf = Open(prefix + ".biases"); var b = bf.Find(prefix + ".biases");
        var q = new byte[count * cols]; var scale = new float[count * cols / 64]; var bias = new float[count * cols / 64];
        f.ReadBytesRange(w, (long)first * cols, q);
        sf.ReadBFloat16Range(s, (long)first * cols / 64, scale); bf.ReadBFloat16Range(b, (long)first * cols / 64, bias);
        var r = new float[count * cols];
        for (var i = 0; i < r.Length; i++) r[i] = q[i] * scale[i / 64] + bias[i / 64];
        return r;
    }
    public float[] Embedding(int token) => Rows("language_model.model.embed_tokens", token, 1).Select(v => (float)v).ToArray();
    public float[] Logits(float[] hidden)
    {
        var logits = new float[Dflash2Geometry.Vocabulary]; const int chunk = 2048;
        for (var first = 0; first < logits.Length; first += chunk)
        {
            var n = Math.Min(chunk, logits.Length - first); var rows = Rows("language_model.lm_head", first, n);
            Parallel.For(0, n, r => { float s = 0; for (var i = 0; i < 2048; i++) s += rows[r * 2048 + i] * hidden[i]; logits[first + r] = s; });
        }
        return logits;
    }
    public void Dispose() { foreach (var f in files.Values) f.Dispose(); }
}
