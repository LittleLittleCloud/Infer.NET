using System.Text.Json;
using DeepInfer.Backends.Models;
using DeepInfer.Backends.Mtp;
using DeepInfer.Backends.Split;
using Xunit;
namespace DeepInfer.Tests;

[CollectionDefinition("MtpGpuSerial", DisableParallelization = true)]
public sealed class MtpGpuSerialCollection { }

// GPU gates G2/G3. Skipped unless DEEPINFER_MTP_GPU_TESTS=1 (run only by the parent under GPU scheduling), e.g.
//   DEEPINFER_MTP_GPU_TESTS=1 DEEPINFER_MTP_TARGET_DIR=<mlx target> DEEPINFER_MTP_SIDECAR_DIR=<converted sidecar> \
//   DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 DEEPINFER_METAL_PREFILL_GEMM=fp32 DEEPINFER_METAL_PREFILL_MATRIX=0 \
//   DEEPINFER_METAL_PREFILL_DELTA=sequential dotnet test tests/DeepInfer.Tests --filter Category=MtpGpu
public sealed class MtpGpuFactAttribute : FactAttribute
{
    public MtpGpuFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("DEEPINFER_MTP_GPU_TESTS") != "1") Skip = "GPU gate; set DEEPINFER_MTP_GPU_TESTS=1 (parent-scheduled).";
    }
}

internal static class MtpGpuFixture
{
    public static string Target => Environment.GetEnvironmentVariable("DEEPINFER_MTP_TARGET_DIR") ?? throw new InvalidOperationException("DEEPINFER_MTP_TARGET_DIR is required.");
    public static string Sidecar => Environment.GetEnvironmentVariable("DEEPINFER_MTP_SIDECAR_DIR") ?? throw new InvalidOperationException("DEEPINFER_MTP_SIDECAR_DIR is required.");
    public const string Text = "Speculative decoding verifies several drafted tokens against the target model in a single batched pass. " +
        "A drafter proposes the next tokens; the target accepts the longest prefix that matches its own greedy choice. " +
        "Exactness requires the batched pass to reproduce single-token decoding bit for bit, including recurrent state and caches.";
    public static MetalSplitSession Open(bool mtp, bool debug = false, int width = 2, MtpReplayMode replay = MtpReplayMode.Batched)
    {
        var s = new MetalSplitSession(Target, 4, true, 1000, prefillScratchTokens: 128, fusedFp32Decode: true);
        if (mtp) s.EnableMtp(Sidecar, width, replay, debug: debug);
        return s;
    }
    public static int[] Prompt(MetalSplitSession s, int length)
    {
        var ids = s.Tokenize(Text + " " + Text, CancellationToken.None);
        return ids.Length >= length ? ids[..length] : throw new InvalidOperationException("Fixture text too short.");
    }
    public static (int[] Tokens, ulong[] Digest, float[][] Logits) SingleToken(MetalSplitSession s, int[] prompt, int steps)
    {
        s.Reset(prompt.Length + 32);
        var next = s.Prefill(prompt, CancellationToken.None);
        var tokens = new List<int> { next }; var logits = new List<float[]>();
        for (var i = 0; i < steps; i++)
        {
            var l = new float[Qwen36Weights.Vocabulary];
            next = s.Forward(tokens[^1], prompt.Length + i, l); tokens.Add(next); logits.Add(l);
        }
        return ([.. tokens], s.StateDigest(), [.. logits]);
    }
    // Independent CPU reader for the affine8/group64 target checkpoint: dequantizes straight from safetensors bytes, never through the GPU.
    public static (SafeTensors W, SafeTensors.Tensor Weight, float[] Scales, float[] Biases) OpenQuantized(string prefix, int rows)
    {
        using var index = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Target, "model.safetensors.index.json")));
        SafeTensors Open(string name) => new(Path.Combine(Target, index.RootElement.GetProperty("weight_map").GetProperty(name).GetString()!));
        var w = Open(prefix + ".weight"); var weight = w.Get(prefix + ".weight", rows, Qwen36Weights.Dimension / 4);
        using var s = Open(prefix + ".scales"); using var b = Open(prefix + ".biases");
        return (w, weight, s.ReadFloat(s.Get(prefix + ".scales", rows, Qwen36Weights.Dimension / 64)), b.ReadFloat(b.Get(prefix + ".biases", rows, Qwen36Weights.Dimension / 64)));
    }
    public static void DequantRow(SafeTensors w, SafeTensors.Tensor weight, float[] scales, float[] biases, int row, Span<float> destination)
    {
        var packed = new byte[Qwen36Weights.Dimension]; w.ReadBytesRange(weight, (long)row * Qwen36Weights.Dimension, packed);
        for (var c = 0; c < Qwen36Weights.Dimension; c++)
        {
            var g = row * (Qwen36Weights.Dimension / 64) + c / 64;
            destination[c] = packed[c] * scales[g] + biases[g];
        }
    }
    public static double RelativeL2(float[] a, float[] b)
    {
        double d = 0, n = 0; for (var i = 0; i < a.Length; i++) { d += (double)(a[i] - b[i]) * (a[i] - b[i]); n += (double)b[i] * b[i]; }
        return Math.Sqrt(d / Math.Max(n, 1e-30));
    }
    public static double MaxAbs(float[] a, float[] b) { double m = 0; for (var i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i])); return m; }
    public static float[] Floats(byte[] bytes) { var f = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length); return f; }
}

[Collection("MtpGpuSerial")]
[Trait("Category", "MtpGpu")]
public class MtpGpuVerifierTests(Xunit.Abstractions.ITestOutputHelper output)
{
    [MtpGpuFact]
    public void MultiRowVerifyMatchesSingleTokenDecodeForEveryRowCountLogitsAndAllLayerState()
    {
        using var s = MtpGpuFixture.Open(false);
        var prompt = MtpGpuFixture.Prompt(s, 70);
        foreach (var rows in new[] { 1, 2, 3 })
        {
            var (reference, digest, logits) = MtpGpuFixture.SingleToken(s, prompt, rows);
            s.Reset(prompt.Length + 32);
            var anchor = s.Prefill(prompt, CancellationToken.None);
            Assert.Equal(reference[0], anchor);
            var input = reference.Take(rows).ToArray();
            var verifyLogits = new float[rows * Qwen36Weights.Vocabulary];
            var (argmax, hidden) = s.VerifyRows(input, verifyLogits);
            for (var r = 0; r < rows; r++)
            {
                Assert.True(reference[r + 1] == argmax[r], $"rows={rows} row={r}: first divergence argmax {argmax[r]} != {reference[r + 1]}");
                var row = verifyLogits.AsSpan(r * Qwen36Weights.Vocabulary, Qwen36Weights.Vocabulary).ToArray();
                // Strict gates are argmax and the committed state digest; full logits are bounded (declared 1e-4 relL2) and bitwise equality is diagnostic only.
                var rel = MtpGpuFixture.RelativeL2(row, logits[r]);
                output.WriteLine($"rows={rows} row={r} logits relL2={rel:E3} maxabs={MtpGpuFixture.MaxAbs(row, logits[r]):E3} bitwise={logits[r].AsSpan().SequenceEqual(row)}");
                Assert.True(rel < 1e-4, $"rows={rows} row={r}: logits relL2 {rel:E3} exceeds declared 1e-4");
            }
            Assert.Equal(prompt.Length + rows, s.ConsumedPosition());
            Assert.All(hidden, v => Assert.True(float.IsFinite(v)));
            var actualDigest = s.StateDigest();
            output.WriteLine($"rows={rows} mismatched state tensors: {string.Join(',', Enumerable.Range(0, 80).Where(i => digest[i] != actualDigest[i]).Select(i => $"layer{i / 2}/kind{i % 2}"))}");
            Assert.True(digest.AsSpan().SequenceEqual(actualDigest), $"rows={rows}: all-layer state digest differs from single-token decode");
        }
    }

    [MtpGpuFact]
    public void RollbackAtEveryOffsetRestoresAndReplaysTheCommittedPrefixExactly()
    {
        using var s = MtpGpuFixture.Open(false);
        var prompt = MtpGpuFixture.Prompt(s, 70);
        var (reference, _, _) = MtpGpuFixture.SingleToken(s, prompt, 3);
        var snapshot = s.SnapshotCreate();
        try
        {
            for (var accepted = 0; accepted <= 2; accepted++)
            foreach (var serial in new[] { false, true })
            {
                var (_, expectedDigest, _) = MtpGpuFixture.SingleToken(s, prompt, accepted + 1); // anchor plus accepted drafts consumed
                s.Reset(prompt.Length + 32);
                s.Prefill(prompt, CancellationToken.None);
                s.SnapshotCapture(snapshot, 1);
                var wrong = reference.Take(3).ToArray(); wrong[1] = (wrong[1] + 1) % 1000; wrong[2] = (wrong[2] + 3) % 1000;
                for (var i = 1; i <= accepted; i++) wrong[i] = reference[i];
                s.VerifyRows(wrong);
                s.SnapshotRestore(snapshot, 1);
                var committed = reference.Take(accepted + 1).ToArray();
                if (serial) for (var i = 0; i < committed.Length; i++) s.Forward(committed[i], prompt.Length + i);
                else s.VerifyRows(committed);
                Assert.Equal(prompt.Length + accepted + 1, s.ConsumedPosition());
                Assert.True(expectedDigest.AsSpan().SequenceEqual(s.StateDigest()), $"accepted={accepted} serial={serial}: replay state differs");
            }
        }
        finally { s.SnapshotRelease(snapshot); }
    }

    [MtpGpuFact]
    public void PoisonedInvisibleKvSuffixIsNeverRead()
    {
        using var s = MtpGpuFixture.Open(false);
        var prompt = MtpGpuFixture.Prompt(s, 70);
        var (reference, digest, _) = MtpGpuFixture.SingleToken(s, prompt, 2);
        s.Reset(prompt.Length + 32);
        s.Prefill(prompt, CancellationToken.None);
        s.PoisonKvSuffix(16);
        var (argmax, _) = s.VerifyRows(reference.Take(2).ToArray());
        Assert.Equal(reference.Skip(1).Take(2), argmax);
        Assert.True(digest.AsSpan().SequenceEqual(s.StateDigest()));
    }
}

[Collection("MtpGpuSerial")]
[Trait("Category", "MtpGpu")]
public class MtpGpuDrafterTests
{
    [MtpGpuFact]
    public void NativeDrafterMatchesTheIndependentFp32OracleAndGrowsItsKvOnlyThroughTheCursor()
    {
        using var s = MtpGpuFixture.Open(true, debug: true);
        var prompt = MtpGpuFixture.Prompt(s, 40);
        s.Reset(prompt.Length + 32); s.MtpReset();
        var rows = new List<(int[] Tokens, float[] Hidden)>();
        s.PrefillWithHidden(prompt, CancellationToken.None, (t, h) => rows.Add((t, h)));
        var hidden = rows.SelectMany(r => r.Hidden).ToArray();
        using var weights = Qwen36MtpWeights.Load(MtpGpuFixture.Sidecar);
        var oracle = new Qwen36MtpReference(weights);
        const int Count = 12; var worstRel = 0d; var worstAbs = 0d;
        var (embW, embWeight, embScales, embBiases) = MtpGpuFixture.OpenQuantized("language_model.model.embed_tokens", Qwen36Weights.Vocabulary);
        using var embOwner = embW;
        for (var p = 0; p < Count; p++)
        {
            var h = hidden.AsSpan(p * 2048, 2048).ToArray();
            var (_, y, _) = s.MtpRun([prompt[p + 1]], h, p, true, false);
            var embN = MtpGpuFixture.Floats(s.MtpDebugTensor(0, 8192));
            var hidN = MtpGpuFixture.Floats(s.MtpDebugTensor(1, 8192));
            Assert.True(MtpGpuFixture.RelativeL2(hidN, oracle.RmsNorm(h, Qwen36MtpNames.PreHiddenNorm)) < 1e-4);
            // G2 embedding oracle: dequantize the shared target embedding row on the CPU and normalize it independently of the GPU.
            var embRow = new float[2048]; MtpGpuFixture.DequantRow(embW, embWeight, embScales, embBiases, prompt[p + 1], embRow);
            var embRel = MtpGpuFixture.RelativeL2(embN, oracle.RmsNorm(embRow, Qwen36MtpNames.PreEmbeddingNorm));
            Assert.True(embRel < 1e-4, $"row {p}: GPU normalized embedding differs from the independent CPU dequantized oracle (relL2={embRel:E3})");
            var inputNorm = MtpGpuFixture.Floats(s.MtpDebugTensor(3, 8192));
            Assert.All(inputNorm, v => Assert.True(float.IsFinite(v)));
            var yo = oracle.StepNormalized(embN, hidN, p);
            Assert.All(y, v => Assert.True(float.IsFinite(v)));
            var nativeFc = MtpGpuFixture.Floats(s.MtpDebugTensor(2, 8192));
            var after = MtpGpuFixture.Floats(s.MtpDebugTensor(10, 8192));
            var experts = new int[8]; Buffer.BlockCopy(s.MtpDebugTensor(12, 32), 0, experts, 0, 32);
            Assert.Equal(oracle.LastExperts.Order(), experts.Order());
            foreach (var (a, b, name) in new[] { (nativeFc, oracle.LastFc, "fc"), (after, oracle.LastAfterAttention, "afterAttention"), (y, yo, "y") })
            {
                var rel = MtpGpuFixture.RelativeL2(a, b); var abs = MtpGpuFixture.MaxAbs(a, b);
                worstRel = Math.Max(worstRel, rel); worstAbs = Math.Max(worstAbs, abs);
                Assert.True(rel < 1e-4 && abs < 0.01, $"row {p} {name}: relL2={rel:E3} maxabs={abs:E3}");
            }
        }
        var (keys, values) = s.MtpExportKv(Count);
        for (var p = 0; p < Count; p++)
        {
            Assert.True(MtpGpuFixture.RelativeL2(keys[(p * 512)..((p + 1) * 512)], oracle.KeyRow(p)) < 1e-4, $"K row {p}");
            Assert.True(MtpGpuFixture.RelativeL2(values[(p * 512)..((p + 1) * 512)], oracle.ValueRow(p)) < 1e-4, $"V row {p}");
        }
        Assert.Equal(Count, s.MtpState().ValidRows);
        Assert.Throws<InvalidOperationException>(() => s.MtpRun([prompt[0]], hidden[..2048], Count + 3, true, false));
        s.MtpTruncate(4);
        Assert.Equal(4, s.MtpState().ValidRows);
        Assert.True(worstRel < 1e-4 && worstAbs < 0.01);
    }

    [MtpGpuFact]
    public void PoisonedDraftRowsAboveTheCursorAreNeverRead()
    {
        using var s = MtpGpuFixture.Open(true);
        var prompt = MtpGpuFixture.Prompt(s, 40);
        s.Reset(prompt.Length + 32); s.MtpReset();
        var rows = new List<float[]>();
        s.PrefillWithHidden(prompt, CancellationToken.None, (_, h) => rows.Add(h));
        var hidden = rows.SelectMany(r => r).ToArray();
        var clean = s.MtpRun(prompt[1..3], hidden[..(2 * 2048)], 0, true, true);
        s.MtpTruncate(0); s.MtpPoison(2, 16);
        var poisoned = s.MtpRun(prompt[1..3], hidden[..(2 * 2048)], 0, true, true);
        Assert.Equal(clean.Next, poisoned.Next);
        Assert.True(clean.Output.AsSpan().SequenceEqual(poisoned.Output));
    }

    [MtpGpuFact]
    public void SharedHeadLogitsEqualTheIndependentCpuClassifierOfTheDraftHiddenAndTheDraftTokenIsItsArgmax()
    {
        using var s = MtpGpuFixture.Open(true);
        var prompt = MtpGpuFixture.Prompt(s, 40);
        s.Reset(prompt.Length + 32); s.MtpReset();
        var rows = new List<float[]>();
        s.PrefillWithHidden(prompt, CancellationToken.None, (_, h) => rows.Add(h));
        var hidden = rows.SelectMany(r => r).ToArray();
        var (next, y, logits) = s.MtpRun([prompt[1]], hidden[..2048], 0, true, true, fullLogits: true);
        Assert.NotNull(logits);
        var (w, weight, scales, biases) = MtpGpuFixture.OpenQuantized("language_model.lm_head", Qwen36Weights.Vocabulary);
        using var owner = w;
        var cpu = new float[Qwen36Weights.Vocabulary]; var row = new float[2048];
        for (var v = 0; v < cpu.Length; v++)
        {
            MtpGpuFixture.DequantRow(w, weight, scales, biases, v, row);
            double sum = 0; for (var c = 0; c < row.Length; c++) sum += (double)row[c] * y[c];
            cpu[v] = (float)sum;
        }
        Assert.True(MtpGpuFixture.RelativeL2(logits!, cpu) < 1e-4, "shared head logits differ from the independent classifier");
        Assert.Equal(Array.IndexOf(cpu, cpu.Max()), next);
    }
}
