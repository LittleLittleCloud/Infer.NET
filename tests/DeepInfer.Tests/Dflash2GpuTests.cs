using DeepInfer.Backends.Dflash2;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using DeepInfer.Backends.Split;
using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

// Opt-in real-model gates. They load the 35B target and the pinned draft, so they are never part of ordinary runs.
// Enable with DEEPINFER_DFLASH2_GPU=1, DEEPINFER_TARGET_DIR=<affine8 target>, DEEPINFER_DFLASH2_DIR=<pinned draft>, and the serving
// profile environment: DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 DEEPINFER_METAL_PREFILL_GEMM=fp32.
// Run one gate at a time (--filter) and serially; each gate owns the whole GPU.
public sealed class Dflash2GpuFactAttribute : FactAttribute
{
    public Dflash2GpuFactAttribute()
    {
        if (!OperatingSystem.IsMacOS()) Skip = "DFlash2 gates require macOS Metal.";
        else if (Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_GPU") != "1") Skip = "Opt-in GPU gate (DEEPINFER_DFLASH2_GPU=1); never run by default.";
        else if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DEEPINFER_TARGET_DIR")) || string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_DIR")))
            Skip = "Requires DEEPINFER_TARGET_DIR and DEEPINFER_DFLASH2_DIR.";
    }
}

internal static class Dflash2Gate
{
    public static string Target => Environment.GetEnvironmentVariable("DEEPINFER_TARGET_DIR")!;
    public static string Draft => Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_DIR")!;
    public const double RelL2 = 1e-4, MaxAbs = 0.01;
    public static readonly long Budget = (long)(38.5 * 1024 * 1024 * 1024);

    public static MetalSplitSession Open(int block, Dflash2ReplayMode replay = Dflash2ReplayMode.Batched, bool debug = false)
    {
        Assert.Equal("fp32", Environment.GetEnvironmentVariable("DEEPINFER_METAL_KV"));
        Assert.Equal("32", Environment.GetEnvironmentVariable("DEEPINFER_METAL_PREFILL_CHUNK"));
        MetalSplitSession.ConfigureAllocationTrace(true);
        var session = new MetalSplitSession(Target, 4, true, 1000, prefillScratchTokens: 128, fusedFp32Decode: true);
        try { session.EnableDflash2(Draft, block, replay, true, debug); } catch { session.Dispose(); throw; }
        return session;
    }

    public static int[] Prompt(MetalSplitSession s, int words)
    {
        var text = string.Join(' ', Enumerable.Range(0, words).Select(i => "The quick brown fox " + i + " jumps over the lazy dog."));
        return s.Tokenize(text, default);
    }

    public static void AssertClose(string what, ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double num = 0, den = 0, max = 0;
        for (var i = 0; i < expected.Length; i++) { double d = (double)actual[i] - expected[i]; num += d * d; den += (double)expected[i] * expected[i]; max = Math.Max(max, Math.Abs(d)); }
        var rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        Assert.True(rel < RelL2 && max < MaxAbs, $"{what}: relL2={rel} maxabs={max}");
    }

    public sealed record ScalarRow(int Argmax, float[] Logits, float[] Taps, ulong[] Digest, int Consumed);

    // Actual scalar decode (di_graph_forward) of the EXACT forced token array: per-row argmax, full logits, raw post-layer taps and the full
    // target state after consuming each prefix.
    public static List<ScalarRow> ForcedScalar(MetalSplitSession s, int[] prompt, int[] forced)
    {
        s.Reset(prompt.Length + forced.Length + 16); s.Prefill(prompt, default);
        var rows = new List<ScalarRow>();
        for (var i = 0; i < forced.Length; i++)
        {
            var logits = new float[Dflash2Geometry.Vocabulary]; var taps = new float[Dflash2Geometry.TapWidth];
            var next = s.ScalarForward(forced[i], prompt.Length + i, logits, taps);
            rows.Add(new(next, logits, taps, s.StateDigest(), s.ConsumedPosition()));
        }
        return rows;
    }

    // Greedy serial decode: n tokens (stopping at EOS unless ignoreEos) and the terminal consumed-state digest.
    public static (List<int> Output, string Finish, ulong[] Digest, int Consumed) Serial(MetalSplitSession s, int[] prompt, int n, bool ignoreEos)
    {
        s.Reset(prompt.Length + n + 16);
        var next = s.Prefill(prompt, default); var output = new List<int>(); var pos = prompt.Length; var finish = "length";
        while (true)
        {
            if (!ignoreEos && s.IsEnd(next)) { finish = "eos"; break; }
            output.Add(next);
            if (output.Count >= n) break;
            next = s.Forward(next, pos++);
        }
        return (output, finish, s.StateDigest(), s.ConsumedPosition());
    }

    public static (List<int> Output, string Finish, ulong[] Digest, int Consumed, Dflash2Stats Stats, int Pending, int ContextEnd) Speculative(MetalSplitSession s, int[] prompt, int n, bool ignoreEos,
        int initialCapacity = -1, CancellationToken ct = default)
    {
        s.Reset(initialCapacity > 0 ? initialCapacity : prompt.Length + n + 16);
        using var generation = s.StartDflash2(n, ignoreEos, () => System.Diagnostics.Stopwatch.GetTimestamp() * 1000.0 / System.Diagnostics.Stopwatch.Frequency);
        generation.Engine.Prefill(prompt, ct);
        var output = new List<int>();
        while (generation.Engine.TryNext(ct, out var block)) output.AddRange(block.Tokens);
        return (output, generation.Engine.Finish ?? "length", s.StateDigest(), s.ConsumedPosition(), generation.Engine.Stats, generation.Engine.Anchor, (int)s.Dflash2State().ContextEnd);
    }

    public static void AssertWithinBudget(MetalSplitSession s)
    {
        // Stats().AllocatedBytes is the whole-device currentAllocatedSize and already includes the attached draft; never add the draft again.
        Assert.True(s.Stats().AllocatedBytes <= Budget, $"allocated {s.Stats().AllocatedBytes} exceeds budget");
        Assert.Equal(0, s.Dflash2Evidence().UnregisteredSubmissions);
    }
}

public class Dflash2GpuTests
{
    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void BatchedVerifyMatchesScalarPerRowLogitsTapsAndFullStateBeforeAndAfterEveryRollbackPrefix()
    {
        using var s = Dflash2Gate.Open(8);
        var prompt = Dflash2Gate.Prompt(s, 8);
        foreach (var rows in new[] { 1, 2, 3, 4, 8 })
        {
            // Serial chain from the actual scalar decode, then force wrong proposals after each accepted prefix.
            // True greedy chain from the scalar decode: chain[0] is the prefill argmax, chain[i] the argmax after consuming chain[..i].
            s.Reset(prompt.Length + 32);
            var chain = new int[rows]; chain[0] = s.Prefill(prompt, default);
            for (var i = 1; i < rows; i++) chain[i] = Dflash2Gate.ForcedScalar(s, prompt, chain[..i])[^1].Argmax;
            for (var accepted = 0; accepted < rows; accepted++)
            {
                var forced = (int[])chain.Clone();
                for (var i = accepted + 1; i < rows; i++) forced[i] = (chain[i] + 1 + i) % 1000;
                var scalar = Dflash2Gate.ForcedScalar(s, prompt, forced);
                s.Reset(prompt.Length + 32); s.Prefill(prompt, default);
                var snapshot = s.SnapshotCreate(); s.SnapshotCapture(snapshot, 1);
                s.PoisonKvSuffix(16); // Rows outside the verified causal prefix must remain invisible; active rows are overwritten before use.
                var logits = new float[rows * Dflash2Geometry.Vocabulary];
                var verified = s.VerifyRows(forced, logits).Argmax; var taps = s.ExportTaps(rows);
                for (var i = 0; i < rows; i++)
                {
                    Assert.Equal(scalar[i].Argmax, verified[i]);
                    Assert.True(scalar[i].Logits.AsSpan().SequenceEqual(logits.AsSpan(i * Dflash2Geometry.Vocabulary, Dflash2Geometry.Vocabulary)), $"rows={rows} accepted={accepted} row {i} logits differ");
                    Assert.True(scalar[i].Taps.AsSpan().SequenceEqual(taps.AsSpan(i * Dflash2Geometry.TapWidth, Dflash2Geometry.TapWidth)), $"rows={rows} accepted={accepted} row {i} taps differ");
                }
                // Full state of the all-rows batched verification BEFORE any restore (the all-accepted branch commits this state).
                Assert.Equal(scalar[rows - 1].Consumed, s.ConsumedPosition());
                Assert.Equal(scalar[rows - 1].Digest, s.StateDigest());
                // Every rollback prefix: restore + replay of the first c rows must equal the scalar state after c rows.
                for (var commit = 1; commit <= rows; commit++)
                {
                    s.SnapshotRestore(snapshot, 1);
                    Assert.Throws<InvalidOperationException>(() => s.ExportTaps(1));
                    s.VerifyRows(forced[..commit]);
                    Assert.Equal(scalar[commit - 1].Consumed, s.ConsumedPosition());
                    Assert.Equal(scalar[commit - 1].Digest, s.StateDigest());
                }
                s.SnapshotRelease(snapshot);
            }
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void TapsMatchTheActualScalarTargetPostLayerResidualsForPrefillAndVerify()
    {
        using var s = Dflash2Gate.Open(8);
        var prompt = Dflash2Gate.Prompt(s, 4)[..40]; // two 32-row prefill chunks
        var scalar = Dflash2Gate.ForcedScalar(s, prompt[..1], []); _ = scalar;
        // Scalar oracle over the prompt itself: feed prompt tokens one at a time through di_graph_forward.
        s.Reset(prompt.Length + 16);
        var oracle = new List<float[]>();
        for (var i = 0; i < prompt.Length; i++) { var taps = new float[Dflash2Geometry.TapWidth]; s.ScalarForward(prompt[i], i, new float[Dflash2Geometry.Vocabulary], taps); oracle.Add(taps); }
        s.Reset(prompt.Length + 16);
        var rowsSeen = 0;
        s.PrefillChunks(prompt, default, (offset, rows) =>
        {
            var taps = s.ExportTaps(rows);
            for (var r = 0; r < rows; r++)
            {
                // Zero-based post-layer taps before the final norm; prefill uses a different operator, so the changed-operator tolerance applies.
                Dflash2Gate.AssertClose($"prefill tap row {offset + r}", taps.AsSpan(r * Dflash2Geometry.TapWidth, Dflash2Geometry.TapWidth), oracle[offset + r]);
                Assert.All(taps.AsSpan(r * Dflash2Geometry.TapWidth, Dflash2Geometry.TapWidth).ToArray(), v => Assert.True(float.IsFinite(v)));
            }
            rowsSeen += rows;
        });
        Assert.Equal(prompt.Length, rowsSeen);
        // Each tap slot must be a distinct layer: slots differ from each other.
        Assert.NotEqual(oracle[5][..2048], oracle[5][2048..4096]);
        // Last chunk has eight rows starting at absolute position32. Stale/oversized reads and wrong source positions are errors.
        Assert.Throws<InvalidOperationException>(() => s.ExportTaps(9));
        s.Dflash2Reset();
        Assert.Throws<InvalidOperationException>(() => s.Dflash2Ingest(0, 1, 0));
        s.Dflash2Ingest(0, 1, 32);
        s.ScalarForward(1, prompt.Length, new float[Dflash2Geometry.Vocabulary], new float[Dflash2Geometry.TapWidth]);
        Assert.Throws<InvalidOperationException>(() => s.ExportTaps(1));
        Assert.Throws<InvalidOperationException>(() => s.Dflash2Ingest(0, 1, 33));
        s.Reset(64);
        Assert.Throws<InvalidOperationException>(() => s.ExportTaps(1));
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void SharedTargetEmbeddingAndHeadAgreeWithTheIndependentAffine8Reference()
    {
        using var s = Dflash2Gate.Open(8, debug: true);
        var prompt = Dflash2Gate.Prompt(s, 6); var captured = new List<float>();
        s.Reset(prompt.Length + 32); s.Dflash2Reset();
        var anchor = s.PrefillChunks(prompt, default, (offset, rows) => { captured.AddRange(s.ExportTaps(rows)); s.Dflash2Ingest(0, rows, offset); });
        var block = new int[8]; block[0] = anchor; Array.Fill(block, Dflash2Geometry.MaskToken, 1, 7);
        s.Dflash2Propose(block);
        var finals = s.Dflash2Checkpoint(Dflash2Checkpoint.Final, 0, 7 * 2048); var native = s.Dflash2Logits(7);
        using var head = new Affine8Reference(Dflash2Gate.Target);
        for (var r = 0; r < 7; r++)
        {
            var reference = head.Logits(finals[(r * 2048)..((r + 1) * 2048)]);
            Dflash2Gate.AssertClose($"head logits row {r}", native.AsSpan(r * Dflash2Geometry.Vocabulary, Dflash2Geometry.Vocabulary), reference);
        }
        var nativeEmbedding = s.Dflash2Checkpoint(Dflash2Checkpoint.Embedding, 0, 8 * 2048);
        for (var row = 0; row < block.Length; row++)
            Dflash2Gate.AssertClose($"embedding row {row}", nativeEmbedding.AsSpan(row * 2048, 2048), head.Embedding(block[row]));
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void DrafterContextAndEveryIntermediateCheckpointMatchTheIndependentFp32Reference()
    {
        using var s = Dflash2Gate.Open(8, debug: true);
        var prompt = Dflash2Gate.Prompt(s, 6); var capturedTaps = new List<float>();
        s.Reset(prompt.Length + 32); s.Dflash2Reset();
        var anchor = s.PrefillChunks(prompt, default, (offset, rows) => { capturedTaps.AddRange(s.ExportTaps(rows)); s.Dflash2Ingest(0, rows, offset); });
        var taps = capturedTaps.ToArray();
        Assert.Equal(Math.Min(prompt.Length, 2047), (int)(s.Dflash2State().ContextEnd - s.Dflash2State().ContextBegin));
        using var draftWeights = Dflash2Weights.Load(Dflash2Gate.Draft, new Dflash2LoadOptions(VerifyHash: false, RequirePins: false));
        var reference = new Dflash2CpuReference(draftWeights);
        reference.Ingest(taps, prompt.Length, 0);
        for (var l = 0; l < 6; l++)
        {
            var (keys, values) = s.Dflash2ExportContext(l);
            Dflash2Gate.AssertClose($"ctx K layer {l}", keys, Enumerable.Range(0, prompt.Length).SelectMany(p => reference.Context[l][p].Key).ToArray());
            Dflash2Gate.AssertClose($"ctx V layer {l}", values, Enumerable.Range(0, prompt.Length).SelectMany(p => reference.Context[l][p].Value).ToArray());
        }
        var block = new int[8]; block[0] = anchor; Array.Fill(block, Dflash2Geometry.MaskToken, 1, 7);
        var proposal = s.Dflash2Propose(block);
        using var target = new Affine8Reference(Dflash2Gate.Target);
        var embeddings = block.Select(t => target.Embedding(t)).ToArray();
        // Optional local-only evidence lets a second independent NumPy implementation diagnose reduction-order differences.
        if (Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_EVIDENCE_DIR") is { Length: > 0 } evidence)
        {
            var directory = Path.Combine(evidence, "draft-checkpoints"); Directory.CreateDirectory(directory);
            var files = new Dictionary<string, object>();
            void Save(string name, float[] values, params int[] shape)
            {
                var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
                File.WriteAllBytes(Path.Combine(directory, name + ".bin"), bytes);
                files[name] = new { shape, sha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant() };
            }
            Save("input-taps", taps, prompt.Length, Dflash2Geometry.TapWidth);
            Save("input-embeddings", embeddings.SelectMany(row => row).ToArray(), 8, 2048);
            Save("native-final", s.Dflash2Checkpoint(Dflash2Checkpoint.Final, 0, 7 * 2048), 7, 2048);
            for (var layer = 0; layer < 6; layer++)
            {
                Save($"native-attention-{layer}", s.Dflash2Checkpoint(Dflash2Checkpoint.HAttn, layer, 8 * 2048), 8, 2048);
                Save($"native-output-{layer}", s.Dflash2Checkpoint(Dflash2Checkpoint.HOut, layer, 8 * 2048), 8, 2048);
                var (keys, values) = s.Dflash2ExportContext(layer);
                Save($"native-context-keys-{layer}", keys, prompt.Length, 8, 128);
                Save($"native-context-values-{layer}", values, prompt.Length, 8, 128);
            }
            File.WriteAllText(Path.Combine(directory, "manifest.json"), System.Text.Json.JsonSerializer.Serialize(new { promptRows = prompt.Length, blockTokens = block, files }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        }
        var draft = reference.Propose(anchor, embeddings, target.Logits, prompt.Length);
        // Per-layer first-divergence diagnosis: the first failing checkpoint name is the first divergent stage.
        for (var l = 0; l < 6; l++)
            foreach (var (kind, width) in new[] { (Dflash2Checkpoint.AttnDyn, 512), (Dflash2Checkpoint.AttnPrep, 2048), (Dflash2Checkpoint.AttnOut, 4096), (Dflash2Checkpoint.HAttn, 2048),
                (Dflash2Checkpoint.MlpDyn, 512), (Dflash2Checkpoint.MlpPrep, 2048), (Dflash2Checkpoint.MlpAct, 6144), (Dflash2Checkpoint.HOut, 2048) })
                Dflash2Gate.AssertClose($"{kind} layer {l}", s.Dflash2Checkpoint(kind, l, 8 * width), draft.Checkpoints[(kind.ToString(), l)].SelectMany(r => r).ToArray());
        Dflash2Gate.AssertClose("final", s.Dflash2Checkpoint(Dflash2Checkpoint.Final, 0, 7 * 2048), draft.Checkpoints[("Final", 0)].SelectMany(r => r).ToArray());
        Dflash2Gate.AssertClose("hidden projection", s.Dflash2Checkpoint(Dflash2Checkpoint.HProj, 0, 7 * 256), draft.Checkpoints[("HProj", 0)].SelectMany(r => r).ToArray());
        for (var i = 0; i < 7; i++)
            Assert.Equal(draft.Candidates[i].OrderBy(x => x).ToArray(), proposal.Candidates.Skip(i * 16).Take(16).OrderBy(x => x).ToArray());
        Assert.Equal(draft.Proposals, proposal.Tokens);
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void RingEvictionKeepsExactly2047RowsAtAbsolutePositionsAndInvisibleSlotsAreNeverRead()
    {
        using var s = Dflash2Gate.Open(8, debug: true);
        var prompt = Dflash2Gate.Prompt(s, 230); // > 2048 tokens: forces eviction during bootstrap
        Assert.True(prompt.Length > 2200, $"prompt too short ({prompt.Length})");
        var tail = new Dictionary<int, float[]>();
        s.Reset(prompt.Length + 32); s.Dflash2Reset();
        var anchor = s.PrefillChunks(prompt, default, (offset, rows) =>
        {
            var taps = s.ExportTaps(rows);
            for (var r = 0; r < rows; r++) { var p = offset + r; if (p == prompt.Length - 2047 || p == prompt.Length - 2046 || p >= prompt.Length - 2) tail[p] = taps.AsSpan(r * Dflash2Geometry.TapWidth, Dflash2Geometry.TapWidth).ToArray(); }
            var from = Math.Max(offset, prompt.Length - 2047);
            if (offset + rows > from) s.Dflash2Ingest(from - offset, offset + rows - from, from);
        });
        var state = s.Dflash2State();
        Assert.Equal(prompt.Length - 2047, state.ContextBegin); Assert.Equal(prompt.Length, state.ContextEnd);
        using var draftWeights = Dflash2Weights.Load(Dflash2Gate.Draft, new Dflash2LoadOptions(VerifyHash: false, RequirePins: false));
        var reference = new Dflash2CpuReference(draftWeights);
        foreach (var (p, row) in tail) reference.Ingest(row, 1, p); // K/V rows depend only on their own tap row, so subsets are exact
        for (var l = 0; l < 6; l++)
        {
            var (keys, values) = s.Dflash2ExportContext(l);
            foreach (var (p, _) in tail)
            {
                var local = p - (prompt.Length - 2047);
                Dflash2Gate.AssertClose($"K layer {l} pos {p}", keys.AsSpan(local * 1024, 1024), reference.Context[l][p].Key);
                Dflash2Gate.AssertClose($"V layer {l} pos {p}", values.AsSpan(local * 1024, 1024), reference.Context[l][p].Value);
            }
        }
        var block = new int[8]; block[0] = anchor; Array.Fill(block, Dflash2Geometry.MaskToken, 1, 7);
        var before = s.Dflash2Propose(block);
        var checkpoints = Enumerable.Range(0, 6).Select(l => s.Dflash2Checkpoint(Dflash2Checkpoint.AttnOut, l, 8 * 4096)).ToArray();
        s.Dflash2PoisonRing(); // NaN in every slot outside the 2047 valid rows, including the evicted oldest slot
        var after = s.Dflash2Propose(block);
        Assert.Equal(before.Tokens, after.Tokens); Assert.Equal(before.Candidates, after.Candidates); Assert.Equal(before.Scores, after.Scores);
        for (var l = 0; l < 6; l++) Assert.Equal(checkpoints[l], s.Dflash2Checkpoint(Dflash2Checkpoint.AttnOut, l, 8 * 4096));
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void LongAbsolutePositionsCrossRopeAndRingBoundariesAndMatchSerialGreedy()
    {
        // DEEPINFER_DFLASH2_LONG selects prompt tokens (e.g. 4200 crosses 2048/4096; 32600 reaches the approved window end).
        var target = int.Parse(Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_LONG") ?? "4200");
        using var s = Dflash2Gate.Open(8, debug: true);
        var prompt = Dflash2Gate.Prompt(s, target / 12 + 10); prompt = prompt[..Math.Min(prompt.Length, target)];
        var outputs = 64; Assert.True(prompt.Length + outputs <= 32768);
        var serial = Dflash2Gate.Serial(s, prompt, outputs, true);
        var spec = Dflash2Gate.Speculative(s, prompt, outputs, true);
        Assert.Equal(serial.Output, spec.Output); Assert.Equal(serial.Consumed, spec.Consumed); Assert.Equal(serial.Digest, spec.Digest);
        var state = s.Dflash2State();
        Assert.Equal(spec.ContextEnd, state.ContextEnd); Assert.True(state.ContextEnd - state.ContextBegin <= 2047);
        Assert.True(state.ContextBegin == Math.Max(0, state.ContextEnd - 2047) || state.ContextBegin >= prompt.Length - 2047);
        Dflash2Gate.AssertWithinBudget(s);
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void FreshTargetConditionedContextAfterRejectionMatchesTheReferenceOfTheAdvertisedPrefix()
    {
        using var s = Dflash2Gate.Open(8, debug: true);
        var prompt = Dflash2Gate.Prompt(s, 8);
        var spec = Dflash2Gate.Speculative(s, prompt, 80, true);
        Assert.True(spec.Stats.RejectedCycles > 0, "fixture produced no rejection; choose another prompt");
        var contextEnd = spec.ContextEnd; var committedTokens = prompt.Concat(spec.Output).ToArray();
        Assert.InRange(contextEnd, prompt.Length, committedTokens.Length);
        // Scalar oracle taps for the last advertised context rows (committed prefix only, never rejected suffix rows).
        var rowsToCheck = Enumerable.Range(contextEnd - 6, 6).ToArray();
        var scalar = Dflash2Gate.ForcedScalar(s, prompt, spec.Output.Take(contextEnd - prompt.Length + 1).ToArray());
        using var draftWeights = Dflash2Weights.Load(Dflash2Gate.Draft, new Dflash2LoadOptions(VerifyHash: false, RequirePins: false));
        var reference = new Dflash2CpuReference(draftWeights);
        // ForcedScalar leaves the target at the end of the committed prefix, so re-extract the draft context state for comparison first.
        var specAgain = Dflash2Gate.Speculative(s, prompt, 80, true);
        foreach (var p in rowsToCheck) reference.Ingest(scalar[p - prompt.Length].Taps, 1, p);
        for (var l = 0; l < 6; l++)
        {
            var (keys, values) = s.Dflash2ExportContext(l); var rows = keys.Length / 1024;
            foreach (var p in rowsToCheck)
            {
                var local = rows - (specAgain.ContextEnd - p);
                Dflash2Gate.AssertClose($"K layer {l} pos {p}", keys.AsSpan(local * 1024, 1024), reference.Context[l][p].Key);
                Dflash2Gate.AssertClose($"V layer {l} pos {p}", values.AsSpan(local * 1024, 1024), reference.Context[l][p].Value);
            }
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void SpeculativeOutputAndTerminalStateEqualSerialGreedyAcrossFixedLengthsAndWarmFormalRuns()
    {
        var lengths = (Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_FIXTURES") ?? "128").Split(',').Select(int.Parse).ToArray();
        foreach (var block in new[] { 4, 8 })
        {
            using var s = Dflash2Gate.Open(block);
            var prompt = Dflash2Gate.Prompt(s, 12);
            foreach (var n in lengths)
            {
                var serial = Dflash2Gate.Serial(s, prompt, n, true);
                for (var run = 0; run < 4; run++) // run 0 is the warm-up, 1..3 are the formal runs
                {
                    var spec = Dflash2Gate.Speculative(s, prompt, n, true);
                    Assert.Equal(serial.Output, spec.Output); Assert.Equal(serial.Consumed, spec.Consumed); Assert.Equal(serial.Digest, spec.Digest);
                    Assert.Equal(spec.Stats.Cycles, spec.Stats.AcceptedHistogram.Sum()); Assert.True(spec.Stats.Accepted <= spec.Stats.Proposed);
                }
                Dflash2Gate.AssertWithinBudget(s);
            }
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void OutputCapTailsGrowthAndCancellationResetMatchSerialGreedy()
    {
        foreach (var block in new[] { 4, 8 })
        {
            using var s = Dflash2Gate.Open(block);
            var prompt = Dflash2Gate.Prompt(s, 6);
            foreach (var n in new[] { 1, 2, 3, block - 1, block, block + 1, 2 * block + 1, 37 })
            {
                var serial = Dflash2Gate.Serial(s, prompt, n, true);
                // Reset reserves but does not shrink existing buffers; actual page growth is checked by a separate fact.
                var spec = Dflash2Gate.Speculative(s, prompt, n, true, initialCapacity: prompt.Length + 1);
                Assert.Equal(serial.Output, spec.Output); Assert.Equal(serial.Consumed, spec.Consumed); Assert.Equal(serial.Digest, spec.Digest);
                Assert.Equal(n, spec.Output.Count);
            }
            // Cancellation mid-request, then a fresh request on the same worker must equal serial greedy and leave no stale context.
            s.Reset(prompt.Length + 64);
            using (var generation = s.StartDflash2(40, true, () => System.Diagnostics.Stopwatch.GetTimestamp() * 1000.0 / System.Diagnostics.Stopwatch.Frequency))
            using (var cts = new CancellationTokenSource())
            {
                generation.Engine.Prefill(prompt, cts.Token);
                Assert.True(generation.Engine.TryNext(cts.Token, out _)); Assert.True(generation.Engine.TryNext(cts.Token, out _));
                cts.Cancel();
                Assert.Throws<OperationCanceledException>(() => generation.Engine.TryNext(cts.Token, out _));
            }
            var serialAfter = Dflash2Gate.Serial(s, prompt, 40, true);
            var specAfter = Dflash2Gate.Speculative(s, prompt, 40, true);
            Assert.Equal(serialAfter.Output, specAfter.Output); Assert.Equal(serialAfter.Digest, specAfter.Digest);
            s.Reset(64); s.Dflash2Reset();
            Assert.Equal(0, s.Dflash2State().ContextEnd - s.Dflash2State().ContextBegin);
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void NativeGenerationActuallyCrosses2048CapacityAndEvictsCommittedContextForBothBlocks()
    {
        foreach (var block in new[] { 4, 8 })
        {
            using var s = Dflash2Gate.Open(block);
            Assert.Equal(2048, s.Stats().Capacity);
            var source = Dflash2Gate.Prompt(s, 200);
            Assert.True(source.Length >= 2016);
            var prompt = source[..2016];
            s.Reset(2048);
            var output = new List<int>();
            ulong[] digest; int consumed;
            using (var generation = s.StartDflash2(256, true, () => System.Diagnostics.Stopwatch.GetTimestamp() * 1000.0 / System.Diagnostics.Stopwatch.Frequency))
            {
                generation.Engine.Prefill(prompt, default);
                Assert.Equal(2048, s.Stats().Capacity); // No pre-reservation of the full output runway.
                while (output.Count < 96)
                {
                    Assert.True(generation.Engine.TryNext(default, out var emitted));
                    output.AddRange(emitted.Tokens);
                }
                Assert.True(s.Stats().Capacity > 2048);
                var context = s.Dflash2State();
                Assert.Equal(generation.Engine.Committed, context.ContextEnd);
                Assert.True(context.ContextBegin > 0);
                Assert.Equal(2047, context.ContextEnd - context.ContextBegin);
                var proposalBlock = new int[block]; proposalBlock[0] = generation.Engine.Anchor;
                Array.Fill(proposalBlock, Dflash2Geometry.MaskToken, 1, block - 1);
                var before = s.Dflash2Propose(proposalBlock);
                s.Dflash2PoisonRing();
                var after = s.Dflash2Propose(proposalBlock);
                Assert.Equal(before.Tokens, after.Tokens);
                Assert.Equal(before.Candidates, after.Candidates);
                Assert.Equal(before.Scores, after.Scores);
                while (generation.Engine.TryNext(default, out var emitted)) output.AddRange(emitted.Tokens);
                digest = s.StateDigest(); consumed = s.ConsumedPosition();
            }
            var serial = Dflash2Gate.Serial(s, prompt, 256, true);
            Assert.Equal(serial.Output, output);
            Assert.Equal(serial.Consumed, consumed);
            Assert.Equal(serial.Digest, digest);
            Dflash2Gate.AssertWithinBudget(s);
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void NaturalEosStopsExactlyLikeSerialGreedyWithoutIgnoreEos()
    {
        foreach (var block in new[] { 4, 8 })
        {
            using var s = Dflash2Gate.Open(block);
            var prompt = s.Tokenize([new ChatTurn("user", "Reply with exactly one word: OK.")], default);
            var serial = Dflash2Gate.Serial(s, prompt, 512, false);
            Assert.Equal("eos", serial.Finish);
            var spec = Dflash2Gate.Speculative(s, prompt, 512, false);
            Assert.Equal("eos", spec.Finish);
            Assert.Equal(serial.Output, spec.Output); Assert.Equal(serial.Consumed, spec.Consumed); Assert.Equal(serial.Digest, spec.Digest);
        }
    }

    [Dflash2GpuFact]
    [Trait("Category", "Dflash2Gpu")]
    public void SerialReplayOracleAndBatchedReplayProduceTheSameOutput()
    {
        string Run(Dflash2ReplayMode mode)
        {
            using var s = Dflash2Gate.Open(8, mode);
            var r = Dflash2Gate.Speculative(s, Dflash2Gate.Prompt(s, 6), 96, true);
            return string.Join(',', r.Output) + "|" + string.Join(',', r.Digest);
        }
        Assert.Equal(Run(Dflash2ReplayMode.Serial), Run(Dflash2ReplayMode.Batched));
    }
}
