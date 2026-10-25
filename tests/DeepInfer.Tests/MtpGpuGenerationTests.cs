using System.Diagnostics;
using System.Text.Json;
using DeepInfer.Backends.Split;
using Xunit;
namespace DeepInfer.Tests;

[Collection("MtpGpuSerial")]
[Trait("Category", "MtpGpu")]
public class MtpGpuGenerationTests
{
    [MtpGpuFact]
    public void RealGenerationMatchesSerialTargetAndReconditionedDraftCacheForFixedAndNaturalStops()
    {
        var width = int.Parse(Environment.GetEnvironmentVariable("DEEPINFER_MTP_FIXTURE_WIDTH") ?? "1");
        var outputDir = Environment.GetEnvironmentVariable("DEEPINFER_MTP_FIXTURE_OUTPUT") ?? throw new InvalidOperationException("DEEPINFER_MTP_FIXTURE_OUTPUT is required.");
        var fixture = Environment.GetEnvironmentVariable("DEEPINFER_MTP_FIXTURE_INPUT") ?? throw new InvalidOperationException("DEEPINFER_MTP_FIXTURE_INPUT is required.");
        Directory.CreateDirectory(outputDir);
        using var doc = JsonDocument.Parse(File.ReadAllBytes(fixture));
        var fixedIds = doc.RootElement.GetProperty("promptTokenIds").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        using var s = MtpGpuFixture.Open(true, width: width);
        var naturalIds = s.Tokenize("<|im_start|>user\nReply with OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", CancellationToken.None);
        var records = new List<object>();
        foreach (var (label, ids, length, ignore) in new[] { ("fixed128", fixedIds, 128, true), ("fixed4096", fixedIds, 4096, true), ("natural", naturalIds, 128, false) })
        {
            // Independent serial target oracle; no MTP operators participate in its token selection.
            s.Reset(ids.Length + length + 8);
            var expected = new List<int>(); var next = s.Prefill(ids, CancellationToken.None);
            var position = ids.Length; var expectedFinish = "length";
            while (expected.Count < length)
            {
                if (!ignore && s.IsEnd(next)) { expectedFinish = "eos"; break; }
                expected.Add(next);
                if (expected.Count == length) break;
                next = s.Forward(next, position++);
            }
            if (!ignore) Assert.Equal("eos", expectedFinish);
            var expectedDigest = s.StateDigest(); var expectedPosition = s.ConsumedPosition(); var expectedPending = next;
            File.WriteAllText(Path.Combine(outputDir, label + "-serial-tokens.json"), JsonSerializer.Serialize(expected));
            for (var trial = 0; trial < 4; trial++)
            {
                s.Reset(ids.Length + length + 8);
                var hidden = new Dictionary<int, float[]>();
                void Observe(int start, float[] rows) { for (var i = 0; i < rows.Length / 2048; i++) hidden[start + i] = rows.AsSpan(i * 2048, 2048).ToArray(); }
                var timer = Stopwatch.StartNew();
                using var generation = s.StartMtp(length, ignore, () => timer.Elapsed.TotalMilliseconds, Observe);
                var engine = generation.Engine; engine.Prefill(ids, CancellationToken.None);
                var prefillMs = timer.Elapsed.TotalMilliseconds; var actual = new List<int>(); var stamps = new List<double>();
                while (engine.TryNext(CancellationToken.None, out var block))
                {
                    Assert.Equal(expected.Skip(actual.Count).Take(block.Tokens.Length), block.Tokens);
                    actual.AddRange(block.Tokens); for (var i = 0; i < block.Tokens.Length; i++) stamps.Add(block.ReadyMs);
                }
                Assert.Equal(expected, actual); Assert.Equal(expectedFinish, engine.Finish);
                Assert.Equal(expectedPosition, s.ConsumedPosition()); Assert.Equal(expectedPosition, engine.Committed); Assert.Equal(expectedPending, engine.Anchor);
                Assert.Equal(expectedDigest, s.StateDigest());
                // Fresh teacher-forced MTP recomputation from accepted TARGET features is independent
                // of the generation loop's provisional cache. Compare every valid KV row bit for bit.
                var valid = checked((int)s.MtpState().ValidRows); var cached = valid > 0 ? s.MtpExportKv(valid) : default;
                var stream = ids.Concat(expected).ToArray(); s.MtpReset();
                for (var offset = 0; offset < valid; offset += 32)
                {
                    var count = Math.Min(32, valid - offset); var rows = new float[count * 2048];
                    for (var i = 0; i < count; i++) hidden[offset + i].CopyTo(rows, i * 2048);
                    s.MtpRun(stream[(offset + 1)..(offset + 1 + count)], rows, offset, true, false);
                }
                if (valid > 0)
                {
                    var fresh = s.MtpExportKv(valid); Assert.Equal(cached.Keys, fresh.Keys); Assert.Equal(cached.Values, fresh.Values);
                }
                Assert.Equal(expectedDigest, s.StateDigest());
                var row = new { label, width, trial, phase = trial == 0 ? "warm" : "formal", outputTokens = actual.Count, prefillMs,
                    decodeMs = stamps.Count > 1 ? stamps[^1] - stamps[0] : (double?)null,
                    decodeIntervals = Math.Max(0, stamps.Count - 1), engine.Committed, engine.Anchor, engine.Finish, validDraftRows = valid,
                    exactTokens = true, exactTargetState = true, exactFreshDraftKv = true, stats = engine.Stats.ToLogLine(),
                    note = "Diagnostic observer/correctness run; not a serving throughput benchmark." };
                records.Add(row); File.WriteAllText(Path.Combine(outputDir, "summary.json"), JsonSerializer.Serialize(records, new JsonSerializerOptions { WriteIndented = true }));
                File.WriteAllText(Path.Combine(outputDir, $"{label}-{trial}-tokens.json"), JsonSerializer.Serialize(actual));
            }
        }
    }
    [MtpGpuFact]
    public void TargetAndDraftKvGrowDuringGenerationWithoutChangingTokensStateOrCache()
    {
        var width = int.Parse(Environment.GetEnvironmentVariable("DEEPINFER_MTP_FIXTURE_WIDTH") ?? "1");
        var outDir = Environment.GetEnvironmentVariable("DEEPINFER_MTP_FIXTURE_OUTPUT") ?? throw new InvalidOperationException("Fixture output directory required.");
        Directory.CreateDirectory(outDir);
        using var s = MtpGpuFixture.Open(true, width: width);
        var ids = s.Tokenize(string.Concat(Enumerable.Repeat(MtpGpuFixture.Text + " ", 64)), CancellationToken.None)[..2000];
        s.Reset(ids.Length + 32);
        Assert.Equal(2048, s.Stats().Capacity); Assert.Equal(2048, s.MtpState().Capacity);
        var hidden = new Dictionary<int, float[]>();
        void Observe(int start, float[] rows) { for (var i = 0; i < rows.Length / 2048; i++) hidden[start + i] = rows.AsSpan(i * 2048, 2048).ToArray(); }
        var actual = new List<int>(); var timer = Stopwatch.StartNew(); int actualPosition, pending; string? finish;
        using (var generation = s.StartMtp(128, true, () => timer.Elapsed.TotalMilliseconds, Observe))
        {
            generation.Engine.Prefill(ids, CancellationToken.None);
            while (generation.Engine.TryNext(CancellationToken.None, out var block)) actual.AddRange(block.Tokens);
            actualPosition = generation.Engine.Committed; pending = generation.Engine.Anchor; finish = generation.Engine.Finish;
        }
        Assert.Equal(128, actual.Count); Assert.Equal("length", finish);
        Assert.Equal(4096, s.Stats().Capacity); Assert.Equal(4096, s.MtpState().Capacity);
        var actualDigest = s.StateDigest(); var valid = checked((int)s.MtpState().ValidRows); var cache = s.MtpExportKv(valid);
        // Serial oracle uses an already-grown graph, so equality proves the generation-time copy preserved active state.
        s.Reset(ids.Length + 128 + 8);
        var expected = new List<int>(); var next = s.Prefill(ids, CancellationToken.None);
        for (var i = 0; i < 128; i++) { expected.Add(next); if (i < 127) next = s.Forward(next, ids.Length + i); }
        Assert.Equal(expected, actual); Assert.Equal(next, pending);
        Assert.Equal(ids.Length + 127, actualPosition); Assert.Equal(actualPosition, s.ConsumedPosition());
        Assert.Equal(actualDigest, s.StateDigest());
        s.MtpReset(); var stream = ids.Concat(expected).ToArray();
        for (var start = 0; start < valid; start += 32)
        {
            var count = Math.Min(32, valid - start); var features = new float[count * 2048];
            for (var i = 0; i < count; i++) hidden[start + i].CopyTo(features, i * 2048);
            s.MtpRun(stream[(start + 1)..(start + 1 + count)], features, start, true, false);
        }
        var fresh = s.MtpExportKv(valid); Assert.Equal(cache.Keys, fresh.Keys); Assert.Equal(cache.Values, fresh.Values);
        Assert.Equal(actualDigest, s.StateDigest());
        File.WriteAllText(Path.Combine(outDir, "growth.json"), JsonSerializer.Serialize(new { width, promptTokens = ids.Length, outputTokens = actual.Count,
            initialTargetCapacity = 2048, finalTargetCapacity = s.Stats().Capacity, initialDraftCapacity = 2048, finalDraftCapacity = s.MtpState().Capacity,
            exactTokens = true, exactState = true, exactFreshDraftKv = true }, new JsonSerializerOptions { WriteIndented = true }));
    }

}
