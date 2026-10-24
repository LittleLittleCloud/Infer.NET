using System.Runtime.InteropServices;
using System.Text.Json;
using DeepInfer.Backends.Speculative;
using DeepInfer.Backends.Split;
using DeepInfer.Core;
namespace DeepInfer.Host;

// Single-load correctness harness for the synthetic speculative-state reference (Issue #72, S0).
// For every saved prompt it generates one non-speculative baseline (recording full-state digests and exact
// logits at checkpoints), then replays each pattern/block/output-length through the real SpeculativeEngine on the
// real Metal graph and compares tokens, per-layer state digests and logits. It writes evidence, not conclusions:
// timings are diagnostic phase durations and are never decode throughput.
internal static class SpeculativeCheck
{
    // Evidence captured at the first divergence while the failing target is still alive.
    private sealed class Divergence(string message) : Exception(message)
    {
        public string Kind { get; init; } = "divergence"; public string Where { get; init; } = "";
        public int Emitted { get; init; } public int[]? Tensors { get; init; }
        public ulong[]? ExpectedDigest { get; init; } public ulong[]? ActualDigest { get; init; }
        public float[]? ExpectedLogits { get; init; } public float[]? ActualLogits { get; init; }
        public Dictionary<int, byte[]>? ActualState { get; init; }
    }
    private sealed record Checkpoint(ulong[] Digest, float[] Logits);
    private sealed record Case(string Name, int[] Prompt, int[]? Saved);
    // The baseline always decodes past the end token (EosAt is its first occurrence, or -1) so one stream serves both
    // fixed-length and natural-end entries; a natural stop is exactly the prefix before EosAt.
    private sealed record Reference(int[] Tokens, SortedDictionary<int, Checkpoint> Checkpoints, int EosAt);
    private sealed record Expectation(int[] Tokens, string Finish, int Terminal);
    private sealed record RunRow(string Case, MatrixEntry Entry, string Phase, int Index, string Status, string? Failure,
        SpeculativeDiagnostics? Diagnostics, int Checkpoints, int TerminalPosition, int TerminalIndex);

    private static int[] Ints(string text, int min, int max, string name) =>
        text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(x =>
            int.TryParse(x, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var v) && v >= min && v <= max ? v
            : throw new ArgumentException($"--{name} entries must be integers in {min}..{max}.")).ToArray();

    private static int[] TokenArray(JsonElement e, string name)
    {
        if (e.ValueKind != JsonValueKind.Array) throw new ArgumentException($"{name} must be an array of token ids.");
        var values = e.EnumerateArray().Select(x => x.ValueKind == JsonValueKind.Number && x.TryGetInt32(out var v) && v is >= 0 and < 248320 ? v
            : throw new ArgumentException($"{name} contains an invalid token id.")).ToArray();
        return values;
    }
    // Accepts a bare token-id array, or {"cases":[{"name":..,"promptTokenIds":[..],"outputTokenIds":[..]}]}.
    internal static (string Name, int[] Prompt, int[]? Saved)[] ReadCases(string text)
    {
        using var document = JsonDocument.Parse(text);
        var root = document.RootElement;
        if (root.ValueKind == JsonValueKind.Array) return [("case-0", Prompt(TokenArray(root, "prompt")), null)];
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("cases", out var cases) || cases.ValueKind != JsonValueKind.Array)
            throw new ArgumentException("--token-ids must be a token-id array or an object with a cases array.");
        return cases.EnumerateArray().Select((c, i) => (
            c.TryGetProperty("name", out var n) && n.ValueKind == JsonValueKind.String ? n.GetString()! : $"case-{i}",
            Prompt(TokenArray(c.TryGetProperty("promptTokenIds", out var p) ? p : throw new ArgumentException("A case needs promptTokenIds."), "promptTokenIds")),
            c.TryGetProperty("outputTokenIds", out var o) ? TokenArray(o, "outputTokenIds") : null)).ToArray();
    }
    private static int[] Prompt(int[] ids) => ids.Length is >= 1 and <= 200000 ? ids : throw new ArgumentException("A prompt needs 1..200000 token ids.");

    public static async Task Run(Options o, CancellationToken ct)
    {
        if (!OperatingSystem.IsMacOS()) throw new ArgumentException("speculative-check requires the Mac metal-local build.");
        var warm = o.Number("warm-runs", 1, 0, 8); var formal = o.Number("formal-runs", 3, 1, 16);
        var interval = o.Number("checkpoint-tokens", 64, 16, 4096);
        var ignoreEos = o.Flag("ignore-eos"); var poison = o.Flag("poison-suffix");
        // Any explicit matrix option selects a custom cross product; otherwise the practical default matrix runs.
        var custom = o.Flag("output-lengths") || o.Flag("patterns") || o.Flag("blocks");
        if (!custom && ignoreEos) throw new ArgumentException("--ignore-eos applies only to a custom matrix; the default matrix fixes it per group.");
        IReadOnlyList<MatrixEntry> entries;
        if (custom)
        {
            var lengths = Ints(o.Get("output-lengths", "128"), 1, Protocol.MaxTokens, "output-lengths").Distinct().Order().ToArray();
            var patterns = o.Get("patterns", "accept-all,reject-all,offset:1,offset:3,mixed").Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(SyntheticPattern.Parse).ToArray();
            var blocks = Ints(o.Get("blocks", "1,2,4,8"), 1, SyntheticPattern.MaxBlockTokens, "blocks").Distinct().ToArray();
            if (patterns.Length == 0 || lengths.Length == 0 || blocks.Length == 0) throw new ArgumentException("patterns, output-lengths and blocks must not be empty.");
            entries = SpeculativeMatrix.Cross(lengths, patterns, blocks, ignoreEos);
        }
        else entries = SpeculativeMatrix.Default();
        if (entries.Count == 0) throw new ArgumentException("The selected matrix is empty.");
        var idsFile = o.Get("token-ids", "");
        if (idsFile.Length == 0) throw new ArgumentException("--token-ids FILE is required.");
        // Same native path as the deployed baseline unless explicitly overridden: fused FP32 decode attention, scratch 128, chunk 32.
        var deployed = DeployedProfile.Parse(o.Get("decode-attention", "fused-fp32"), o.Number("metal-scratch-tokens", 128, 128, 512));
        var cases = ReadCases(await File.ReadAllTextAsync(idsFile, ct))
            .Select(c => new Case(c.Name, c.Prompt, c.Saved)).ToArray();
        var directory = o.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx");
        var output = Path.GetFullPath(o.Get("output", "artifacts/speculative-check/run-" + DateTime.UtcNow.ToString("yyyyMMddTHHmmss")));
        Directory.CreateDirectory(output);
        SplitService.Profile(o); // Same FP32 deployment profile as the service (sets managed prefill chunk 32); rejects unsupported options.
        await PinnedSplitModel.Verify(directory, ct);
        Console.Error.WriteLine("Loading full resident Metal model once...");
        using var gpu = new MetalSplitSession(directory, 4, true, 1000, prefillScratchTokens: deployed.ScratchTokens, fusedFp32Decode: deployed.FusedFp32Decode);
        var profile = gpu.ExecutionProfile();
        deployed.Verify(profile, gpu.FusedFp32Decode, gpu.PrefillChunkTokens);
        Console.WriteLine($"SPECULATIVE_CHECK_PROFILE decodeAttention={deployed.DecodeAttention} scratch={deployed.ScratchTokens} prefillChunk={deployed.PrefillChunkTokens} kv=fp32 entries={entries.Count}");
        var rows = new List<RunRow>(); var notes = new List<string>(); var failures = 0;
        var maxLength = entries.Max(e => e.Length);
        var lifecycleLength = Math.Min(maxLength, 64);
        var wanted = new SortedSet<int> { 0 };
        for (var p = interval; p < maxLength; p += interval) wanted.Add(p);
        foreach (var l in entries.Select(e => e.Length).Append(lifecycleLength).Distinct()) wanted.Add(l - 1);

        foreach (var item in cases)
        {
            ct.ThrowIfCancellationRequested();
            var baseline = Generate(gpu, item.Prompt, maxLength, wanted, ct);
            Console.WriteLine($"BASELINE case={item.Name} prompt={item.Prompt.Length} outputs={baseline.Tokens.Length} checkpoints={baseline.Checkpoints.Count} firstEndToken={baseline.EosAt}");
            if (item.Saved is { } saved)
            {
                var common = Math.Min(saved.Length, baseline.Tokens.Length);
                var mismatch = Enumerable.Range(0, common).FirstOrDefault(i => saved[i] != baseline.Tokens[i], -1);
                if (mismatch >= 0) { failures++; notes.Add($"{item.Name}: live baseline differs from saved outputTokenIds at output {mismatch}"); }
                else if (saved.Length < baseline.Tokens.Length) notes.Add($"{item.Name}: saved output has {saved.Length} tokens, live baseline {baseline.Tokens.Length}");
            }
            var noted = entries.Count(e => !e.IgnoreEos && baseline.EosAt is < 0 || !e.IgnoreEos && baseline.EosAt >= e.Length);
            if (noted > 0) notes.Add($"{item.Name}: {noted} natural-end entries never reached an end token within their length and ran as length-capped.");
            // Lifecycle: cancellation, recovery by reset, staleness and abandonment on the real graph.
            var lifecycle = new MatrixEntry("lifecycle", lifecycleLength, SyntheticPattern.Parse("mixed"), 4, true, false);
            try { await Lifecycle(gpu, item, lifecycle, baseline, ct); rows.Add(new(item.Name, lifecycle, "lifecycle", 0, "pass", null, null, 0, 0, 0)); }
            catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
            { failures++; rows.Add(new(item.Name, lifecycle, "lifecycle", 0, "fail", $"{ex.GetType().Name}: {ex.Message}", null, 0, 0, 0)); }

            foreach (var entry in entries)
            {
                var runs = entry.Repeated ? warm + formal : 1;
                for (var run = 0; run < runs; run++)
                {
                    ct.ThrowIfCancellationRequested();
                    var phase = !entry.Repeated ? "single" : run < warm ? "warm" : "formal"; var index = !entry.Repeated ? 1 : run < warm ? run + 1 : run - warm + 1;
                    RunRow row;
                    try
                    {
                        var (result, compared, terminal) = await Speculate(gpu, item, entry, baseline, poison, ct);
                        row = new(item.Name, entry, phase, index, "pass", null, result.Diagnostics, compared, result.TerminalPosition, terminal);
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException || !ct.IsCancellationRequested)
                    {
                        failures++;
                        row = new(item.Name, entry, phase, index, "fail", $"{ex.GetType().Name}: {ex.Message}", null, 0, 0, 0);
                        try { await File.WriteAllTextAsync(Path.Combine(output, $"failure-{failures}.json"), await Evidence(output, failures, row, ex, gpu, item, baseline, ct), ct); }
                        catch (Exception evidence) when (evidence is not OperationCanceledException) { Console.Error.WriteLine($"Evidence capture failed: {evidence.Message}"); }
                        Console.Error.WriteLine($"FAIL {item.Name} {entry.Group} L={entry.Length} {entry.Pattern.Name} B={entry.Block} {phase}{index}: {ex.Message}");
                    }
                    rows.Add(row);
                    if (row.Status == "pass" && run == runs - 1)
                        Console.WriteLine($"PASS case={item.Name} group={entry.Group} outputs={row.Diagnostics!.Emitted} pattern={entry.Pattern.Name} block={entry.Block} cycles={row.Diagnostics.Cycles} proposed={row.Diagnostics.Proposed} matched={row.Diagnostics.Matched} accepted={row.Diagnostics.Accepted} corrected={row.Diagnostics.Corrected} eosTruncations={row.Diagnostics.EosTruncations} finish={row.Diagnostics.FinishReason}");
                }
            }
        }
        await File.WriteAllTextAsync(Path.Combine(output, "speculative-check.json"), Report(profile.ToString(), deployed, custom, cases.Length, interval, warm, formal, poison, rows, notes, failures), ct);
        Console.WriteLine($"SPECULATIVE_CHECK runs={rows.Count} failures={failures} report={Path.Combine(output, "speculative-check.json")}");
        Console.WriteLine("Synthetic proposal correctness reference only: no draft model and no performance claim.");
        if (failures > 0) throw new InvalidOperationException($"Speculative state check failed in {failures} run(s); see {output}.");
    }

    // Serial non-speculative stream (the deployed decode loop shape) that never stops at an end token. Checkpoint i is the
    // state after consuming i outputs; position = prompt + i. Also records the first end token's boundary.
    private static Reference Generate(MetalSplitSession gpu, int[] prompt, int maxTokens, SortedSet<int> wanted, CancellationToken ct)
    {
        gpu.Reset(prompt.Length + maxTokens);
        var pending = gpu.Prefill(prompt, ct);
        var target = gpu.BeginSpeculative(prompt.Length);
        var tokens = new int[maxTokens]; var checkpoints = new SortedDictionary<int, Checkpoint>(); var eosAt = -1;
        for (var i = 0; i < maxTokens; i++)
        {
            ct.ThrowIfCancellationRequested();
            var end = eosAt < 0 && gpu.IsEnd(pending);
            if (end) eosAt = i;
            if (end || wanted.Contains(i)) checkpoints[i] = new(target.Digest(), target.LastLogits());
            tokens[i] = pending;
            if (i + 1 < maxTokens) { target.EnsureCapacity(prompt.Length + i + 1); pending = target.Consume(pending); }
        }
        return new(tokens, checkpoints, eosAt);
    }

    // The standard loop stops at the first end token (natural stop, state = all emitted tokens consumed, end token pending)
    // or at the cap (last emitted token unconsumed).
    private static Expectation Expect(Reference baseline, MatrixEntry entry)
        => !entry.IgnoreEos && baseline.EosAt >= 0 && baseline.EosAt < entry.Length
            ? new(baseline.Tokens[..baseline.EosAt], "eos", baseline.EosAt)
            : new(baseline.Tokens[..entry.Length], "length", entry.Length - 1);

    private static bool Same(float[] a, float[] b) => MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan()));

    // Exact full-state/logit comparison. Evidence from the live (diverged) target is captured before the engine abandons it.
    private static void Compare(MetalSpeculativeTarget target, Checkpoint reference, int consumed, string where)
    {
        var digest = target.Digest(); var logits = target.LastLogits();
        var bad = Enumerable.Range(0, 80).Where(i => digest[i] != reference.Digest[i]).ToArray();
        var logitsBad = !Same(logits, reference.Logits);
        if (bad.Length == 0 && !logitsBad) return;
        Dictionary<int, byte[]>? actual = null;
        // Bounded: at most two slots, none above 64 MiB.
        foreach (var slot in bad.Take(2))
        {
            var attention = (slot / 2 + 1) % 4 == 0;
            if (attention && (long)target.Position * 512 * 4 > 64 << 20) continue;
            (actual ??= [])[slot] = target.ExportState(slot / 2, slot % 2);
        }
        throw new Divergence(bad.Length > 0
            ? $"State digest differs at {where} after {consumed} consumed outputs (layer/tensor slots {string.Join(',', bad.Take(8))})."
            : $"Logits differ at {where} after {consumed} consumed outputs.")
        { Kind = bad.Length > 0 ? "state" : "logits", Where = where, Emitted = consumed, Tensors = bad, ExpectedDigest = reference.Digest, ActualDigest = digest,
          ExpectedLogits = reference.Logits, ActualLogits = logits, ActualState = actual };
    }

    // Runs one configuration through the real engine and verifies tokens, finish reason, terminal position, the unconsumed
    // pending token and the exact terminal state and logits (not only the periodic checkpoints).
    private static async Task<(SpeculativeRun Run, int Compared, int Terminal)> Speculate(MetalSplitSession gpu, Case item, MatrixEntry entry, Reference baseline,
        bool poison, CancellationToken ct)
    {
        var expect = Expect(baseline, entry);
        // Terminal boundaries are checked after the engine returns, never forced as block boundaries, so an end token
        // inside an accepted block is exercised as it would be in service.
        var points = baseline.Checkpoints.Keys.Where(k => k < expect.Terminal).ToArray();
        gpu.Reset(item.Prompt.Length + entry.Length);
        var pending = gpu.Prefill(item.Prompt, ct);
        var target = gpu.BeginSpeculative(item.Prompt.Length);
        var compared = 0;
        var run = await SpeculativeEngine.Run(target, pending, entry.Length, entry.IgnoreEos, gpu.IsEnd, entry.Pattern, entry.Block, new ReferenceOracle(baseline.Tokens), ct, null, points, (emitted, next) =>
        {
            if (next != baseline.Tokens[emitted]) throw new Divergence($"Pending token {next} differs from baseline {baseline.Tokens[emitted]} at output {emitted}.") { Kind = "pending", Emitted = emitted };
            Compare(target, baseline.Checkpoints[emitted], emitted, "checkpoint");
            compared++;
            // Suffix rows were not consumed; NaNs there fail the next block if any kernel reads past the logical position.
            if (poison) target.PoisonKvSuffix(16);
        }, keepDetail: false);
        if (!run.Output.AsSpan().SequenceEqual(expect.Tokens))
        {
            var common = Math.Min(expect.Tokens.Length, run.Output.Length);
            throw new Divergence($"Output tokens differ at index {Enumerable.Range(0, common).FirstOrDefault(i => expect.Tokens[i] != run.Output[i], common)} (expected {expect.Tokens.Length} tokens, got {run.Output.Length}).") { Kind = "tokens", Emitted = expect.Terminal };
        }
        if (run.FinishReason != expect.Finish) throw new Divergence($"Finish reason {run.FinishReason} differs from the baseline's {expect.Finish}.") { Kind = "finish", Emitted = expect.Terminal };
        if (run.TerminalPosition != item.Prompt.Length + expect.Terminal)
            throw new Divergence($"Terminal position {run.TerminalPosition} differs from the standard loop's {item.Prompt.Length + expect.Terminal}.") { Kind = "terminal-position", Emitted = expect.Terminal };
        if (run.FinalPending != baseline.Tokens[expect.Terminal])
            throw new Divergence($"Terminal pending token {run.FinalPending} differs from baseline {baseline.Tokens[expect.Terminal]}.") { Kind = "terminal-pending", Emitted = expect.Terminal };
        Compare(target, baseline.Checkpoints[expect.Terminal], expect.Terminal, "terminal");
        return (run, compared + 1, expect.Terminal);
    }

    // Exercises cancellation, abandonment, recovery by reset, snapshot staleness and cross-request rejection on the real graph.
    private static async Task Lifecycle(MetalSplitSession gpu, Case item, MatrixEntry entry, Reference baseline, CancellationToken ct)
    {
        var prompt = item.Prompt; var length = entry.Length;
        if (length < 8) return;
        gpu.Reset(prompt.Length + length);
        var pending = gpu.Prefill(prompt, CancellationToken.None);
        var target = gpu.BeginSpeculative(prompt.Length);
        using var cancel = new CancellationTokenSource();
        var cancelled = false;
        try
        {
            await SpeculativeEngine.Run(target, pending, length, true, gpu.IsEnd, entry.Pattern, entry.Block, new ReferenceOracle(baseline.Tokens), cancel.Token,
                async _ => { if (!cancel.IsCancellationRequested) cancel.Cancel(); await ValueTask.CompletedTask; });
        }
        catch (OperationCanceledException) { cancelled = true; }
        if (!cancelled) throw new InvalidOperationException("Cancellation did not stop the speculative run.");
        if (!Throws(() => target.Consume(0))) throw new InvalidOperationException("Cancelled target remained usable.");
        // The next request recovers by resetting the graph and must reproduce the baseline exactly.
        await Speculate(gpu, item, entry, baseline, false, ct);
        // A reset starts a new request epoch: the previous request's snapshot must be rejected natively.
        gpu.Reset(prompt.Length + length);
        _ = gpu.Prefill(prompt, CancellationToken.None);
        var live = gpu.BeginSpeculative(prompt.Length);
        live.Capture();
        if (live.SnapshotInfo()[9] != 1) throw new InvalidOperationException("A captured snapshot is not restorable.");
        gpu.Reset(prompt.Length + length);
        if (!Throws(() => live.Restore())) throw new InvalidOperationException("Snapshot survived a graph reset.");
        _ = gpu.Prefill(prompt, CancellationToken.None);
        var fresh = gpu.BeginSpeculative(prompt.Length);
        if (fresh.SnapshotInfo()[9] != 0) throw new InvalidOperationException("A stale snapshot reports as restorable for a new request.");
    }

    private static bool Throws(Action action) { try { action(); return false; } catch (InvalidOperationException) { return true; } }

    private static string Hex(uint bits) => "0x" + bits.ToString("x8");
    private static void Number(Utf8JsonWriter w, string name, double value) { if (double.IsFinite(value)) w.WriteNumber(name, value); else w.WriteNull(name); }
    private sealed record FloatDiff(int Count, int First, double MaxAbs, uint ExpectedBits, uint ActualBits);
    // Bitwise comparison; JSON never carries non-finite numbers (raw bits are reported instead).
    private static FloatDiff Diff(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        var count = 0; var first = -1; var max = 0d; uint eb = 0, ab = 0;
        for (var i = 0; i < Math.Min(expected.Length, actual.Length); i++)
        {
            var x = BitConverter.SingleToUInt32Bits(expected[i]); var y = BitConverter.SingleToUInt32Bits(actual[i]);
            if (x == y) continue;
            if (first < 0) { first = i; eb = x; ab = y; }
            count++;
            var d = Math.Abs((double)expected[i] - actual[i]);
            if (double.IsFinite(d)) max = Math.Max(max, d);
        }
        if (expected.Length != actual.Length) count += Math.Abs(expected.Length - actual.Length);
        return new(count, first, max, eb, ab);
    }
    private static async Task WriteFloats(string path, ReadOnlyMemory<float> values, CancellationToken ct) => await File.WriteAllBytesAsync(path, MemoryMarshal.AsBytes(values.Span).ToArray(), ct);

    // The reference state is regenerated serially after a failure (the failing target is already abandoned), so evidence
    // contains the actual and the independent expected tensor, not merely their digests.
    private static Dictionary<int, byte[]> ReferenceState(MetalSplitSession gpu, Case item, Reference baseline, int consumed, IEnumerable<int> slots)
    {
        gpu.Reset(item.Prompt.Length + consumed + 1);
        _ = gpu.Prefill(item.Prompt, CancellationToken.None);
        var target = gpu.BeginSpeculative(item.Prompt.Length);
        for (var i = 0; i < consumed; i++) { target.EnsureCapacity(item.Prompt.Length + i + 1); _ = target.Consume(baseline.Tokens[i]); }
        return slots.ToDictionary(s => s, s => target.ExportState(s / 2, s % 2));
    }

    private static async Task<string> Evidence(string directory, int number, RunRow row, Exception ex, MetalSplitSession gpu, Case item, Reference baseline, CancellationToken ct)
    {
        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream, new() { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("case", row.Case); w.WriteString("group", row.Entry.Group); w.WriteNumber("length", row.Entry.Length); w.WriteString("pattern", row.Entry.Pattern.Name);
            w.WriteNumber("block", row.Entry.Block); w.WriteBoolean("ignoreEos", row.Entry.IgnoreEos); w.WriteString("phase", row.Phase);
            w.WriteString("exception", ex.GetType().Name); w.WriteString("message", ex.Message);
            if (ex is Divergence d)
            {
                w.WriteString("kind", d.Kind); w.WriteString("where", d.Where); w.WriteNumber("consumedOutputs", d.Emitted);
                if (d.ExpectedLogits is not null && d.ActualLogits is not null)
                {
                    var diff = Diff(d.ExpectedLogits, d.ActualLogits);
                    var prefix = $"failure-{number}.logits";
                    await WriteFloats(Path.Combine(directory, prefix + ".expected.f32"), d.ExpectedLogits, ct);
                    await WriteFloats(Path.Combine(directory, prefix + ".actual.f32"), d.ActualLogits, ct);
                    w.WriteStartObject("logits");
                    w.WriteNumber("differentElements", diff.Count); w.WriteNumber("firstDifferentIndex", diff.First); Number(w, "maxAbsDifference", diff.MaxAbs);
                    w.WriteString("firstExpectedBits", Hex(diff.ExpectedBits)); w.WriteString("firstActualBits", Hex(diff.ActualBits));
                    w.WriteString("expectedFile", prefix + ".expected.f32"); w.WriteString("actualFile", prefix + ".actual.f32"); w.WriteString("format", "float32 little-endian, 248320 values");
                    w.WriteEndObject();
                }
                if (d.Tensors is { Length: > 0 })
                {
                    Dictionary<int, byte[]>? expected = null; string? reference = null;
                    if (d.ActualState is not null)
                        try { expected = ReferenceState(gpu, item, baseline, d.Emitted, d.ActualState.Keys); }
                        catch (Exception regenerate) when (regenerate is not OperationCanceledException) { reference = regenerate.Message; }
                    w.WriteStartArray("slots");
                    foreach (var slot in d.Tensors)
                    {
                        w.WriteStartObject();
                        w.WriteNumber("slot", slot); w.WriteNumber("layer", slot / 2);
                        w.WriteString("tensor", (slot / 2 + 1) % 4 == 0 ? (slot % 2 == 0 ? "keys" : "values") : (slot % 2 == 0 ? "recurrent-state" : "convolution-history"));
                        w.WriteString("expectedDigest", "0x" + d.ExpectedDigest![slot].ToString("x16")); w.WriteString("actualDigest", "0x" + d.ActualDigest![slot].ToString("x16"));
                        if (d.ActualState is not null && d.ActualState.TryGetValue(slot, out var actual))
                        {
                            var file = $"failure-{number}.slot{slot}";
                            await File.WriteAllBytesAsync(Path.Combine(directory, file + ".actual.f32"), actual, ct);
                            w.WriteString("actualFile", file + ".actual.f32");
                            if (expected is not null && expected.TryGetValue(slot, out var want))
                            {
                                await File.WriteAllBytesAsync(Path.Combine(directory, file + ".expected.f32"), want, ct);
                                var diff = Diff(MemoryMarshal.Cast<byte, float>(want), MemoryMarshal.Cast<byte, float>(actual));
                                w.WriteString("expectedFile", file + ".expected.f32"); w.WriteNumber("differentElements", diff.Count);
                                w.WriteNumber("firstDifferentElement", diff.First); Number(w, "maxAbsDifference", diff.MaxAbs);
                                w.WriteString("firstExpectedBits", Hex(diff.ExpectedBits)); w.WriteString("firstActualBits", Hex(diff.ActualBits));
                            }
                        }
                        w.WriteEndObject();
                    }
                    w.WriteEndArray();
                    if (reference is not null) w.WriteString("referenceRegenerationFailed", reference);
                }
            }
            w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }


    private static string Report(string profile, DeployedProfile deployed, bool custom, int cases, int interval, int warm, int formal, bool poison,
        List<RunRow> rows, List<string> notes, int failures)
    {
        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream, new() { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("schema", "speculative-state-check-v2");
            w.WriteString("scope", "synthetic proposal correctness reference; no draft model; diagnostic durations only; no acceleration claim");
            w.WriteString("nativeProfile", profile); w.WriteString("decodeAttention", deployed.DecodeAttention);
            w.WriteNumber("scratchTokens", deployed.ScratchTokens); w.WriteNumber("prefillChunkTokens", deployed.PrefillChunkTokens);
            w.WriteString("matrix", custom ? "custom" : "default"); w.WriteNumber("cases", cases);
            w.WriteNumber("checkpointTokens", interval); w.WriteNumber("warmRuns", warm); w.WriteNumber("formalRuns", formal);
            w.WriteBoolean("poisonSuffix", poison); w.WriteNumber("failures", failures);
            w.WriteStartArray("notes"); foreach (var n in notes) w.WriteStringValue(n); w.WriteEndArray();
            w.WriteStartArray("runs");
            foreach (var r in rows)
            {
                w.WriteStartObject();
                w.WriteString("case", r.Case); w.WriteString("group", r.Entry.Group); w.WriteNumber("lengthCap", r.Entry.Length);
                w.WriteString("pattern", r.Entry.Pattern.Name); w.WriteNumber("block", r.Entry.Block); w.WriteBoolean("ignoreEos", r.Entry.IgnoreEos);
                w.WriteString("phase", r.Phase); w.WriteNumber("index", r.Index); w.WriteString("status", r.Status);
                w.WriteString("oracle", "saved-baseline-token-ids"); w.WriteNumber("checkpointsCompared", r.Checkpoints);
                if (r.Status == "pass" && r.Phase != "lifecycle") { w.WriteNumber("terminalPosition", r.TerminalPosition); w.WriteNumber("terminalConsumedOutputs", r.TerminalIndex); }
                if (r.Failure is not null) w.WriteString("failure", r.Failure);
                if (r.Diagnostics is { } d)
                {
                    w.WriteString("finishReason", d.FinishReason);
                    w.WriteNumber("cycles", d.Cycles); w.WriteNumber("proposed", d.Proposed); w.WriteNumber("verified", d.Verified); w.WriteNumber("matched", d.Matched);
                    w.WriteNumber("accepted", d.Accepted); w.WriteNumber("corrected", d.Corrected); w.WriteNumber("bonus", d.Bonus); w.WriteNumber("emitted", d.Emitted);
                    w.WriteNumber("eosTruncations", d.EosTruncations);
                    w.WriteNumber("verifyForwards", d.VerifyForwards); w.WriteNumber("replayForwards", d.ReplayForwards);
                    Number(w, "snapshotMs", d.SnapshotMs); Number(w, "restoreMs", d.RestoreMs); Number(w, "replayMs", d.ReplayMs);
                    Number(w, "verifyMs", d.VerifyMs); Number(w, "firstEmissionMs", d.FirstEmissionMs); Number(w, "emissionLifecycleMs", d.EmissionLifecycleMs);
                }
                w.WriteEndObject();
            }
            w.WriteEndArray(); w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }
}
