using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using DeepInfer.Backends.Dflash2;
using DeepInfer.Backends.Split;

const long Dflash2HostBudget = (long)(38.5 * 1024 * 1024 * 1024);
if (args.Length < 4 || args[0] is not ("gate" or "request"))
    throw new ArgumentException("Usage: gate|request TARGET DRAFT OUTPUT [off|scalar|native] [inputTokens] [outputTokens]");
var mode = args.Length > 4 ? args[4] : "native";
var folder = Path.GetFullPath(args[3]); Directory.CreateDirectory(folder);
void Save(string name, object value) => File.WriteAllText(Path.Combine(folder, name), JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }) + "\n");
MetalSplitSession.ConfigureAllocationTrace(true);
using var session = new MetalSplitSession(args[1], 4, true, 1000, 128, true);
var replay = mode switch { "scalar" => Dflash2ReplayMode.Scalar, "native" => Dflash2ReplayMode.Native, "off" => Dflash2ReplayMode.Batched, _ => throw new ArgumentException("Unknown mode") };
if (mode != "off") session.EnableDflash2(args[2], 8, replay, true, args[0] == "gate");
var native = Path.Combine(AppContext.BaseDirectory, "libdeepinfer_metal.dylib");
Save("runtime.json", new { mode, profile = session.ExecutionProfile(), stats = Stats(), nativeSha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(native))), source = typeof(MetalSplitSession).Assembly.GetCustomAttributes(false).OfType<System.Reflection.AssemblyInformationalVersionAttribute>().Single().InformationalVersion });
object Stats() { var v = session.Stats(); return new { v.Capacity, v.KvBytes, v.AllocatedBytes }; }
void Budget() { if (session.Stats().AllocatedBytes > Dflash2HostBudget || MetalSplitSession.AllocatorTrace().PeakBytes > Dflash2HostBudget) throw new Exception("Metal allocation exceeds experiment budget"); }
int[] Prompt(int count)
{
    var source = session.Tokenize(string.Concat(Enumerable.Repeat("A graph traversal records each visited vertex and checks its neighbors in a queue. ", 64)), default);
    return Enumerable.Range(0, count).Select(i => source[i % source.Length]).ToArray();
}
if (args[0] == "request")
{
    var length = int.Parse(args[5]); var max = int.Parse(args[6]); var prompt = Prompt(length);
    Save("fixture.json", new { inputTokens = length, outputTokens = max, tokenIdsSha256 = Convert.ToHexStringLower(SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(prompt.AsSpan()))) });
    session.Reset(length + max + 8); Budget();
    var timer = Stopwatch.StartNew(); var times = new List<double>(); var tokens = new List<int>(); var prefill = 0d; Dflash2Stats? speculation = null;
    if (mode == "off")
    {
        var next = session.Prefill(prompt, default); prefill = timer.Elapsed.TotalMilliseconds;
        for (var i = 0; i < max; i++) { if (i > 0) next = session.Forward(next, length + i - 1); tokens.Add(next); times.Add(timer.Elapsed.TotalMilliseconds); }
    }
    else
    {
        using var run = session.StartDflash2(max, true, () => timer.Elapsed.TotalMilliseconds);
        run.Engine.Prefill(prompt, default); prefill = timer.Elapsed.TotalMilliseconds;
        while (run.Engine.TryNext(default, out var block)) foreach (var token in block.Tokens) { tokens.Add(token); times.Add(block.ReadyMs); }
        speculation = run.Engine.Stats;
    }
    timer.Stop(); Budget();
    if (tokens.Count != max) throw new Exception("Output length mismatch");
    var decode = times[^1] - times[0]; var quarter = Math.Max(1, max / 4); var quarterMs = times[^1] - times[^quarter];
    Save("request.json", new { mode, inputTokens = length, outputTokens = tokens.Count, prefillMs = prefill, decodeMs = decode, decodeIntervals = max - 1, decodeTokensPerSecond = (max - 1) * 1000 / decode, totalMs = timer.Elapsed.TotalMilliseconds, lastQuarterTokensPerSecond = (quarter - 1) * 1000 / quarterMs, tokenIds = tokens, readyMs = times, speculation, stats = Stats(), allocation = MetalSplitSession.AllocatorTrace() });
}
else
{
    // Forced rows are separate from real proposal timing; they validate every prefix and all live state.
    var prompt = Prompt(21); session.Reset(256); var anchor = session.Prefill(prompt, default);
    var candidates = Enumerable.Range(0, 8).Select(i => prompt[(i * 7 + 3) % prompt.Length]).ToArray(); candidates[0] = anchor;
    var reference = new List<byte[][]>(); var next = new int[8]; var taps = new List<float[]>();
    for (var row = 0; row < 8; row++)
    {
        var logits = new float[248320]; var rowTaps = new float[16384]; next[row] = session.ScalarForward(candidates[row], prompt.Length + row, logits, rowTaps); taps.Add(rowTaps);
        var states = new byte[80][]; for (var l = 0; l < 40; l++) for (var k = 0; k < 2; k++) states[l * 2 + k] = session.TargetState(l, k);
        reference.Add(states);
    }
    var passed = new List<object>();
    foreach (var accepted in Enumerable.Range(0, 9))
    {
        session.Reset(256); session.Prefill(prompt, default);
        var before = session.StateDigest(); var actual = session.VerifyTargetBlock(candidates); if (!actual.SequenceEqual(next)) throw new Exception("Native row argmax differs from scalar");
        var capture = session.ExportTaps(8);
        for (var r = 0; r < 8; r++) Near(capture.AsSpan(r * 16384, 16384), taps[r]);
        // Public read-only diagnostics can observe taps; draft ingest must reject uncommitted rows.
        try { session.Dflash2Ingest(0, 1, prompt.Length); throw new Exception("Tentative ingest accepted"); } catch (InvalidOperationException) { }
        session.CommitTargetBlock(accepted);
        if (accepted == 0) { if (!before.SequenceEqual(session.StateDigest())) throw new Exception("Discard changed visible state"); }
        else
        {
            for (var l = 0; l < 40; l++) for (var k = 0; k < 2; k++) Near(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(session.TargetState(l, k)), System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(reference[accepted - 1][l * 2 + k]));
            // Reset draft context, then ingest only the published prefix at its absolute positions.
            session.Dflash2Reset(); session.Dflash2Ingest(0, accepted, prompt.Length);
            if (session.Dflash2State().ContextEnd != prompt.Length + accepted) throw new Exception("Draft context prefix mismatch");
        }
        passed.Add(new { accepted, passed = true }); Save("gate.json", passed); Budget();
    }
    foreach (var max in new[] { 1, 2, 7, 8, 9, 15, 16, 17, 32, 128 })
    {
        session.Reset(256); var expected = new List<int>(); var first = session.Prefill(prompt, default);
        for (var i = 0; i < max; i++) { if (i > 0) first = session.Forward(first, prompt.Length + i - 1); expected.Add(first); }
        session.Reset(256); var timer = Stopwatch.StartNew(); using var run = session.StartDflash2(max, true, () => timer.Elapsed.TotalMilliseconds); run.Engine.Prefill(prompt, default); var output = new List<int>();
        while (run.Engine.TryNext(default, out var b)) output.AddRange(b.Tokens);
        if (!expected.SequenceEqual(output)) throw new Exception("Real draft cycle differs from scalar at output limit " + max);
        passed.Add(new { outputLimit = max, passed = true, stats = run.Engine.Stats }); Save("gate.json", passed); Budget();
    }
}
static void Near(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
{
    if (actual.Length != expected.Length) throw new Exception("Geometry mismatch"); double sum = 0, norm = 0, max = 0;
    for (var i = 0; i < actual.Length; i++) { if (!float.IsFinite(actual[i]) || !float.IsFinite(expected[i])) throw new Exception("Non-finite state"); var d = (double)actual[i] - expected[i]; sum += d*d; norm += (double)expected[i]*expected[i]; max = Math.Max(max, Math.Abs(d)); }
    if (Math.Sqrt(sum / Math.Max(norm, 1e-30)) >= 1e-4 || max >= 0.01) throw new Exception($"State mismatch rel={Math.Sqrt(sum / Math.Max(norm, 1e-30))} abs={max}");
}
