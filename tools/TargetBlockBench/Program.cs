using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using DeepInfer.Backends.Split;

// Target transactions are measured in milliseconds, never advertised as emitted
// autoregressive tokens/s. GPU time, CPU wall and commit overhead stay separate.
if (args.Length != 3 || args[0] is not ("gate" or "bench" or "long-gate"))
    throw new ArgumentException("Usage: TargetBlockBench gate|long-gate|bench MODEL_DIRECTORY OUTPUT_DIRECTORY");
var output = Path.GetFullPath(args[2]); Directory.CreateDirectory(output);
void Save(string name, object value) => File.WriteAllText(Path.Combine(output, name), JsonSerializer.Serialize(value, new JsonSerializerOptions { WriteIndented = true }));
void Near(string label, ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
{
    if (expected.Length != actual.Length) throw new Exception(label + " shape mismatch");
    double error = 0, norm = 0, maximum = 0; var worst = 0;
    for (var i = 0; i < expected.Length; i++)
    {
        if (!float.IsFinite(actual[i]) || !float.IsFinite(expected[i])) throw new Exception(label + " non-finite at " + i);
        var d = (double)actual[i] - expected[i]; error += d * d; norm += (double)expected[i] * expected[i];
        if (Math.Abs(d) > maximum) { maximum = Math.Abs(d); worst = i; }
    }
    var relative = Math.Sqrt(error / Math.Max(norm, 1e-30));
    if (relative >= 1e-4 || maximum >= 0.01) throw new Exception($"{label}: relativeL2={relative:R} maxAbs={maximum:R} index={worst}");
}
void SameTokens(string label, int[] expected, int[] actual)
{
    if (!expected.SequenceEqual(actual)) throw new Exception(label + ": token mismatch at " + Enumerable.Range(0, Math.Min(expected.Length, actual.Length)).FirstOrDefault(i => expected[i] != actual[i], -1));
}
using var s = new MetalSplitSession(args[1], 4, true, 1000, args[0] == "gate" ? 128 : 2048, true);
s.PrepareTargetBlock(8);
object Stats() { var v = s.Stats(); return new { v.Capacity, v.KvBytes, v.AllocatedBytes }; }
object BlockStats() { var v = s.TargetBlockStats(); return new { v.Bytes, v.Position, v.Pending }; }
object ManagedHeap() { var v = GC.GetGCMemoryInfo(); return new { liveBytes = GC.GetTotalMemory(false), v.HeapSizeBytes, v.TotalCommittedBytes, v.TotalAvailableMemoryBytes }; }
Save("runtime.json", new { ManagedHeap = ManagedHeap(), Profile = s.ExecutionProfile(), Stats = Stats(), Block = BlockStats(), NativeSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "libdeepinfer_metal.dylib")))), AssemblySha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(typeof(MetalSplitSession).Assembly.Location))) });
var fixtureText = args[0] == "gate" ? "Explain how a Python iterator keeps its state and advances through a sequence. " : "A sparse graph algorithm visits neighboring vertices and records each shortest path in a queue. ";
var fixture = s.Tokenize(string.Concat(Enumerable.Repeat(fixtureText, 20)), CancellationToken.None);
Save("fixture.json", new { mode = args[0], tokensSha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(fixture.AsSpan()))), lengths = args[0] == "gate" ? new[] { 21, 2047 } : new[] { 21, 8192, 32768 } });
int[] Prompt(int length) => Enumerable.Range(0, length).Select(i => fixture[i % fixture.Length]).ToArray();
var completed = new List<object>();
if (args[0] == "gate")
{
    var prompt = Prompt(21);
    foreach (var rows in new[] { 1, 4, 8 })
    {
        s.Reset(2048); var first = s.Prefill(prompt, CancellationToken.None);
        var inputs = new int[rows]; var expectedNext = new int[rows]; var expectedLogits = new float[rows * 248320]; var expectedTrace = new float[rows * 40 * 2048];
        for (var row = 0; row < rows; row++)
        {
            var current = row; inputs[row] = row == 0 ? first : expectedNext[row - 1]; var logits = new float[248320];
            expectedNext[row] = s.Forward(inputs[row], prompt.Length + row, logits, (_, layer, values) => values.CopyTo(expectedTrace, (current * 40 + layer) * 2048));
            logits.CopyTo(expectedLogits, row * 248320);
        }
        for (var accepted = 0; accepted <= rows; accepted++)
        {
            s.Reset(2048); s.Prefill(prompt, CancellationToken.None);
            for (var i = 0; i < accepted; i++) s.Forward(inputs[i], prompt.Length + i);
            var expectedState = new byte[80][];
            for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++) expectedState[layer * 2 + kind] = s.TargetState(layer, kind);
            var nextInput = accepted == rows ? expectedNext[^1] : inputs[accepted]; var expectedContinuation = new int[3];
            for (var i = 0; i < 3; i++) { nextInput = s.Forward(nextInput, prompt.Length + accepted + i); expectedContinuation[i] = nextInput; }
            s.Reset(2048); s.Prefill(prompt, CancellationToken.None);
            var before = new string[80];
            for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++) before[layer * 2 + kind] = Convert.ToHexString(SHA256.HashData(s.TargetState(layer, kind)));
            var actualLogits = new float[rows * 248320]; var actualTrace = new float[rows * 40 * 2048];
            SameTokens("verify", expectedNext, s.VerifyTargetBlock(inputs, actualLogits, actualTrace));
            for (var row = 0; row < rows; row++)
            {
                Near($"block logits row={row}", expectedLogits.AsSpan(row * 248320, 248320), actualLogits.AsSpan(row * 248320, 248320));
                for (var layer = 0; layer < 40; layer++) Near($"block trace row={row} layer={layer}", expectedTrace.AsSpan((row * 40 + layer) * 2048, 2048), actualTrace.AsSpan((row * 40 + layer) * 2048, 2048));
            }
            for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++)
                if (before[layer * 2 + kind] != Convert.ToHexString(SHA256.HashData(s.TargetState(layer, kind)))) throw new Exception("Verification mutated live state");
            try { s.Forward(first, prompt.Length); throw new Exception("Pending decode was accepted"); } catch (InvalidOperationException) { }
            try { s.VerifyTargetBlock(inputs); throw new Exception("Nested verify was accepted"); } catch (InvalidOperationException) { }
            try { s.EnsureCapacity(4096); throw new Exception("Pending grow was accepted"); } catch (InvalidOperationException) { }
            s.CommitTargetBlock(accepted);
            if (s.TargetBlockStats().Position != prompt.Length + accepted || s.TargetBlockStats().Pending) throw new Exception("Incorrect committed position");
            for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++)
                Near($"committed state layer={layer} kind={kind}", MemoryMarshal.Cast<byte, float>(expectedState[layer * 2 + kind]), MemoryMarshal.Cast<byte, float>(s.TargetState(layer, kind)));
            nextInput = accepted == rows ? expectedNext[^1] : inputs[accepted]; var actualContinuation = new int[3];
            for (var i = 0; i < 3; i++) { nextInput = s.Forward(nextInput, prompt.Length + accepted + i); actualContinuation[i] = nextInput; }
            SameTokens("continuation", expectedContinuation, actualContinuation);
            completed.Add(new { rows, accepted, passed = true }); Save("gate.json", completed); Console.WriteLine($"PASS rows={rows} accepted={accepted}");
        }
    }
    // A pending block can be discarded by reset; context growth is explicit.
    s.Reset(2048); s.Prefill(Prompt(2047), CancellationToken.None); s.EnsureCapacity(4096);
    var growthInputs = Enumerable.Repeat(fixture[5], 8).ToArray(); s.VerifyTargetBlock(growthInputs); s.CommitTargetBlock(4);
    if (s.TargetBlockStats().Position != 2051) throw new Exception("Context boundary commit failed");
    s.VerifyTargetBlock(growthInputs); s.Reset(4096);
    if (s.TargetBlockStats().Pending || s.TargetBlockStats().Position != 0) throw new Exception("Pending reset failed");
    completed.Add(new { growth = true, reset = true, passed = true }); Save("gate.json", completed);
}
else if (args[0] == "long-gate")
{
    Save("fixture.json", new { mode = args[0], tokensSha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(fixture.AsSpan()))), lengths = new[] { 2047, 4093, 8192, 32768 } });
    foreach (var length in new[] { 2047, 4093, 8192, 32768 })
    {
        var prompt = Prompt(length); s.Reset(length + 64); var first = s.Prefill(prompt, CancellationToken.None);
        var inputs = Enumerable.Range(0, 8).Select(i => fixture[(i * 11 + 3) % fixture.Length]).ToArray(); inputs[0] = first;
        var expectedNext = new int[8]; var expectedLogits = new float[8 * 248320]; var expectedTrace = new float[8 * 40 * 2048];
        var continuations = new Dictionary<int, int[]>();
        var states = new Dictionary<int, byte[][]>(); var kvFiles = new Dictionary<int, string[]>();
        for (var row = 0; row < 8; row++)
        {
            var current = row; var logits = new float[248320];
            expectedNext[row] = s.Forward(inputs[row], length + row, logits, (_, layer, values) => values.CopyTo(expectedTrace, (current * 40 + layer) * 2048));
            logits.CopyTo(expectedLogits, row * 248320);
            if (row is 0 or 3 or 7)
            {
                var snapshot = new byte[80][]; var files = new string[80];
                for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++)
                {
                    var bytes = s.TargetState(layer, kind);
                    if ((layer + 1) % 4 == 0)
                    {
                        var file = Path.Combine(output, $"state-{length}-{row + 1}-{layer}-{kind}.bin"); File.WriteAllBytes(file, bytes); files[layer * 2 + kind] = file;
                    }
                    else snapshot[layer * 2 + kind] = bytes;
                }
                states[row + 1] = snapshot; kvFiles[row + 1] = files;
            }
        }
        foreach (var accepted in new[] { 1, 4, 8 }) foreach (var arm in new[] { "block", "scalar-transaction" })
        {
            s.Reset(length + 64); s.Prefill(prompt, CancellationToken.None);
            if (arm == "block")
            {
                var logits = new float[8 * 248320]; var trace = new float[8 * 40 * 2048];
                SameTokens("long verify", expectedNext, s.VerifyTargetBlock(inputs, logits, trace));
                for (var row = 0; row < 8; row++)
                {
                    Near($"long logits row={row}", expectedLogits.AsSpan(row * 248320, 248320), logits.AsSpan(row * 248320, 248320));
                    for (var layer = 0; layer < 40; layer++) Near($"long trace row={row} layer={layer}", expectedTrace.AsSpan((row * 40 + layer) * 2048, 2048), trace.AsSpan((row * 40 + layer) * 2048, 2048));
                }
                foreach (var prefix in new[] { 1, 4, 8 }) for (var layer = 0; layer < 40; layer++) if ((layer + 1) % 4 != 0)
                    Near($"pending prefix={prefix} layer={layer}", MemoryMarshal.Cast<byte, float>(states[prefix][layer * 2]), MemoryMarshal.Cast<byte, float>(s.TargetState(layer, 0, prefix)));
                s.CommitTargetBlock(accepted);
            }
            else s.ScalarTargetTransaction(inputs, accepted);
            for (var layer = 0; layer < 40; layer++) for (var kind = 0; kind < 2; kind++)
            {
                var actual = s.TargetState(layer, kind);
                if ((layer + 1) % 4 == 0)
                {
                    Near($"long KV context={length} prefix={accepted} layer={layer} kind={kind}", MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(kvFiles[accepted][layer * 2 + kind])), MemoryMarshal.Cast<byte, float>(actual));
                }
                else Near($"long committed prefix={accepted} layer={layer} kind={kind}", MemoryMarshal.Cast<byte, float>(states[accepted][layer * 2 + kind]), MemoryMarshal.Cast<byte, float>(actual));
            }
            if (s.TargetBlockStats().Position != length + accepted) throw new Exception("Long committed position mismatch");
            var nextInput = expectedNext[accepted - 1]; var continuation = new int[3];
            for (var i = 0; i < 3; i++) { nextInput = s.Forward(nextInput, length + accepted + i); continuation[i] = nextInput; }
            if (arm == "block") continuations[accepted] = continuation;
            else SameTokens("long continuation", continuations[accepted], continuation);
            completed.Add(new { length, accepted, arm, passed = true, stats = Stats(), block = BlockStats(), managedHeap = ManagedHeap() }); Save("long-gate.json", completed);
            Console.WriteLine($"PASS long context={length} prefix={accepted} arm={arm}");
        }
    }
}

else
{
    // Fixed candidate inputs are independent of per-arm execution. The same
    // full block is executed in both arms; only the chosen prefix is published.
    foreach (var length in new[] { 21, 8192, 32768 }) foreach (var rows in new[] { 1, 4, 8 }) foreach (var accepted in new[] { 1, 4, 8 }.Where(n => n <= rows))
    {
        var prompt = Prompt(length);
        foreach (var trial in Enumerable.Range(-1, 4)) foreach (var arm in trial % 2 == 0 ? new[] { "block", "scalar" } : new[] { "scalar", "block" })
        {
            var startedUtc = DateTimeOffset.UtcNow;
            string VmStat() { using var p = Process.Start(new ProcessStartInfo("/usr/bin/vm_stat") { RedirectStandardOutput = true })!; var text = p.StandardOutput.ReadToEnd(); p.WaitForExit(); return text; }
            var vmBefore = VmStat();
            s.Reset(length + 256); var first = s.Prefill(prompt, CancellationToken.None); var position = length;
            var totalWall = 0d; var commitWall = 0d; var gpuStart = s.GpuMilliseconds; var dispatchStart = s.Dispatches; var submissionStart = s.Submissions;
            for (var step = 0; step < 128 / rows; step++)
            {
                var input = Enumerable.Range(0, rows).Select(i => fixture[(step * 13 + i * 7) % fixture.Length]).ToArray(); input[0] = first;
                if (arm == "block")
                {
                    var clock = Stopwatch.StartNew(); s.VerifyTargetBlock(input); var commit = Stopwatch.StartNew(); s.CommitTargetBlock(accepted); commit.Stop(); clock.Stop();
                    totalWall += clock.Elapsed.TotalMilliseconds; commitWall += commit.Elapsed.TotalMilliseconds; position += accepted;
                }
                else
                {
                    var clock = Stopwatch.StartNew(); s.ScalarTargetTransaction(input, accepted); clock.Stop(); totalWall += clock.Elapsed.TotalMilliseconds; position += accepted;
                }
            }
            completed.Add(new { length, rows, accepted, trial, arm, startedUtc, finishedUtc = DateTimeOffset.UtcNow, vmBefore, vmAfter = VmStat(), totalWallMs = totalWall, commitWallMs = commitWall, gpuMs = s.GpuMilliseconds - gpuStart, dispatches = s.Dispatches - dispatchStart, submissions = s.Submissions - submissionStart, stats = Stats(), block = BlockStats() });
            Save("transactions.json", completed); Console.WriteLine($"{arm} context={length} rows={rows} accepted={accepted} trial={trial} ms={totalWall:F3}");
        }
    }
}
Console.WriteLine("PASS target block " + args[0]);
