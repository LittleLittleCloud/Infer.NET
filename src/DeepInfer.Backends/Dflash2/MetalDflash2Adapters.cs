using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
namespace DeepInfer.Backends.Dflash2;

public enum Dflash2ReplayMode { Batched, Serial }

// Native Metal target. Batched replay is the accelerated path; Serial replay is a diagnostic oracle only.
internal sealed class MetalDflash2Target(Qwen36Gpu graph, Dflash2ReplayMode replay, ulong request) : IDflash2Target
{
    private ulong snapshot; private int snapshotPosition = -1; private int position;
    public int Prefill(int[] prompt, Action<int, int> chunkSink, CancellationToken ct)
    {
        graph.PrepareVerify();
        var next = graph.PrefillChunks(prompt, ct, chunkSink);
        position = prompt.Length;
        return next;
    }
    public void EnsureCapacity(int tokens) { if (tokens > graph.Stats().Capacity) graph.Grow(tokens); }
    public int Forward(int token, int pos)
    {
        if (pos != position) throw new InvalidOperationException($"Target position {position} differs from requested decode position {pos}.");
        var next = graph.Forward(token, pos, true, [], null, false); position++; return next;
    }
    public void Snapshot()
    {
        if (snapshot == 0) snapshot = graph.SnapshotCreate();
        graph.SnapshotCapture(snapshot, request); snapshotPosition = position;
    }
    public int[] Verify(int[] tokens) { var r = graph.Verify(tokens, null, false); position += tokens.Length; return r.Argmax; }
    public void RestoreAndReplay(int[] committed)
    {
        if (snapshot == 0 || snapshotPosition < 0) throw new InvalidOperationException("No snapshot to restore.");
        graph.SnapshotRestore(snapshot, request); position = snapshotPosition;
        if (replay == Dflash2ReplayMode.Batched) Verify(committed);
        else foreach (var token in committed) Forward(token, position);
    }
    public void Release() { if (snapshot != 0) { graph.SnapshotRelease(snapshot); snapshot = 0; } }
}

internal sealed class MetalDflash2Drafter(Dflash2Device device) : IDflash2Drafter
{
    public int ContextEnd => (int)device.State().ContextEnd;
    public void Reset() => device.Reset();
    public void Ingest(int tapRow, int rows, int firstPosition) => device.Ingest(tapRow, rows, firstPosition);
    public int[] Propose(int[] block) => device.Propose(block).Tokens;
}

// Public handle for one DFlash2 request: owns the request snapshot and exposes the engine.
public sealed class MetalDflash2Generation : IDisposable
{
    private readonly MetalDflash2Target target;
    public Dflash2Generation Engine { get; }
    internal MetalDflash2Generation(MetalDflash2Target target, MetalDflash2Drafter drafter, Dflash2Options options, int maxTokens, Func<int, bool> isEnd, Func<double> clock)
    { this.target = target; Engine = new(target, drafter, options, maxTokens, isEnd, clock); }
    public void Dispose() => target.Release();
}
