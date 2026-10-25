using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
namespace DeepInfer.Backends.Mtp;

public enum MtpReplayMode { Batched, Serial }

// Native Metal target. Batched replay is the accelerated path; Serial replay is a diagnostic oracle only.
internal sealed class MetalMtpTarget(Qwen36Gpu graph, MtpReplayMode replay, ulong request) : IMtpTarget
{
    private ulong snapshot; private int snapshotPosition = -1; private int position;
    public int Prefill(int[] prompt, Action<int[], float[]> sink, CancellationToken ct)
    {
        graph.PrepareVerify();
        var next = graph.PrefillWithHidden(prompt, ct, (_, tokens, hidden) => sink(tokens, hidden));
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
    public (int[] Argmax, float[] Hidden) Verify(int[] tokens) { var r = graph.Verify(tokens); position += tokens.Length; return r; }
    public void RestoreAndReplay(int[] committed)
    {
        if (snapshot == 0 || snapshotPosition < 0) throw new InvalidOperationException("No snapshot to restore.");
        graph.SnapshotRestore(snapshot, request); position = snapshotPosition;
        if (replay == MtpReplayMode.Batched) Verify(committed);
        else foreach (var token in committed) Forward(token, position);
    }
    public void Release() { if (snapshot != 0) { graph.SnapshotRelease(snapshot); snapshot = 0; } }
}

internal sealed class MetalMtpDrafter(MtpDevice device) : IMtpDrafter
{
    public int ValidRows => device.ValidRows;
    public void Reset() => device.Reset();
    public void EnsureCapacity(int tokens) => device.Reserve(tokens);
    public int CatchUp(int[] tokens, float[] hidden, int rows, bool classify)
    {
        var next = -1;
        for (var offset = 0; offset < rows; offset += 32)
        {
            var count = Math.Min(32, rows - offset);
            var last = classify && offset + count == rows;
            var result = device.Run(tokens[offset..(offset + count)], hidden[(offset * 2048)..((offset + count) * 2048)], device.ValidRows, true, last);
            if (last) next = result.Next;
            LastOutput = result.Output;
        }
        return next;
    }
    public float[]? LastOutput { get; private set; }
    // Second draft: feed the first draft token with the first draft step's post-mtp.norm output as hidden.
    public int Provisional(int token)
    {
        var result = device.Run([token], null, device.ValidRows, false, true);
        return result.Next;
    }
}

// Public handle for one MTP request: owns the request snapshot and exposes the engine.
public sealed class MetalMtpGeneration : IDisposable
{
    private readonly MetalMtpTarget target;
    public MtpGeneration Engine { get; }
    internal MetalMtpGeneration(MetalMtpTarget target, MetalMtpDrafter drafter, MtpOptions options, int maxTokens, Func<int, bool> isEnd, Func<double> clock)
    { this.target = target; Engine = new(target, drafter, options, maxTokens, isEnd, clock); }
    public void Dispose() => target.Release();
}
