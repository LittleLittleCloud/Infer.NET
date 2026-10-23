using System.Threading.Channels;
namespace DeepInfer.Runtime;

// One synchronous GPU producer, one network sender. At most two queued frames
// plus the producer's current frame. Joining on every exit keeps reset/disposal
// from racing an in-flight GPU call. No retries or background work after return.
public static class SplitFramePump
{
    public static async Task Run(IEnumerable<SplitWire.Frame> frames,
        Func<SplitWire.Frame, CancellationToken, ValueTask> send, CancellationToken ct)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var channel = Channel.CreateBounded<SplitWire.Frame>(new BoundedChannelOptions(2)
        { SingleWriter = true, SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var producer = Task.Run(async () =>
        {
            try
            {
                using var iterator = frames.GetEnumerator();
                while (true)
                {
                    stop.Token.ThrowIfCancellationRequested();
                    if (!iterator.MoveNext()) break;
                    await channel.Writer.WriteAsync(iterator.Current, stop.Token);
                }
                channel.Writer.TryComplete();
            }
            catch (Exception ex) { channel.Writer.TryComplete(ex); throw; }
        }, CancellationToken.None);
        try
        {
            await foreach (var frame in channel.Reader.ReadAllAsync(stop.Token))
                await send(frame, stop.Token);
            await producer;
        }
        catch
        {
            stop.Cancel();
            // Preserve the original send/producer failure after ensuring GPU ownership ends.
            try { await producer; } catch { }
            throw;
        }
    }
}
