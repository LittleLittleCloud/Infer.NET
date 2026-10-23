using System.Threading.Channels;
namespace DeepInfer.Runtime;

// One asynchronous network reader and one GPU consumer. Two queued suffix batches,
// one being received and one being consumed. Join the reader before releasing its wire.
public static class SplitReceivePump
{
    public static async Task Run<T>(IAsyncEnumerable<T> batches,
        Func<T, CancellationToken, ValueTask> consume, CancellationToken ct)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var channel = Channel.CreateBounded<T>(new BoundedChannelOptions(2)
        { SingleWriter = true, SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
        var reader = Task.Run(async () =>
        {
            try
            {
                await foreach (var batch in batches.WithCancellation(stop.Token))
                    await channel.Writer.WriteAsync(batch, stop.Token);
                channel.Writer.TryComplete();
            }
            catch (Exception ex) { channel.Writer.TryComplete(ex); throw; }
        }, CancellationToken.None);
        try
        {
            await foreach (var batch in channel.Reader.ReadAllAsync(stop.Token))
            { stop.Token.ThrowIfCancellationRequested(); await consume(batch, stop.Token); }
            await reader;
        }
        catch
        {
            stop.Cancel();
            try { await reader; } catch { }
            throw;
        }
    }
}
