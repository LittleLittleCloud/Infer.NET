using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public class SplitFramePumpTests
{
    private static SplitWire.Frame Frame(int n) => new(SplitWire.Hidden, 20, 0, n, 1, []);
    [Fact]
    public async Task ChunkBundlesStayOrderedAndBoundedOnSendFailure()
    {
        var full = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var produced = 0; var disposed = false; var sent = new List<int>();
        IEnumerable<SplitWire.Frame[]> Batches()
        {
            try { for (var i=0;i<100;i++) { if(Interlocked.Increment(ref produced)==4)full.SetResult(); yield return [Frame(i*2),Frame(i*2+1)]; } }
            finally { disposed=true; }
        }
        var run=SplitFramePump.RunBatches(Batches(),async (f,ct)=>
        {
            sent.Add(f.Position);
            if(f.Position==1) { await full.Task.WaitAsync(TimeSpan.FromSeconds(5),ct); Assert.Equal(4,Volatile.Read(ref produced)); throw new IOException("KV send failed"); }
        },CancellationToken.None);
        await Assert.ThrowsAsync<IOException>(()=>run.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(new[]{0,1},sent);Assert.True(disposed);Assert.Equal(4,produced);
    }
    [Fact]
    public async Task PreservesOrderAcrossBoundedProducerAndAsyncSender()
    {
        var actual = new List<int>();
        await SplitFramePump.Run(Enumerable.Range(0, 100).Select(Frame), async (f, ct) =>
        { await Task.Yield(); ct.ThrowIfCancellationRequested(); actual.Add(f.Position); }, CancellationToken.None);
        Assert.Equal(Enumerable.Range(0, 100), actual);
    }
    [Fact]
    public async Task SendFailureJoinsBackpressuredProducerAndDisposesIterator()
    {
        var full = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var produced = 0; var disposed = false;
        IEnumerable<SplitWire.Frame> Frames()
        {
            try { for (var i = 0; i < 100; i++) { if (Interlocked.Increment(ref produced) == 4) full.SetResult(); yield return Frame(i); } }
            finally { disposed = true; }
        }
        var run = SplitFramePump.Run(Frames(), async (_, ct) =>
        { await full.Task.WaitAsync(TimeSpan.FromSeconds(5), ct); Assert.Equal(4, Volatile.Read(ref produced)); throw new IOException("send failed"); }, CancellationToken.None);
        var error = await Assert.ThrowsAsync<IOException>(() => run.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal("send failed", error.Message); Assert.True(disposed); Assert.Equal(4, produced);
    }
    [Fact]
    public async Task CancellationJoinsProducerBeforeReturning()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var disposed = false;
        IEnumerable<SplitWire.Frame> Frames()
        {
            try { for (var i = 0; i < 100; i++) yield return Frame(i); }
            finally { disposed = true; }
        }
        using var cancel = new CancellationTokenSource();
        var run = SplitFramePump.Run(Frames(), async (_, ct) =>
        { entered.SetResult(); await Task.Delay(Timeout.InfiniteTimeSpan, ct); }, cancel.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5)); cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.True(disposed);
    }
    [Fact]
    public async Task ProducerFailurePropagatesWithoutRetry()
    {
        var disposed = false; var produced = 0;
        IEnumerable<SplitWire.Frame> Frames()
        {
            try { produced++; yield return Frame(0); throw new InvalidOperationException("GPU failed"); }
            finally { disposed = true; }
        }
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => SplitFramePump.Run(Frames(), (_, _) => ValueTask.CompletedTask, CancellationToken.None));
        Assert.Equal("GPU failed", error.Message); Assert.True(disposed); Assert.Equal(1, produced);
    }
}
