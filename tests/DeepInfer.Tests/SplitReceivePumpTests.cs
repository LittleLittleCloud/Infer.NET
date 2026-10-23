using System.Runtime.CompilerServices;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public class SplitReceivePumpTests
{
    [Fact]
    public async Task ReceivesInOrderAndStopsBeforeFollowingProtocolFrames()
    {
        var received=0;var consumed=new List<int>();
        async IAsyncEnumerable<int> Batches([EnumeratorCancellation] CancellationToken ct=default)
        { for(var i=0;i<100;i++){ct.ThrowIfCancellationRequested();await Task.Yield();received++;yield return i;} }
        await SplitReceivePump.Run(Batches(),(x,_)=>{consumed.Add(x);return ValueTask.CompletedTask;},CancellationToken.None);
        Assert.Equal(100,received);Assert.Equal(Enumerable.Range(0,100),consumed);
    }
    [Fact]
    public async Task ConsumerFailureJoinsBoundedReader()
    {
        var full=new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);var produced=0;var disposed=false;
        async IAsyncEnumerable<int> Batches([EnumeratorCancellation] CancellationToken ct=default)
        {
            try { for(var i=0;i<100;i++){ct.ThrowIfCancellationRequested();await Task.Yield();if(Interlocked.Increment(ref produced)==4)full.SetResult();yield return i;} }
            finally {disposed=true;}
        }
        var run=SplitReceivePump.Run(Batches(),async (_,ct)=>{await full.Task.WaitAsync(TimeSpan.FromSeconds(5),ct);Assert.Equal(4,Volatile.Read(ref produced));throw new IOException("GPU consumer failed");},CancellationToken.None);
        var error=await Assert.ThrowsAsync<IOException>(()=>run.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal("GPU consumer failed",error.Message);Assert.True(disposed);Assert.Equal(4,produced);
    }
    [Fact]
    public async Task CancellationInterruptsAndJoinsPendingRead()
    {
        var entered=new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);var disposed=false;
        async IAsyncEnumerable<int> Batches([EnumeratorCancellation] CancellationToken ct=default)
        { try {entered.SetResult();await Task.Delay(Timeout.InfiniteTimeSpan,ct);yield return 0;} finally {disposed=true;} }
        using var stop=new CancellationTokenSource();
        var run=SplitReceivePump.Run(Batches(),(_,_)=>ValueTask.CompletedTask,stop.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(()=>run.WaitAsync(TimeSpan.FromSeconds(5)));Assert.True(disposed);
    }
    [Fact]
    public async Task ReadFailureIsNotRetriedAndDisposesIterator()
    {
        var disposed=false;var reads=0;
        async IAsyncEnumerable<int> Batches()
        {try {await Task.Yield();reads++;yield return 0;throw new InvalidDataException("bad frame");}finally {disposed=true;}}
        var error=await Assert.ThrowsAsync<InvalidDataException>(()=>SplitReceivePump.Run(Batches(),(_,_)=>ValueTask.CompletedTask,CancellationToken.None));
        Assert.Equal("bad frame",error.Message);Assert.Equal(1,reads);Assert.True(disposed);
    }
}
