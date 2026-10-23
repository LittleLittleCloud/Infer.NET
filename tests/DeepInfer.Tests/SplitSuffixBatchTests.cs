using System.Runtime.InteropServices;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public class SplitSuffixBatchTests
{
    private static Queue<SplitWire.Frame> Frames(int tokens,bool incremental)
    {
        var frames=new Queue<SplitWire.Frame>();
        for(var pos=0;pos<tokens;pos+=32)
        {
            var count=Math.Min(32,tokens-pos);var hidden=Enumerable.Range(pos*2048,count*2048).Select(x=>(float)x).ToArray();
            frames.Enqueue(new(SplitWire.Hidden,20,0,pos,count,MemoryMarshal.AsBytes(hidden.AsSpan()).ToArray()));
            if(incremental)for(var l=3;l<20;l+=4)for(var t=0;t<2;t++)
            { var data=Enumerable.Range(pos*512*4,count*512*4).Select(x=>(byte)(x+l+t)).ToArray();frames.Enqueue(new(SplitWire.KvRange,l,t,pos,count,data)); }
        }
        frames.Enqueue(new(SplitWire.End,20,0,tokens,0,[]));return frames;
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ReassemblesIrregularTailWithoutReadingHandoff(bool incremental)
    {
        var frames=Frames(129,incremental);var batches=new List<SplitSuffixBatch>();
        await foreach(var b in SplitSuffixBatches.Read(_=>Task.FromResult(frames.Dequeue()),129,128,incremental,CancellationToken.None))batches.Add(b);
        Assert.Equal(new[]{128,1},batches.Select(x=>x.Count));Assert.Equal(new[]{0,128},batches.Select(x=>x.Position));
        Assert.Single(frames);Assert.Equal(SplitWire.End,frames.Peek().Kind);
        foreach(var b in batches)
        {
            Assert.Equal(Enumerable.Range(b.Position*2048,b.Count*2048).Select(x=>(float)x),b.Hidden);
            if(incremental)for(var l=3;l<20;l+=4)for(var t=0;t<2;t++)Assert.Equal(Enumerable.Range(b.Position*512*4,b.Count*512*4).Select(x=>(byte)(x+l+t)),b.Kv![(l/4)*2+t]);
            else Assert.Null(b.Kv);
        }
    }
    [Theory]
    [InlineData("position")]
    [InlineData("kv")]
    [InlineData("nonfinite")]
    public async Task RejectsBrokenBatchBeforeConsumer(string fault)
    {
        var frames=Frames(33,true).ToArray();
        if(fault=="position")frames[0]=frames[0] with{Position=1};
        if(fault=="kv")frames[1]=frames[1] with{Tensor=1};
        if(fault=="nonfinite")BitConverter.GetBytes(float.NaN).CopyTo(frames[0].Data,0);
        var index=0;var consumed=0;
        var source=SplitSuffixBatches.Read(_=>Task.FromResult(frames[index++]),33,128,true,CancellationToken.None);
        await Assert.ThrowsAsync<InvalidDataException>(()=>SplitReceivePump.Run(source,(_,_)=>{consumed++;return ValueTask.CompletedTask;},CancellationToken.None));Assert.Equal(0,consumed);
    }
}
