using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

public sealed class TimingTests
{
    [Theory]
    [InlineData(0,0,0)]
    [InlineData(1,0,0)]
    [InlineData(2,1,4)]
    [InlineData(128,127,508)]
    public void DecodeCountsOnlyIntervalsAfterFirstOutput(int outputs,int intervals,double rate)
    {
        var result=InferenceTimings.Create(1800,1850,250,outputs);
        Assert.Equal(intervals,result.DecodeTokens); Assert.Equal(rate,result.DecodeTokensPerSecond);
        Assert.Equal(1800,result.PrefillMs); Assert.Equal(1850,result.FirstTokenMs);
        Assert.Equal(intervals==0?0:250,result.DecodeMs);
    }
    [Theory]
    [InlineData(-1,0,0,1)]
    [InlineData(0,-1,0,1)]
    [InlineData(0,0,-1,1)]
    [InlineData(0,0,0,-1)]
    public void InvalidTimingsAreRejected(double prefill,double first,double decode,int outputs)
        => Assert.Throws<ArgumentOutOfRangeException>(()=>InferenceTimings.Create(prefill,first,decode,outputs));
    [Fact] public void ZeroElapsedDoesNotProduceInfiniteRate()
        =>Assert.Equal(0,InferenceTimings.Create(0,0,0,2).DecodeTokensPerSecond);
}
