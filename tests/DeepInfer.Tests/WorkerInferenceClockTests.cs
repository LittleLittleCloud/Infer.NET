using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

public class WorkerInferenceClockTests
{
    [Fact]
    public void BlockTokensShareReadinessAndCancelledTransportKeepsActualWorkerCounts()
    {
        var clock = new WorkerInferenceClock("request", "build", 21, "metal-local", () => 80);
        clock.PrefillDone(10); clock.Ready(1, 11); clock.Ready(8, 71);
        var partial = clock.Snapshot("cancelled", null);
        Assert.Equal(9, partial.OutputTokens); Assert.Equal(8, partial.DecodeIntervals); Assert.Equal(60, partial.DecodeMs);
        Assert.Equal("cancelled", partial.Outcome); Assert.True(partial.TimingsComplete);
        // Delivery failure cannot be renamed successful completion or invent client TTFT.
        Assert.Null(partial.FinishReason); Assert.Equal(80, partial.WorkerMs);
    }
    [Fact]
    public void CancelledPrefillHasMissingClockAndEosWithoutOutputsDoesNotInventDecodeIntervals()
    {
        var clock = new WorkerInferenceClock("request", "build", 21, "metal-local", () => 12);
        Assert.False(clock.Snapshot("cancelled", null).TimingsComplete);
        Assert.Null(clock.Snapshot("cancelled", null).PrefillMs);
        clock.PrefillDone(11); var eos = clock.Snapshot("completed", "eos");
        Assert.Equal(0, eos.OutputTokens); Assert.Equal(0, eos.DecodeIntervals); Assert.Equal(0, eos.DecodeMs);
        Assert.True(eos.TimingsComplete);
    }
}
