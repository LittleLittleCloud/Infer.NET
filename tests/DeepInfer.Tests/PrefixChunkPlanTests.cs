using DeepInfer.Runtime;
using Xunit;

namespace DeepInfer.Tests;

public sealed class PrefixChunkPlanTests
{
    [Fact]
    public void ShortFirstTileKeepsAllTokensInOrder()
    {
        var plan = PrefixChunkPlan.Create(8192, 1024, 128).ToArray();
        Assert.Equal((0, 128), plan[0]);
        Assert.Equal((7296, 896), plan[^1]);
        Assert.Equal(8192, plan.Sum(x => x.Count));
        for (var i = 1; i < plan.Length; i++)
            Assert.Equal(plan[i - 1].Position + plan[i - 1].Count, plan[i].Position);
        Assert.All(plan.Skip(1).SkipLast(1), x => Assert.Equal(1024, x.Count));
    }

    [Fact]
    public void DefaultAndShortPromptsKeepTheirOriginalShape()
    {
        Assert.Equal(8, PrefixChunkPlan.Create(8192, 1024, 0).Count());
        Assert.Equal([(0, 17)], PrefixChunkPlan.Create(17, 1024, 128).ToArray());
        Assert.Throws<ArgumentException>(() => PrefixChunkPlan.Create(100, 32, 128).ToArray());
    }
}
