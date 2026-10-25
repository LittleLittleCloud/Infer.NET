using DeepInfer.Backends.Split;
using Xunit;

namespace DeepInfer.Tests;

public sealed class Q8StateLayoutTests
{
    [Fact]
    public void QuantizedStateUsesBlockMetadataAndKeepsRecurrenceFp32()
    {
        Assert.Equal(544, CudaPrefixSession.KvTokenBytes(1));
        Assert.Equal(131072 * 544, CudaPrefixSession.StateBytes(3, 0, 131072, 1));
        Assert.Equal(131072 * 544, CudaPrefixSession.StateBytes(3, 1, 131072, 1));
        Assert.Equal(32 * 128 * 128 * 4, CudaPrefixSession.StateBytes(0, 0, 131072, 1));
        Assert.Equal(3 * 8192 * 4, CudaPrefixSession.StateBytes(0, 1, 131072, 1));
        Assert.Throws<ArgumentException>(() => CudaPrefixSession.KvTokenBytes(3));
    }
}
