using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

public sealed class ModelTests
{
    [Fact] public void CpuGemvHandlesNonVectorAlignedShape()
    {
        using var device = new CpuMatVecDevice();
        var weights = Enumerable.Range(0, 53 * 77).Select(i => MathF.Sin(i)).ToArray();
        var input = Enumerable.Range(0, 77).Select(i => MathF.Cos(i)).ToArray();
        using var matrix = device.Upload(weights, 53, 77);
        var output = new float[53]; matrix.Multiply(input, output);
        for (var row = 0; row < 53; row++)
        {
            var expected = 0f; for (var column = 0; column < 77; column++) expected += weights[row * 77 + column] * input[column];
            Assert.InRange(MathF.Abs(output[row] - expected), 0, 0.0001f);
        }
        Assert.Throws<ArgumentException>(() => matrix.Multiply(new float[2], output));
    }
    [Fact] public void QwenArgMaxPreservesTieBreakingAndRejectsNonFiniteLogits()
    {
        Assert.Equal(1, Qwen36Model.ArgMax([1f, 3f, 3f]));
        Assert.Throws<InvalidOperationException>(() => Qwen36Model.ArgMax([0f, float.NaN]));
        Assert.Throws<InvalidOperationException>(() => Qwen36Model.ArgMax([float.PositiveInfinity]));
    }
}
