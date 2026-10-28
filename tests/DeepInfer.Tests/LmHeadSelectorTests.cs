using DeepInfer.Backends.Models;
using DeepInfer.Core;
using Xunit;
namespace DeepInfer.Tests;

[CollectionDefinition("LM head environment", DisableParallelization = true)]
public sealed class LmHeadEnvironmentCollection;

[Collection("LM head environment")]
public sealed class LmHeadSelectorTests
{
    [Theory]
    [InlineData("")] [InlineData("0")] [InlineData("1")]
    [InlineData("auto")] [InlineData("tile8")] [InlineData("TILE4")]
    public void InvalidModeRejectsBeforeNativeDeviceOrWeightLoading(string value)
    {
        WithEnvironment("DEEPINFER_METAL_LMHEAD", value, () =>
        {
            var failure = Assert.Throws<ArgumentException>(() => Create());
            Assert.Contains("DEEPINFER_METAL_LMHEAD", failure.Message);
        });
    }

    [Theory]
    [InlineData("DEEPINFER_METAL_SERIAL")]
    [InlineData("DEEPINFER_METAL_EXPERT_UNFUSED")]
    [InlineData("DEEPINFER_METAL_GENERIC")]
    [InlineData("DEEPINFER_METAL_HOST_EXECUTOR")]
    public void ExplicitCandidateCannotSilentlyUseHostBaseline(string setting)
    {
        WithEnvironment("DEEPINFER_METAL_LMHEAD", "tile4", () =>
            WithEnvironment(setting, "1", () =>
            {
                var failure = Assert.Throws<NotSupportedException>(() => Create());
                Assert.Contains("native GPU graph", failure.Message);
            }));
    }

    private static Qwen36Model Create() =>
        new(new ModelInfo(Protocol.Qwen36Model, "mlx-qwen36-affine", false), "unused", "unused");

    private static void WithEnvironment(string name, string value, Action action)
    {
        var saved = Environment.GetEnvironmentVariable(name);
        var forced = Environment.GetEnvironmentVariable("DEEPINFER_GRAPH_FORCE_ROUTES");
        try
        {
            Environment.SetEnvironmentVariable("DEEPINFER_GRAPH_FORCE_ROUTES", null);
            Environment.SetEnvironmentVariable(name, value);
            action();
        }
        finally
        {
            Environment.SetEnvironmentVariable(name, saved);
            Environment.SetEnvironmentVariable("DEEPINFER_GRAPH_FORCE_ROUTES", forced);
        }
    }
}
