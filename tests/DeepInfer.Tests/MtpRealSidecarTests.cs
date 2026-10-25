using DeepInfer.Backends.Models;
using Xunit;
namespace DeepInfer.Tests;

public sealed class MtpArtifactFactAttribute : FactAttribute
{
    public MtpArtifactFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DEEPINFER_MTP_SIDECAR_DIR")))
            Skip = "Requires the authenticated official sidecar; no GPU is used.";
    }
}

public class MtpRealSidecarTests
{
    [MtpArtifactFact]
    [Trait("Category", "MtpArtifact")]
    public void AuthenticatedOfficialConverterOutputLoadsWithTheNativeTensorLayoutWithoutGpu()
    {
        using var weights = Qwen36MtpWeights.Load(Environment.GetEnvironmentVariable("DEEPINFER_MTP_SIDECAR_DIR")!);
        Assert.Equal(Qwen36MtpGeometry.Official, weights.Geometry);
        Assert.Equal(20, Qwen36MtpNames.Layout(weights.Geometry).Length);
        Assert.Equal("mtp.layers.0.mlp.experts.down_proj", Qwen36MtpNames.ExpertDown);
        Assert.Equal(536_870_912, weights.TensorBytes(Qwen36MtpNames.ExpertDown));
        Assert.Equal(1_689_303_040L, Qwen36MtpNames.Layout(weights.Geometry).Sum(t => weights.TensorBytes(t.Name)));
        foreach (var norm in Qwen36MtpNames.Norms)
            Assert.All(weights.ReadNorm(norm), v => Assert.True(float.IsFinite(v)));
    }
}
