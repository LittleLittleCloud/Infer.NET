using DeepInfer.Backends.Models;
using Xunit;
namespace DeepInfer.Tests;

public sealed class Dflash2ArtifactFactAttribute : FactAttribute
{
    public Dflash2ArtifactFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_DIR")))
            Skip = "Requires the authenticated DFlash2 draft directory (DEEPINFER_DFLASH2_DIR); no GPU is used.";
    }
}

public class Dflash2SchemaTests
{
    [Fact]
    public void ExpectedTensorSetHasTheTrainedSelectorAndBothDynamicConvolutionsPerLayer()
    {
        var names = Dflash2Names.Matrices.Select(m => m.Name).Concat(Dflash2Names.Vectors.Select(v => v.Name)).ToArray();
        Assert.Equal(96, names.Length); Assert.Equal(names.Length, names.Distinct().Count());
        Assert.Equal(58, Dflash2Names.Matrices.Count); Assert.Equal(38, Dflash2Names.Vectors.Count);
        Assert.Contains("candidate_selector.predecessor_codebook", names);
        Assert.Contains("candidate_selector.successor_codebook", names);
        Assert.Contains("candidate_selector.hidden_projection.weight", names);
        for (var l = 0; l < 6; l++)
        {
            Assert.Contains($"layers.{l}.attention_conv.kernel_projection.weight", names);
            Assert.Contains($"layers.{l}.mlp_conv.base_kernel", names);
        }
        Assert.Equal([2, 2, 2048], Dflash2Names.Vectors.Single(v => v.Name == "layers.3.attention_conv.base_kernel").Shape);
        Assert.Equal([512, 2048], Dflash2Names.Matrices.Single(m => m.Name == "layers.0.mlp_conv.kernel_projection.weight").Shape);
        Assert.Equal([2048, 16384], Dflash2Names.Matrices.Single(m => m.Name == "fc.weight").Shape);
        Assert.Equal([1, 6, 11, 16, 22, 27, 32, 37], Dflash2Geometry.TapLayers);
    }

    [Fact]
    public void PinsMatchTheIssueSpecification()
    {
        Assert.Equal("51ef7b6923ad6c14cb1bb41c37a9041446496ab6", Dflash2Pins.Revision);
        Assert.Equal(1_052_513_456, Dflash2Pins.WeightBytes);
        Assert.Equal("d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d", Dflash2Pins.WeightSha256);
    }

    [Fact]
    public void MissingFilesAndPinMismatchesAreHardErrors()
    {
        var dir = Path.Combine(Path.GetTempPath(), "dflash2-loader-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            Assert.Throws<FileNotFoundException>(() => Dflash2Weights.Load(dir));
            File.WriteAllText(Path.Combine(dir, "config.json"), "{}"); File.WriteAllBytes(Path.Combine(dir, "model.safetensors"), new byte[16]);
            Assert.ThrowsAny<Exception>(() => Dflash2Weights.Load(dir));
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public void ConfigGeometryDriftIsRejectedBeforeTensorsAreRead()
    {
        var dir = Path.Combine(Path.GetTempPath(), "dflash2-config-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            File.WriteAllText(Path.Combine(dir, "config.json"), """{"architectures":["Other"],"hidden_size":2048}"""); File.WriteAllBytes(Path.Combine(dir, "model.safetensors"), new byte[16]);
            Assert.Throws<InvalidDataException>(() => Dflash2Weights.Load(dir, new Dflash2LoadOptions(VerifyHash: false, RequirePins: false)));
        }
        finally { Directory.Delete(dir, true); }
    }
}

public class Dflash2RealArtifactTests
{
    [Dflash2ArtifactFact]
    [Trait("Category", "Dflash2Artifact")]
    public void AuthenticatedDraftPassesStrictSchemaWithoutGpu()
    {
        var hash = Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_HASH") == "1";
        using var weights = Dflash2Weights.Load(Environment.GetEnvironmentVariable("DEEPINFER_DFLASH2_DIR")!, new Dflash2LoadOptions(VerifyHash: hash, RequirePins: hash));
        Assert.Equal(Dflash2Pins.WeightBytes, weights.WeightBytes);
        if (hash) Assert.Equal(Dflash2Pins.WeightSha256, weights.WeightSha256);
        foreach (var (name, shape) in Dflash2Names.Vectors.Take(2))
            Assert.All(weights.ReadVector(name), v => Assert.True(float.IsFinite(v)));
        Assert.Equal(2L * 2048 * 16384, weights.TensorBytes("fc.weight"));
    }
}
