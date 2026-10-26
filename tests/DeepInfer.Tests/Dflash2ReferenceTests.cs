using Xunit;
namespace DeepInfer.Tests;

// CPU-only hand-computed checks of the independent reference primitives that the opt-in GPU gates rely on.
public class Dflash2ReferencePrimitiveTests
{
    [Fact]
    public void RmsNormUsesTheDirectScaleNotOnePlusScale()
    {
        var x = new float[] { 3, 4 }; var scale = new float[] { 2, 0.5f };
        var r = Dflash2CpuReference.Rms(x, scale);
        var inv = 1 / MathF.Sqrt(12.5f + 1e-6f);
        Assert.Equal(3 * inv * 2f, r[0], 3); Assert.Equal(4 * inv * 0.5f, r[1], 3);
    }

    [Fact]
    public void DynamicConvolutionIsCausalWithinTheBlockAndUsesTheCurrentRowCoefficients()
    {
        const int H = 2048;
        var hidden = new[] { Enumerable.Repeat(1f, H).ToArray(), Enumerable.Repeat(10f, H).ToArray(), Enumerable.Repeat(100f, H).ToArray() };
        var dyn = new float[3][];
        for (var t = 0; t < 3; t++) { dyn[t] = new float[512]; for (var i = 0; i < 512; i++) dyn[t][i] = 0.5f * (t + 1); }
        var baseKernel = new float[2 * 2 * H];
        for (var which = 0; which < 2; which++) for (var o = 0; o < 2; o++) for (var j = 0; j < H; j++) baseKernel[(which * 2 + o) * H + j] = (which + 1) * 0.25f * (o + 1);
        var r0 = Dflash2CpuReference.DynamicConv(hidden, dyn, 0, baseKernel);
        // which=0: base offset0 = 0.25, offset1 = 0.5.
        Assert.Equal((0.25f + 0.5f) * 1, r0[0][7], 3);                                  // row 0 sees no previous row
        Assert.Equal((0.25f + 1.0f) * 10 + (0.5f + 1.0f) * 1, r0[1][2047], 3);
        Assert.Equal((0.25f + 1.5f) * 100 + (0.5f + 1.5f) * 10, r0[2][0], 3);
        var r1 = Dflash2CpuReference.DynamicConv(hidden, dyn, 1, baseKernel);
        Assert.Equal((0.5f + 0.5f) * 1, r1[0][3], 3);                                    // which=1 base offset0 = 0.5
    }

    [Fact]
    public void RopePairsFirstAndSecondHalfAndLeavesPositionZeroUnchanged()
    {
        var v = Enumerable.Range(0, 128).Select(i => (float)i + 1).ToArray(); var original = (float[])v.Clone();
        Dflash2CpuReference.Rope(v, 0, 0); Assert.Equal(original, v);
        Dflash2CpuReference.Rope(v, 0, 3);
        var angle = 3.0; var (sin, cos) = MathF.SinCos((float)angle);
        Assert.Equal(original[0] * cos - original[64] * sin, v[0], 3);
        Assert.Equal(original[0] * sin + original[64] * cos, v[64], 3);
    }
}
