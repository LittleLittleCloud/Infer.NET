using System.Runtime.InteropServices;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using Xunit;
namespace DeepInfer.Tests;

// Hosted macOS can expose a paravirtual GPU that executes GEMV but cannot create
// MTLResidencySet. Keep this hardware test mandatory on physical Macs; don't hide
// a real residency allocation failure with a catch-and-return in the test body.
public sealed class ResidencyFactAttribute : FactAttribute
{
    public ResidencyFactAttribute()
    {
        if (!OperatingSystem.IsMacOS()) Skip = "Metal residency requires a physical macOS GPU.";
        else if (new DeepInfer.Backends.MetalBackend().DiscoverDevices().Any(d => d.Name.Contains("Paravirtual", StringComparison.OrdinalIgnoreCase)))
            Skip = "Paravirtual Metal GPU does not implement residency sets; validated separately on physical M5 Pro.";
    }
}

public class QuantizedTests
{
    [Fact]
    public void PackedCudaWeightsRejectSerialExecutionBeforeLoadingNativeLibrary()
    {
        Assert.Throws<ArgumentException>(() => new DeepInfer.Backends.Split.CudaPrefixSession("unused", 32, layerBatch: false, packedWeights: true));
    }

    [ResidencyFact]
    public void ExplicitResidencyCrossesManagedNativeBoundaryAndReleasesWeights()
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var enabled = new QuantizedMetal(residency: true, residencyRefreshMilliseconds: 20);
        using var disabled = new QuantizedMetal(residency: false);
        using var ordinary = disabled.Allocate(1, 64, 64, 8);
        Assert.Equal(0UL, disabled.ResidencyBytes);
        using (var matrix = enabled.Allocate(1, 64, 64, 8))
        {
            Assert.True(enabled.ResidencyBytes >= 4096);
            enabled.PrepareResidency(); enabled.PrepareResidency();
            Populate(matrix, 3); Populate(ordinary, 3);
            float[] input = Enumerable.Repeat(.125f, 64).ToArray(), a = new float[64], b = new float[64];
            matrix.MultiplyExperts(input, a, [0], false);
            ordinary.MultiplyExperts(input, b, [0], false);
            Assert.Equal(b, a);
        }
        Assert.Equal(0UL, enabled.ResidencyBytes);
    }

    [Theory]
    [InlineData(5, 512, true)] [InlineData(8, 512, false)]
    [InlineData(5, 2048, false)] [InlineData(8, 2048, true)]
    [InlineData(5, 4096, true)] [InlineData(8, 4096, false)]
    [InlineData(5, 64, true)] [InlineData(8, 64, false)]
    [InlineData(5, 192, false)] [InlineData(8, 192, true)]
    [InlineData(5, 256, true)] [InlineData(8, 256, false)]
    public void TiledGemvMatchesIndependentScalarWithTailRowsAndEightExperts(int bits, int columns, bool batched)
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal(); using var m = device.Allocate(9, 129, columns, bits);
        Populate(m, 3);
        int[] ids = [8, 0, 6, 2, 5, 1, 7, 3];
        var x = Enumerable.Range(0, columns * (batched ? 8 : 1)).Select(i => MathF.Sin(i * .013f) * .1f).ToArray();
        var output = new float[129 * 8]; m.MultiplyExperts(x, output, ids, batched);
        for (var slot = 0; slot < 8; slot++) for (var row = 0; row < 129; row++)
        {
            double expected = 0;
            for (var c = 0; c < columns; c++) expected += (double)ScalarWeight(m, ids[slot], row, c, 3) * x[(batched ? slot * columns : 0) + c];
            Assert.InRange(Math.Abs(expected - output[slot * 129 + row]), 0, .0003);
        }
    }
    [Theory]
    [InlineData(5, 128)] [InlineData(8, 128)]
    [InlineData(5, 256)] [InlineData(8, 256)]
    public void FusedExpertsMatchScalarSwiGluAndDownAcrossChangedInputs(int bits, int columns)
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal();
        using var gate = device.Allocate(9, 192, columns, bits); using var up = device.Allocate(9, 192, columns, bits);
        using var down = device.Allocate(9, columns, 192, bits == 5 ? 8 : 5); using var shared = device.Allocate(1, columns, 64, 5);
        Populate(gate, 1); Populate(up, 2); Populate(down, 3); Populate(shared, 4);
        int[] ids = [8, 0, 6, 2, 5, 1, 7, 3];
        var input = Enumerable.Range(0, columns).Select(i => MathF.Sin(i * .13f) * .1f).ToArray();
        var sharedInput = Enumerable.Range(0, 64).Select(i => MathF.Cos(i * .07f) * .2f).ToArray();
        var output = new float[columns * 8]; var sharedOutput = new float[columns];
        for (var pass = 0; pass < 2; pass++)
        {
            input[0] += .15f; sharedInput[0] -= .2f; Array.Reverse(ids);
            device.ExpertMlp(gate, up, down, shared, input, sharedInput, ids, output, sharedOutput);
            for (var slot = 0; slot < 8; slot++)
            {
                var activation = new double[192];
                for (var row = 0; row < 192; row++)
                {
                    double g = 0, u = 0;
                    for (var c = 0; c < columns; c++)
                    {
                        g += (double)ScalarWeight(gate, ids[slot], row, c, 1) * input[c];
                        u += (double)ScalarWeight(up, ids[slot], row, c, 2) * input[c];
                    }
                    activation[row] = (g / (1 + Math.Exp(-g))) * u;
                }
                for (var row = 0; row < columns; row++)
                {
                    double expected = 0;
                    for (var c = 0; c < 192; c++) expected += ScalarWeight(down, ids[slot], row, c, 3) * activation[c];
                    Assert.InRange(Math.Abs(expected - output[slot * columns + row]), 0, .0003);
                }
            }
            for (var row = 0; row < columns; row++)
            {
                double expected = 0;
                for (var c = 0; c < 64; c++) expected += (double)ScalarWeight(shared, 0, row, c, 4) * sharedInput[c];
                Assert.InRange(Math.Abs(expected - sharedOutput[row]), 0, .0003);
            }
        }
        Assert.Equal(6, device.Dispatches); Assert.Equal(2, device.Submissions);
    }
    [Fact]
    public void InvalidFusedExpertCallDoesNotSubmitOrOverwriteOutputs()
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var d = new QuantizedMetal(); using var other = new QuantizedMetal();
        using var gate = d.Allocate(3, 64, 64, 5); using var up = d.Allocate(3, 64, 64, 5);
        using var down = d.Allocate(3, 64, 64, 8); using var shared = d.Allocate(1, 64, 64, 5);
        using var foreign = other.Allocate(3, 64, 64, 5);
        var x = new float[64]; var y = Enumerable.Repeat(-2f, 128).ToArray(); var sy = new float[64];
        Assert.Throws<ArgumentOutOfRangeException>(() => d.ExpertMlp(gate, up, down, shared, x, x, [3, 0], y, sy));
        Assert.Throws<ArgumentException>(() => d.ExpertMlp(gate, up, down, shared, new float[63], x, [1, 0], y, sy));
        Assert.Throws<ArgumentException>(() => d.ExpertMlp(gate, foreign, down, shared, x, x, [1, 0], y, sy));
        Assert.Throws<ArgumentException>(() => d.ExpertMlp(gate, gate, down, shared, x, x, [1, 0], y, sy));
        Assert.Throws<ArgumentException>(() => d.ExpertMlp(gate, up, down, shared, x, x, [1, 0], y, y.AsSpan(0, 64)));
        up.Dispose();
        Assert.Throws<ObjectDisposedException>(() => d.ExpertMlp(gate, up, down, shared, x, x, [1, 0], y, sy));
        Assert.Equal(0, d.Submissions); Assert.All(y, value => Assert.Equal(-2f, value));
    }
    private static float ScalarWeight(QuantMatrix m, int expert, int row, int column, int seed)
    {
        var group = column / 64;
        var q = (column * 7 + expert * 3 + row * 5 + seed * 11) & ((1 << m.Bits) - 1);
        return q * (((group + seed) % 3 + 1) / 512f) + ((row + group + seed) % 5 - 2) / 32f;
    }
    private static void Populate(QuantMatrix m, int seed)
    {
        var words = m.Columns * m.Bits / 32; var groups = m.Columns / 64;
        var packed = new uint[m.Experts * m.Rows * words];
        var scales = new ushort[m.Experts * m.Rows * groups]; var biases = new ushort[scales.Length];
        for (var e = 0; e < m.Experts; e++) for (var r = 0; r < m.Rows; r++)
        {
            for (var g = 0; g < groups; g++)
            {
                var at = (e * m.Rows + r) * groups + g;
                scales[at] = (ushort)(BitConverter.SingleToInt32Bits(((g + seed) % 3 + 1) / 512f) >> 16);
                biases[at] = (ushort)(BitConverter.SingleToInt32Bits(((r + g + seed) % 5 - 2) / 32f) >> 16);
            }
            for (var c = 0; c < m.Columns; c++)
            {
                uint q = (uint)((c * 7 + e * 3 + r * 5 + seed * 11) & ((1 << m.Bits) - 1));
                var bit = c * m.Bits; var at = (e * m.Rows + r) * words + bit / 32; var shift = bit % 32;
                packed[at] |= q << shift;
                if (shift + m.Bits > 32) packed[at + 1] |= q >> (32 - shift);
            }
        }
        m.Write(0, MemoryMarshal.AsBytes(packed.AsSpan()));
        m.Write(1, MemoryMarshal.AsBytes(scales.AsSpan())); m.Write(2, MemoryMarshal.AsBytes(biases.AsSpan()));
    }
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void IndependentMixedGemvsPreserveOutputsWithOneBatchWait(bool batching)
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal(batching);
        using var a = device.Allocate(1, 3, 64, 5);
        using var b = device.Allocate(3, 7, 128, 8);
        using var c = device.Allocate(2, 5, 64, 5);
        using var d = device.Allocate(1, 4, 64, 8);
        var matrices = new[] { a, b, c, d };
        for (var j = 0; j < matrices.Length; j++)
        {
            var m = matrices[j]; var groups = m.Columns / 64;
            m.Write(0, new byte[m.Experts * m.Rows * m.Columns * m.Bits / 8]);
            var scales = Enumerable.Repeat((ushort)0x3f80, m.Experts * m.Rows * groups).ToArray();
            var biases = Enumerable.Range(0, scales.Length).Select(i =>
                (ushort)(BitConverter.SingleToInt32Bits((j + 1) + i / groups * .25f) >> 16)).ToArray();
            m.Write(1, MemoryMarshal.AsBytes(scales.AsSpan())); m.Write(2, MemoryMarshal.AsBytes(biases.AsSpan()));
        }
        var input = Enumerable.Range(0, 64).Select(i => (i - 20) * .01f).ToArray();
        var batched = Enumerable.Range(0, 256).Select(i => (i - 100) * .005f).ToArray();
        QuantizedJob[] jobs = [new(a, input, new float[3]), new(b, batched, new float[14], [2, 0], true),
            new(c, input, new float[10], [1, 0]), new(d, input, new float[4])];
        for (var pass = 0; pass < 2; pass++)
        {
            input[0] += .25f; batched[0] -= .5f;
            device.MultiplyBatch(jobs);
            for (var j = 0; j < jobs.Length; j++)
            {
                var job = jobs[j]; var m = job.Matrix; var count = job.Experts?.Length ?? 1;
                for (var slot = 0; slot < count; slot++) for (var row = 0; row < m.Rows; row++)
                {
                    var expert = job.Experts?[slot] ?? 0; var expected = 0f;
                    for (var k = 0; k < m.Columns; k++)
                        expected += ((j + 1) + (expert * m.Rows + row) * .25f) * job.Input[(job.Batched ? slot * m.Columns : 0) + k];
                    Assert.InRange(MathF.Abs(expected - job.Output[slot * m.Rows + row]), 0, .0002f);
                }
            }
        }
        Assert.Equal(8, device.Dispatches);
        Assert.Equal(batching ? 2 : 8, device.Submissions);
    }
    [Fact]
    public void InvalidBatchHasNoPartialSubmissionOrOutput()
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal(); using var other = new QuantizedMetal();
        using var a = device.Allocate(1, 64, 64, 5); using var b = device.Allocate(1, 64, 64, 5);
        using var foreign = other.Allocate(1, 64, 64, 5);
        var input = new float[64]; var output = Enumerable.Repeat(-3f, 64).ToArray();
        Assert.Throws<ArgumentException>(() => device.MultiplyBatch([new(a, input, output), new(a, input, new float[64])]));
        Assert.Throws<ArgumentException>(() => device.MultiplyBatch([new(a, input, output), new(b, output, new float[64])]));
        Assert.Throws<ArgumentException>(() => device.MultiplyBatch([new(a, input, output), new(foreign, input, new float[64])]));
        Assert.Throws<ArgumentException>(() => device.MultiplyBatch([new(a, input, output), new(b, input, new float[63])]));
        Assert.Throws<ArgumentOutOfRangeException>(() => device.MultiplyBatch([new(a, input, output), new(b, input, new float[64], [1])]));
        b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => device.MultiplyBatch([new(a, input, output), new(b, input, new float[64])]));
        Assert.Equal(0, device.Submissions);
        Assert.All(output, value => Assert.Equal(-3f, value));
    }
    [Theory]
    [InlineData(5)]
    [InlineData(8)]
    public void SelectedExpertGemvMatchesScalarQuantization(int bits)
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal();
        using var matrix = device.Allocate(3, 7, 128, bits);
        var packed = new uint[3 * 7 * 128 * bits / 32];
        var scale = Enumerable.Repeat((ushort)0x3f80, 3 * 7 * 2).ToArray();
        var bias = Enumerable.Repeat((ushort)0x3f00, scale.Length).ToArray();
        for (var e = 0; e < 3; e++) for (var r = 0; r < 7; r++) for (var c = 0; c < 128; c++)
        {
            var q = (uint)((c * 7 + e * 3 + r) & ((1 << bits) - 1));
            var bit = c * bits; var at = (e * 7 + r) * (128 * bits / 32) + bit / 32; var shift = bit % 32;
            packed[at] |= q << shift;
            if (shift + bits > 32) packed[at + 1] |= q >> (32 - shift);
        }
        matrix.Write(0, MemoryMarshal.AsBytes(packed.AsSpan()));
        matrix.Write(1, MemoryMarshal.AsBytes(scale.AsSpan())); matrix.Write(2, MemoryMarshal.AsBytes(bias.AsSpan()));
        var x = Enumerable.Range(0, 256).Select(i => MathF.Sin(i * .17f)).ToArray(); var y = new float[14];
        matrix.MultiplyExperts(x, y, [2, 0], batched: true);
        for (var slot = 0; slot < 2; slot++) for (var r = 0; r < 7; r++)
        {
            var e = slot == 0 ? 2 : 0; var expected = 0f;
            for (var c = 0; c < 128; c++) expected += (((c * 7 + e * 3 + r) & ((1 << bits) - 1)) + .5f) * x[slot * 128 + c];
            Assert.InRange(MathF.Abs(expected - y[slot * 7 + r]), 0, .004f);
        }
    }
    [Fact]
    public void DeltaRecurrenceRetainsStateAcrossTwoTokens()
    {
        if (!OperatingSystem.IsMacOS()) return;
        using var device = new QuantizedMetal(); using var memory = device.NewDeltaMemory();
        var q = Enumerable.Range(0, 2048).Select(i => MathF.Sin(i * .1f) * .01f).ToArray();
        var k = Enumerable.Range(0, 2048).Select(i => MathF.Cos(i * .07f) * .04f).ToArray();
        var v = Enumerable.Range(0, 4096).Select(i => MathF.Sin(i * .03f)).ToArray();
        var decay = Enumerable.Repeat(.8f, 32).ToArray(); var beta = Enumerable.Repeat(.6f, 32).ToArray();
        var result = new float[4096]; var state = new float[32 * 128 * 128];
        for (var token = 0; token < 2; token++)
        {
            memory.Step(q, k, v, decay, beta, result);
            for (var row = 0; row < 4096; row++)
            {
                var kh = row / 128 / 2; float prediction = 0;
                for (var d = 0; d < 128; d++) { state[row * 128 + d] *= .8f; prediction += state[row * 128 + d] * k[kh * 128 + d]; }
                var correction = (v[row] - prediction) * .6f; float expected = 0;
                for (var d = 0; d < 128; d++) { state[row * 128 + d] += k[kh * 128 + d] * correction; expected += state[row * 128 + d] * q[kh * 128 + d]; }
                Assert.InRange(MathF.Abs(expected - result[row]), 0, 1e-6f);
            }
        }
    }
    [Fact]
    public void RoutingSelectsAndRenormalizesTopEight()
    {
        var logits = Enumerable.Range(0, 256).Select(i => i * .01f).ToArray(); var ids = new int[8]; var probabilities = new float[8];
        Qwen36Model.SelectExperts(logits, ids, probabilities);
        Assert.Equal(Enumerable.Range(248, 8).Reverse(), ids);
        Assert.InRange(MathF.Abs(probabilities.Sum() - 1), 0, 1e-6f);
        Assert.True(probabilities[0] > probabilities[7]);
    }
}
