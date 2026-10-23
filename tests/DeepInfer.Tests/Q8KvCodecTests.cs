using DeepInfer.Runtime;
using Xunit;

namespace DeepInfer.Tests;

public sealed class Q8KvCodecTests
{
    [Fact]
    public void RoundTripsSignedValuesWithBoundedBlockError()
    {
        var input = Enumerable.Range(0, Q8KvCodec.Components)
            .Select(index => (float)(Math.Sin(index * 0.37) * (index / 32 + 1) / 3.0)).ToArray();
        var bytes = new byte[Q8KvCodec.TensorTokenBytes];
        var output = new float[input.Length];
        Q8KvCodec.Encode(input, bytes);
        Q8KvCodec.Decode(bytes, output);
        Assert.Equal(544, bytes.Length);
        for (var block = 0; block < 16; block++)
        {
            var maximum = input.Skip(block * 32).Take(32).Max(value => MathF.Abs(value));
            for (var index = block * 32; index < (block + 1) * 32; index++)
                Assert.InRange(MathF.Abs(input[index] - output[index]), 0, maximum / 127f);
        }
        var repeated = new byte[bytes.Length];
        Q8KvCodec.Encode(input, repeated);
        Assert.Equal(bytes, repeated);
    }

    [Fact]
    public void RejectsMalformedScaleAndNonfiniteInput()
    {
        var source = new float[Q8KvCodec.Components];
        var data = new byte[Q8KvCodec.TensorTokenBytes];
        Q8KvCodec.Encode(source, data);
        Assert.All(data, value => Assert.Equal(0, value));
        source[0] = float.NaN;
        Assert.Throws<ArgumentException>(() => Q8KvCodec.Encode(source, data));
        data[0] = 0xff; data[1] = 0x7f;
        Assert.Throws<InvalidDataException>(() => Q8KvCodec.Decode(data, new float[source.Length]));
    }
}
