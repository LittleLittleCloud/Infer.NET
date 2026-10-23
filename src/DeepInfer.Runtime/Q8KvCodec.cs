using System.Buffers.Binary;

namespace DeepInfer.Runtime;

/// <summary>
/// Reference format for one K or V tensor token: 512 FP32 components, split into
/// 16 contiguous 32-component blocks. Each block is little-endian FP16 scale
/// followed by 32 signed int8 values. A zero block has zero scale and values.
/// Native backends must reproduce this byte layout before Q8 serving is enabled.
/// </summary>
public static class Q8KvCodec
{
    public const int Components = 512;
    public const int BlockComponents = 32;
    public const int BlockBytes = 2 + BlockComponents;
    public const int TensorTokenBytes = Components / BlockComponents * BlockBytes;

    public static void Encode(ReadOnlySpan<float> source, Span<byte> destination)
    {
        if (source.Length != Components || destination.Length != TensorTokenBytes)
            throw new ArgumentException("Q8 KV token shape mismatch.");

        for (var block = 0; block < Components / BlockComponents; block++)
        {
            var values = source.Slice(block * BlockComponents, BlockComponents);
            var bytes = destination.Slice(block * BlockBytes, BlockBytes);
            var maximum = 0f;
            foreach (var value in values)
            {
                if (!float.IsFinite(value)) throw new ArgumentException("Q8 KV input must be finite.");
                maximum = MathF.Max(maximum, MathF.Abs(value));
            }
            if (maximum == 0)
            {
                bytes.Clear();
                continue;
            }
            var scale = (Half)(maximum / 127f);
            if (!Half.IsFinite(scale) || scale == (Half)0)
                throw new ArgumentOutOfRangeException(nameof(source), "Q8 KV scale is outside FP16 range.");
            BinaryPrimitives.WriteUInt16LittleEndian(bytes, BitConverter.HalfToUInt16Bits(scale));
            var divisor = (float)scale;
            for (var index = 0; index < BlockComponents; index++)
            {
                var quantized = Math.Clamp((int)MathF.Round(values[index] / divisor,
                    MidpointRounding.ToEven), -127, 127);
                bytes[2 + index] = unchecked((byte)(sbyte)quantized);
            }
        }
    }

    public static void Decode(ReadOnlySpan<byte> source, Span<float> destination)
    {
        if (source.Length != TensorTokenBytes || destination.Length != Components)
            throw new ArgumentException("Q8 KV token shape mismatch.");

        for (var block = 0; block < Components / BlockComponents; block++)
        {
            var bytes = source.Slice(block * BlockBytes, BlockBytes);
            var values = destination.Slice(block * BlockComponents, BlockComponents);
            var scale = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(bytes));
            if (!float.IsFinite(scale) || scale < 0)
                throw new InvalidDataException("Invalid Q8 KV scale.");
            for (var index = 0; index < BlockComponents; index++)
            {
                var quantized = unchecked((sbyte)bytes[2 + index]);
                if (quantized == -128 || (scale == 0 && quantized != 0))
                    throw new InvalidDataException("Invalid Q8 KV payload.");
                values[index] = quantized * scale;
            }
        }
    }
}
