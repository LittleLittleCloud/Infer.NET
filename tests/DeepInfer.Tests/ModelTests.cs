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
    [Fact] public void TokenizerMergesByScoreAndRoundTripsUnicodeBytes()
    {
        using var fixture = new Fixture();
        var tokenizer = new LlamaTokenizer(fixture.Tokenizer, 263);
        Assert.Equal(new[] { 1, 259, 262 }, tokenizer.Encode("ab"));
        foreach (var text in new[] { "ab", "Hello 世界🌍", "a\nb", "<s>abc</s>" })
        {
            var ids = tokenizer.Encode(text);
            Assert.Equal(1, ids[0]); Assert.DoesNotContain(1, ids.Skip(1)); Assert.DoesNotContain(2, ids);
            Assert.Equal(text, tokenizer.Decode(ids));
        }
    }
    [Fact] public void CheckpointRejectsTruncationAndInvalidHeaderBeforeAllocating()
    {
        using var fixture = new Fixture(); using var device = new CpuMatVecDevice();
        var bytes = File.ReadAllBytes(fixture.Weights);
        File.WriteAllBytes(fixture.Weights, bytes[..^4]);
        Assert.Throws<InvalidDataException>(() => new LlamaCheckpoint(fixture.Weights, device));
        BitConverter.GetBytes(int.MaxValue).CopyTo(bytes, 0); File.WriteAllBytes(fixture.Weights, bytes);
        Assert.Throws<InvalidDataException>(() => new LlamaCheckpoint(fixture.Weights, device));
    }
    [Fact] public void TokenizerRejectsTruncatedFile()
    {
        using var fixture = new Fixture(); var bytes = File.ReadAllBytes(fixture.Tokenizer);
        File.WriteAllBytes(fixture.Tokenizer, bytes[..^1]);
        Assert.Throws<EndOfStreamException>(() => new LlamaTokenizer(fixture.Tokenizer, 263));
    }
    [Fact] public async Task RealModelResetsKvCacheAndHonorsContextAndCancellation()
    {
        using var fixture = new Fixture(); var info = new ModelInfo("fixture", "llama2c-f32", false);
        await using var model = new LlamaModel(info, fixture.Weights, fixture.Tokenizer, "cpu");
        var a = await model.GenerateAsync(new("ab", 4, "fixture", "cpu"), default);
        await model.GenerateAsync(new("b", 4, "fixture", "cpu"), default);
        var b = await model.GenerateAsync(new("ab", 4, "fixture", "cpu"), default);
        Assert.False(a.IsSimulation); Assert.Equal(a.TokenIds, b.TokenIds); Assert.Equal(a.Text, b.Text);
        Assert.Equal(0, a.GpuDispatches); Assert.Equal(3, a.PromptTokens);
        Assert.All(model.EvaluatePrompt("ab"), value => Assert.True(float.IsFinite(value)));
        await Assert.ThrowsAsync<InferenceInputException>(async () => await model.GenerateAsync(new(new string('z', 20), 4, "fixture", "cpu"), default));
        using var stop = new CancellationTokenSource(); stop.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await model.GenerateAsync(new("ab", 4, "fixture", "cpu"), stop.Token));
        var limited = await model.GenerateAsync(new("ab", 20, "fixture", "cpu"), default);
        Assert.True(limited.FinishReason is "context_length" or "eos");
    }
    private sealed class Fixture : IDisposable
    {
        private readonly string directory = Path.Combine(Path.GetTempPath(), "deepinfer-test-" + Guid.NewGuid().ToString("N"));
        public string Tokenizer => Path.Combine(directory, "tokenizer.bin");
        public string Weights => Path.Combine(directory, "model.bin");
        public Fixture()
        {
            Directory.CreateDirectory(directory);
            using (var writer = new BinaryWriter(File.Create(Tokenizer)))
            {
                writer.Write(16);
                for (var i = 0; i < 263; i++)
                {
                    var piece = i switch { 0 => "<unk>", 1 => "<s>", 2 => "</s>", 259 => " ", 260 => "a", 261 => "b", 262 => "ab", _ => $"<0x{i-3:X2}>" };
                    var bytes = System.Text.Encoding.UTF8.GetBytes(piece);
                    writer.Write(i == 262 ? 10f : 0f); writer.Write(bytes.Length); writer.Write(bytes);
                }
            }
            // One GQA layer, dimension 4, 2 query heads / 1 KV head, tied classifier.
            using (var writer = new BinaryWriter(File.Create(Weights)))
            {
                foreach (var value in new[] { 4, 8, 1, 2, 1, 263, 16 }) writer.Write(value);
                var count = 263 * 4 + (2 * 4 + 2 * 4 * 4 + 2 * 2 * 4 + 3 * 4 * 8) + 4 + 16 * 2;
                for (var i = 0; i < count; i++) writer.Write(0.02f + (i % 13) * 0.001f);
            }
        }
        public void Dispose() => Directory.Delete(directory, true);
    }
}
