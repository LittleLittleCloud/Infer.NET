using DeepInfer.Backends.Compute;
namespace DeepInfer.Backends.Models;

public interface ITransformerWeights : IDisposable
{
    LlamaConfig Config { get; }
    float Epsilon { get; }
    float RopeTheta { get; }
    bool SplitHalfRope { get; }
    void CopyEmbedding(int token, Span<float> output);
    float[][] AttentionNorm { get; }
    float[][] FfnNorm { get; }
    float[] FinalNorm { get; }
    IMatrix[] Query { get; }
    IMatrix[] Key { get; }
    IMatrix[] Value { get; }
    IMatrix[] AttentionOutput { get; }
    IMatrix[] Gate { get; }
    IMatrix[] Down { get; }
    IMatrix[] Up { get; }
    IMatrix Classifier { get; }
    float[][]? QueryBias { get; }
    float[][]? KeyBias { get; }
    float[][]? ValueBias { get; }
}
public interface IModelTokenizer
{
    int[] EncodePrompt(string text, CancellationToken cancellationToken);
    byte[] DecodePiece(int previous, int token);
    string Decode(ReadOnlySpan<int> tokens, int previous = 1);
    bool IsEnd(int token);
}
