namespace DeepInfer.Backends.Models;

public interface IModelTokenizer
{
    int[] EncodePrompt(string text, CancellationToken cancellationToken);
    byte[] DecodePiece(int previous, int token);
    string Decode(ReadOnlySpan<int> tokens, int previous = 1);
    bool IsEnd(int token);
}
