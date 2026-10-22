using System.Globalization;
using System.Text;
namespace DeepInfer.Backends.Models;

// Score-ranked adjacent BPE merging for the tokenizer.bin exported by llama2.c.
public sealed class LlamaTokenizer : IModelTokenizer
{
    private readonly byte[][] pieces;
    private readonly string[] vocabulary;
    private readonly float[] scores;
    private readonly Dictionary<string, int> ids = new(StringComparer.Ordinal);
    public int VocabularySize => pieces.Length;
    public LlamaTokenizer(string path, int vocabularySize)
    {
        if (vocabularySize is < 259 or > 128000) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        using var stream = File.OpenRead(path); using var reader = new BinaryReader(stream);
        if (stream.Length is < 4 or > 16777216) throw new InvalidDataException("Invalid tokenizer size.");
        var maxLength = reader.ReadInt32();
        if (maxLength is < 1 or > 4096) throw new InvalidDataException("Invalid maximum token length.");
        pieces = new byte[vocabularySize][]; vocabulary = new string[vocabularySize]; scores = new float[vocabularySize];
        for (var i = 0; i < vocabularySize; i++)
        {
            scores[i] = reader.ReadSingle(); var length = reader.ReadInt32();
            if (length < 0 || length > maxLength || float.IsNaN(scores[i])) throw new InvalidDataException("Invalid tokenizer entry.");
            var bytes = reader.ReadBytes(length);
            if (bytes.Length != length) throw new EndOfStreamException("Truncated tokenizer.");
            vocabulary[i] = new UTF8Encoding(false, true).GetString(bytes);
            pieces[i] = bytes;
            if (!ids.TryAdd(vocabulary[i], i)) throw new InvalidDataException("Duplicate tokenizer piece.");
        }
        if (stream.Position != stream.Length || !ids.ContainsKey(" ")) throw new InvalidDataException("Tokenizer vocabulary mismatch.");
        for (var b = 0; b < 256; b++)
            if (vocabulary[b + 3] != $"<0x{b:X2}>") throw new InvalidDataException("Tokenizer must use llama byte-fallback IDs 3..258.");
    }
    public int[] EncodePrompt(string text, CancellationToken cancellationToken) => Encode(text, cancellationToken);
    public bool IsEnd(int token) => token is 1 or 2;
    public int[] Encode(string text, CancellationToken cancellationToken = default)
    {
        var tokens = new List<int> { 1 }; // BOS, never parse special tokens from user input
        if (text.Length != 0) tokens.Add(ids[" "]);
        foreach (var rune in text.EnumerateRunes())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (ids.TryGetValue(rune.ToString(), out var token)) tokens.Add(token);
            else foreach (var b in Encoding.UTF8.GetBytes(rune.ToString())) tokens.Add(b + 3);
        }
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var position = -1; var merged = -1; var score = float.NegativeInfinity;
            for (var i = 1; i < tokens.Count - 1; i++) // do not merge BOS
                if (ids.TryGetValue(vocabulary[tokens[i]] + vocabulary[tokens[i + 1]], out var candidate) && scores[candidate] > score)
                { position = i; merged = candidate; score = scores[candidate]; }
            if (position < 0) return tokens.ToArray();
            tokens[position] = merged; tokens.RemoveAt(position + 1);
        }
    }
    public byte[] DecodePiece(int previous, int token)
    {
        if (token < 0 || token >= pieces.Length) throw new ArgumentOutOfRangeException(nameof(token));
        if (token is 1 or 2) return [];
        var raw = pieces[token];
        if (raw.Length == 6 && raw[0] == '<' && raw[1] == '0' && raw[2] == 'x' && raw[5] == '>')
            return [byte.Parse(vocabulary[token].AsSpan(3,2), NumberStyles.HexNumber, CultureInfo.InvariantCulture)];
        return previous == 1 && raw.Length > 0 && raw[0] == ' ' ? raw[1..] : raw;
    }
    public string Decode(ReadOnlySpan<int> tokens, int previous = 1)
    {
        using var stream = new MemoryStream();
        foreach (var token in tokens) { stream.Write(DecodePiece(previous, token)); previous = token; }
        return Encoding.UTF8.GetString(stream.ToArray());
    }
}
