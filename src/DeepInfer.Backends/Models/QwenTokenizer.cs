using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
namespace DeepInfer.Backends.Models;

// Byte-level BPE with pinned Qwen3.6 pretokenization. User text never injects special-token IDs.
public sealed class QwenTokenizer : IModelTokenizer
{
    public const int EndOfText = 248044, MessageStart = 248045, MessageEnd = 248046;
    private readonly Dictionary<string, int> vocabulary = new(StringComparer.Ordinal);
    private readonly Dictionary<(string, string), int> ranks = new();
    private readonly Dictionary<int, byte[]> pieces = new();
    private readonly char[] byteEncoder = new char[256];
    private readonly Regex extendedPreTokenizer;
    private readonly int endOfText, messageStart, messageEnd;
    public QwenTokenizer(string path)
    {
        endOfText = EndOfText;
        messageStart = endOfText + 1; messageEnd = endOfText + 2;
        extendedPreTokenizer = new(Pattern36, RegexOptions.CultureInvariant, TimeSpan.FromSeconds(2));
        var bytes = Enumerable.Range(33,94).Concat(Enumerable.Range(161,12)).Concat(Enumerable.Range(174,82)).ToList();
        var mapped = bytes.ToList(); var extra = 0;
        for (var b = 0; b < 256; b++) if (!bytes.Contains(b)) { bytes.Add(b); mapped.Add(256 + extra++); }
        var decode = new Dictionary<char, byte>();
        for (var i = 0; i < bytes.Count; i++) { byteEncoder[bytes[i]] = (char)mapped[i]; decode[(char)mapped[i]] = (byte)bytes[i]; }
        using var document = JsonDocument.Parse(File.ReadAllBytes(path)); var root = document.RootElement;
        if (root.GetProperty("normalizer").GetProperty("type").GetString() != "NFC" || root.GetProperty("model").GetProperty("type").GetString() != "BPE")
            throw new NotSupportedException("Only the pinned Qwen3.6 byte-BPE tokenizer is supported.");
        var pre = root.GetProperty("pre_tokenizer").GetProperty("pretokenizers");
        if (pre[0].GetProperty("pattern").GetProperty("Regex").GetString() != Pattern36 ||
            pre[1].GetProperty("type").GetString() != "ByteLevel" || pre[1].GetProperty("add_prefix_space").GetBoolean())
            throw new NotSupportedException("Unsupported Qwen pretokenizer configuration.");
        var model = root.GetProperty("model");
        foreach (var item in model.GetProperty("vocab").EnumerateObject())
        {
            var id = item.Value.GetInt32(); vocabulary.Add(item.Name, id);
            pieces.Add(id, item.Name.Select(c => decode.TryGetValue(c, out var b) ? b : throw new InvalidDataException("Unknown byte-BPE character.")).ToArray());
        }
        var rank = 0;
        foreach (var merge in model.GetProperty("merges").EnumerateArray())
        {
            var pair = merge.ValueKind == JsonValueKind.String ? merge.GetString()!.Split(' ') : merge.EnumerateArray().Select(x => x.GetString()!).ToArray();
            if (pair.Length != 2 || !ranks.TryAdd((pair[0],pair[1]),rank++)) throw new InvalidDataException("Invalid BPE merges.");
        }
        foreach (var added in root.GetProperty("added_tokens").EnumerateArray())
            pieces[added.GetProperty("id").GetInt32()] = Encoding.UTF8.GetBytes(added.GetProperty("content").GetString()!);
        if (vocabulary.Count != endOfText || !pieces.ContainsKey(messageEnd)) throw new NotSupportedException("Qwen vocabulary IDs do not match.");
    }
    public int[] EncodeText(string text, CancellationToken ct = default)
    {
        var tokens = new List<int>();
        foreach (Match match in extendedPreTokenizer.Matches(text.Normalize(NormalizationForm.FormC)))
        {
            ct.ThrowIfCancellationRequested();
            var encoded = new string(Encoding.UTF8.GetBytes(match.Value).Select(b => byteEncoder[b]).ToArray());
            if (vocabulary.TryGetValue(encoded, out var whole)) { tokens.Add(whole); continue; }
            var symbols = encoded.Select(c => c.ToString()).ToList();
            while (symbols.Count > 1)
            {
                ct.ThrowIfCancellationRequested();
                var best = int.MaxValue; var position = -1;
                for (var i = 0; i < symbols.Count - 1; i++)
                    if (ranks.TryGetValue((symbols[i], symbols[i + 1]), out var candidate) && candidate < best) { best = candidate; position = i; }
                if (position < 0) break;
                symbols[position] += symbols[position + 1]; symbols.RemoveAt(position + 1);
            }
            foreach (var symbol in symbols) tokens.Add(vocabulary[symbol]);
        }
        return tokens.ToArray();
    }
    public int[] EncodePrompt(string text, CancellationToken cancellationToken)
    {
        var output = new List<int>();
        void Text(string content) => output.AddRange(EncodeText(content, cancellationToken));
        output.Add(messageStart); Text("user\n"); Text(text); output.Add(messageEnd); Text("\n");
        output.Add(messageStart); Text("assistant\n"); output.Add(248068); Text("\n\n");
        output.Add(248069); Text("\n\n");
        return output.ToArray();
    }
    public bool IsEnd(int token) => token == endOfText || token == messageEnd;
    public int[] EncodeMessages(DeepInfer.Core.ChatTurn[] messages, CancellationToken ct)
    {
        var result = new List<int>();
        foreach (var message in messages)
        {
            result.Add(messageStart);
            result.AddRange(EncodeText(message.Role + "\n", ct));
            result.AddRange(EncodeText(message.Content, ct));
            result.Add(messageEnd); result.AddRange(EncodeText("\n", ct));
        }
        result.Add(messageStart); result.AddRange(EncodeText("assistant\n", ct));
        result.Add(248068); result.AddRange(EncodeText("\n\n", ct));
        result.Add(248069); result.AddRange(EncodeText("\n\n", ct));
        return result.ToArray();
    }
    public byte[] DecodePiece(int previous, int token) => IsEnd(token) ? [] : pieces.TryGetValue(token, out var value) ? value :
        throw new InvalidDataException($"Model produced unknown tokenizer ID {token}.");
    public string Decode(ReadOnlySpan<int> tokens, int previous = 1)
    {
        using var stream = new MemoryStream(); foreach (var token in tokens) stream.Write(DecodePiece(previous, token));
        return Encoding.UTF8.GetString(stream.ToArray());
    }
    private const string Pattern36 = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
}
