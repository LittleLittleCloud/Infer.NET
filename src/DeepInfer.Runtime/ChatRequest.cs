using System.Text.Json;
using DeepInfer.Core;
namespace DeepInfer.Runtime;

public enum ChatFormat { Completions, Messages, Responses }
public sealed record ChatRequest(InferenceRequest Inference, bool Stream, bool IncludeUsage, ChatTool[]? Tools = null)
{
    public static ChatRequest Parse(JsonElement root, ChatFormat format)
    {
        if (root.ValueKind != JsonValueKind.Object) throw new ArgumentException("Request must be an object.");
        var allowed = new HashSet<string>(["model", "stream", "temperature", "top_p"]);
        allowed.UnionWith(format switch {
            ChatFormat.Completions => ["messages", "max_tokens", "max_completion_tokens", "n", "stream_options", "tools", "tool_choice", "parallel_tool_calls"],
            ChatFormat.Messages => ["messages", "system", "max_tokens"],
            _ => new[] { "input", "instructions", "max_output_tokens", "store" }
        });
        var names = new HashSet<string>();
        foreach (var p in root.EnumerateObject())
            if (!names.Add(p.Name) || !allowed.Contains(p.Name)) throw new ArgumentException($"Unsupported or duplicate parameter: {p.Name}.");
        string String(JsonElement e) => e.ValueKind == JsonValueKind.String ? e.GetString()! : throw new ArgumentException("Expected a string.");
        string Text(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.String) return e.GetString()!;
            if (e.ValueKind != JsonValueKind.Array) throw new ArgumentException("Only text content is supported.");
            var parts = new List<string>();
            foreach (var block in e.EnumerateArray())
            {
                if (block.ValueKind != JsonValueKind.Object || !block.TryGetProperty("type", out var type) || type.GetString() is not ("text" or "input_text" or "output_text") || !block.TryGetProperty("text", out var text)) throw new ArgumentException("Only text content blocks are supported.");
                if (block.EnumerateObject().Any(p => p.Name is not ("type" or "text" or "annotations" or "logprobs"))) throw new ArgumentException("Unsupported content block parameter.");
                if (block.TryGetProperty("annotations", out var annotations) && (annotations.ValueKind != JsonValueKind.Array || annotations.GetArrayLength() != 0)) throw new ArgumentException("Annotations are not supported.");
                if (block.TryGetProperty("logprobs", out var logprobs) && logprobs.ValueKind != JsonValueKind.Null && (logprobs.ValueKind != JsonValueKind.Array || logprobs.GetArrayLength() != 0)) throw new ArgumentException("Logprobs are not supported.");
                parts.Add(String(text));
            }
            return string.Concat(parts);
        }
        bool Boolean(string name, bool fallback = false) => root.TryGetProperty(name, out var e) ? e.ValueKind is JsonValueKind.True or JsonValueKind.False ? e.GetBoolean() : throw new ArgumentException($"{name} must be boolean.") : fallback;
        int Number(string name, int fallback) => root.TryGetProperty(name, out var e) && (e.ValueKind != JsonValueKind.Number || !e.TryGetInt32(out _)) ? throw new ArgumentException($"{name} must be an integer.") : root.TryGetProperty(name, out e) ? e.GetInt32() : fallback;
        foreach (var (name, supported) in new[] { ("temperature", 0d), ("top_p", 1d) })
            if (root.TryGetProperty(name, out var v) && (v.ValueKind != JsonValueKind.Number || !v.TryGetDouble(out var d) || d != supported)) throw new ArgumentException($"Only {name}={supported} is supported (greedy decoding).");
        if (Number("n", 1) != 1) throw new ArgumentException("Only n=1 is supported.");
        if (Boolean("store")) throw new ArgumentException("Response storage is not implemented; use store=false and resend history.");
        var model = root.TryGetProperty("model", out var m) ? String(m) : throw new ArgumentException("model is required.");
        if (model != Protocol.Qwen36Model) throw new ArgumentException("Only " + Protocol.Qwen36Model + " supports these chat APIs.");
        var turns = new List<ChatTurn>();
        ChatTool[]? tools = null;
        if (root.TryGetProperty("tools", out var definitions)) tools = ChatTools.ParseDefinitions(definitions);
        if (root.TryGetProperty("tool_choice", out var choice))
        {
            if (tools == null || choice.ValueKind != JsonValueKind.String || choice.GetString() is not ("auto" or "required" or "none"))
                throw new ArgumentException("tool_choice must be auto, required or none with declared tools.");
            if (choice.GetString() == "none") tools = null;
        }
        if (root.TryGetProperty("parallel_tool_calls", out var parallel) && parallel.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new ArgumentException("parallel_tool_calls must be boolean.");
        if (tools != null) turns.Add(new("system", ChatTools.Prompt(definitions)));
        var systemField = format == ChatFormat.Messages ? "system" : "instructions";
        if (root.TryGetProperty(systemField, out var system)) turns.Add(new("system", Text(system)));
        var field = format == ChatFormat.Responses ? "input" : "messages";
        if (!root.TryGetProperty(field, out var input)) throw new ArgumentException(field + " is required.");
        if (format == ChatFormat.Responses && input.ValueKind == JsonValueKind.String) turns.Add(new("user", input.GetString()!));
        else
        {
            if (input.ValueKind != JsonValueKind.Array) throw new ArgumentException(field + " must be an array.");
            foreach (var item in input.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object || !item.TryGetProperty("role", out var r)) throw new ArgumentException("Messages require role.");
                if (item.EnumerateObject().Any(p => p.Name is not ("role" or "content" or "type" or "id" or "status" or "tool_calls" or "tool_call_id" or "name" or "refusal"))) throw new ArgumentException("Unsupported message parameter.");
                if (item.TryGetProperty("refusal", out var refusal) && refusal.ValueKind != JsonValueKind.Null) throw new ArgumentException("Refusal history is unsupported.");
                if (item.TryGetProperty("type", out var type) && type.GetString() != "message") throw new ArgumentException("Only message items are supported.");
                var role = String(r);
                if (format == ChatFormat.Messages && role is not ("user" or "assistant")) throw new ArgumentException("Messages API uses the top-level system field.");
                if (role == "developer") role = "system";
                if (role == "tool")
                {
                    if (format != ChatFormat.Completions || !item.TryGetProperty("tool_call_id", out var callId) || callId.ValueKind != JsonValueKind.String || !item.TryGetProperty("content", out var result))
                        throw new ArgumentException("Tool results require content and tool_call_id.");
                    turns.Add(new("user", "<tool_response>\n" + Text(result) + "\n</tool_response>"));
                }
                else if (role == "assistant" && item.TryGetProperty("tool_calls", out var calls))
                {
                    var body = item.TryGetProperty("content", out var content) && content.ValueKind != JsonValueKind.Null ? Text(content) : "";
                    turns.Add(new("assistant", body + (body.Length == 0 ? "" : "\n") + ChatTools.SerializeHistory(calls)));
                }
                else
                {
                    if (!item.TryGetProperty("content", out var content)) throw new ArgumentException("Messages require content.");
                    turns.Add(new(role, Text(content)));
                }
            }
        }
        if (root.TryGetProperty("max_tokens", out _) && root.TryGetProperty("max_completion_tokens", out _)) throw new ArgumentException("Specify only one output token limit.");
        if (format == ChatFormat.Messages && !root.TryGetProperty("max_tokens", out _)) throw new ArgumentException("max_tokens is required.");
        var max = Number(format == ChatFormat.Responses ? "max_output_tokens" : "max_completion_tokens", Number("max_tokens", Protocol.MaxTokens));
        if (max is < 1 or > Protocol.MaxTokens) throw new ArgumentException($"Output token limit must be 1..{Protocol.MaxTokens}.");
        var request = new InferenceRequest("", max, model, "metal", Messages: turns.ToArray(), UseRemainingContext: !root.TryGetProperty("max_tokens", out _) && !root.TryGetProperty("max_completion_tokens", out _) && !root.TryGetProperty("max_output_tokens", out _));
        if (RequestValidation.Error(request) is { } error) throw new ArgumentException(error);
        var stream = Boolean("stream"); var usage = false;
        if (root.TryGetProperty("stream_options", out var opts))
        {
            if (!stream || opts.ValueKind != JsonValueKind.Object || opts.EnumerateObject().Any(p => p.Name != "include_usage")) throw new ArgumentException("Only stream_options.include_usage is supported with stream=true.");
            if (opts.TryGetProperty("include_usage", out var u)) usage = u.ValueKind is JsonValueKind.True or JsonValueKind.False ? u.GetBoolean() : throw new ArgumentException("include_usage must be boolean.");
        }
        return new(request, stream, usage, tools);
    }
}
