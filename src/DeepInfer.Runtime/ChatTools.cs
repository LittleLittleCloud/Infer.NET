using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DeepInfer.Runtime;

public sealed record ChatTool(string Name, JsonElement Parameters);
public sealed record ChatToolCall(string Id, string Name, string Arguments);

/// <summary>Qwen3.6 tool text and OpenAI function-call conversion. The server only
/// describes tools; the client owns and executes them.</summary>
public static partial class ChatTools
{
    public static string Prompt(JsonElement tools)
    {
        var lines = tools.EnumerateArray().Select(tool => tool.GetRawText());
        return "# Tools\n\nYou have access to the following functions:\n\n<tools>\n" +
            string.Join("\n", lines) +
            "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
            "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n" +
            "If no function call is needed, answer normally. Do not claim that tools are unavailable when functions are listed above.";
    }

    public static ChatTool[] ParseDefinitions(JsonElement tools)
    {
        if (tools.ValueKind != JsonValueKind.Array || tools.GetArrayLength() is < 1 or > 128)
            throw new ArgumentException("tools must be a nonempty array of at most 128 functions.");
        var result = new List<ChatTool>();
        foreach (var tool in tools.EnumerateArray())
        {
            if (tool.ValueKind != JsonValueKind.Object || !tool.TryGetProperty("type", out var type) || type.GetString() != "function" ||
                !tool.TryGetProperty("function", out var function) || function.ValueKind != JsonValueKind.Object ||
                !function.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String ||
                !function.TryGetProperty("parameters", out var parameters) || parameters.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("Only named function tools with JSON object parameters are supported.");
            var identifier = name.GetString()!;
            if (!ToolName().IsMatch(identifier) || result.Any(x => x.Name == identifier))
                throw new ArgumentException("Invalid or duplicate tool name.");
            result.Add(new(identifier, parameters.Clone()));
        }
        return result.ToArray();
    }

    public static string SerializeHistory(JsonElement calls)
    {
        if (calls.ValueKind != JsonValueKind.Array || calls.GetArrayLength() is < 1 or > 32)
            throw new ArgumentException("Invalid assistant tool_calls.");
        var output = new List<string>();
        foreach (var call in calls.EnumerateArray())
        {
            if (call.ValueKind != JsonValueKind.Object || !call.TryGetProperty("type", out var type) || type.GetString() != "function" ||
                !call.TryGetProperty("function", out var function) || !function.TryGetProperty("name", out var name) ||
                !function.TryGetProperty("arguments", out var arguments) || arguments.ValueKind != JsonValueKind.String)
                throw new ArgumentException("Invalid assistant function history.");
            var identifier = name.GetString()!;
            if (!ToolName().IsMatch(identifier)) throw new ArgumentException("Invalid tool name in history.");
            using var json = JsonDocument.Parse(arguments.GetString()!);
            if (json.RootElement.ValueKind != JsonValueKind.Object) throw new ArgumentException("Tool history arguments must be an object.");
            var parameters = string.Concat(json.RootElement.EnumerateObject().Select(p =>
                $"<parameter={p.Name}>\n{(p.Value.ValueKind == JsonValueKind.String ? p.Value.GetString() : p.Value.GetRawText())}\n</parameter>\n"));
            output.Add($"<tool_call>\n<function={identifier}>\n{parameters}</function>\n</tool_call>");
        }
        return string.Join("\n", output);
    }

    public static (string Text, ChatToolCall[] Calls) ParseOutput(string text, ChatTool[] tools)
    {
        // Qwen3.6 sometimes omits the two opening XML tags and starts with
        // <function_name>, while retaining the parameter/function/call close.
        // Accept only that complete, unambiguous shape for a declared tool.
        if (!text.Contains("<tool_call>", StringComparison.Ordinal) &&
            !text.Contains("</tool_call>", StringComparison.Ordinal) &&
            !text.Contains("<parameter=", StringComparison.Ordinal)) return (text, []);
        var matches = CallPattern().Matches(text);
        if (matches.Count == 0 || matches.Count > 32) throw new InvalidDataException("Malformed model tool call.");
        var prefix = text[..matches[0].Index].TrimEnd();
        var suffix = text[(matches[^1].Index + matches[^1].Length)..];
        if (suffix.Trim().Length != 0 || prefix.Contains("<tool_call>", StringComparison.Ordinal) ||
            prefix.Contains("<parameter=", StringComparison.Ordinal))
            throw new InvalidDataException("Malformed model tool call framing.");
        var calls = new List<ChatToolCall>();
        foreach (Match match in matches)
        {
            if (calls.Count > 0 && text[(matches[calls.Count - 1].Index + matches[calls.Count - 1].Length)..match.Index].Trim().Length != 0)
                throw new InvalidDataException("Unexpected text between tool calls.");
            var name = match.Groups["name"].Value;
            var tool = tools.FirstOrDefault(t => t.Name == name) ?? throw new InvalidDataException("Model called an undeclared tool.");
            var body = match.Groups["body"].Value;
            var parameters = new JsonObject();
            var cursor = 0;
            foreach (Match parameter in ParameterPattern().Matches(body))
            {
                if (body[cursor..parameter.Index].Trim().Length != 0) throw new InvalidDataException("Malformed tool parameter block.");
                var key = parameter.Groups["name"].Value;
                if (!ToolName().IsMatch(key) || parameters.ContainsKey(key)) throw new InvalidDataException("Invalid or duplicate tool parameter.");
                var value = parameter.Groups["value"].Value.Trim('\r', '\n');
                if (!tool.Parameters.TryGetProperty("properties", out var props) || props.ValueKind != JsonValueKind.Object || !props.TryGetProperty(key, out var property))
                    throw new InvalidDataException("Model supplied an undeclared tool parameter.");
                var kind = property.ValueKind == JsonValueKind.Object && property.TryGetProperty("type", out var t) ? t.GetString() : "string";
                try { parameters[key] = kind == "string" ? JsonValue.Create(value) : JsonNode.Parse(value); }
                catch (JsonException ex) { throw new InvalidDataException("Model supplied invalid typed tool argument.", ex); }
                cursor = parameter.Index + parameter.Length;
            }
            if (body[cursor..].Trim().Length != 0) throw new InvalidDataException("Malformed tool parameter tail.");
            if (tool.Parameters.TryGetProperty("required", out var required) && required.ValueKind == JsonValueKind.Array)
                foreach (var requiredName in required.EnumerateArray())
                    if (!parameters.ContainsKey(requiredName.GetString()!)) throw new InvalidDataException("Model omitted a required tool parameter.");
            calls.Add(new("call_" + Guid.NewGuid().ToString("N"), name, parameters.ToJsonString()));
        }
        return (prefix, calls.ToArray());
    }

    [GeneratedRegex("^[A-Za-z_][A-Za-z0-9_-]{0,63}$")]
    private static partial Regex ToolName();
    [GeneratedRegex(@"(?:<tool_call>\s*<function=|<)(?<name>[A-Za-z_][A-Za-z0-9_-]{0,63})>(?<body>(?:\s*<parameter=[A-Za-z_][A-Za-z0-9_-]{0,63}>.*?</parameter>)*)\s*</function>\s*</tool_call>", RegexOptions.Singleline)]
    private static partial Regex CallPattern();
    [GeneratedRegex(@"<parameter=(?<name>[A-Za-z_][A-Za-z0-9_-]{0,63})>(?<value>.*?)</parameter>", RegexOptions.Singleline)]
    private static partial Regex ParameterPattern();
}
