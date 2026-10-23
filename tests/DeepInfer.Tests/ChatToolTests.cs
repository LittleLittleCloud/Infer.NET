using System.Text.Json;
using DeepInfer.Runtime;
using Xunit;

namespace DeepInfer.Tests;

public sealed class ChatToolTests
{
    private const string Tools = """[{"type":"function","function":{"name":"bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]""";

    [Fact]
    public void AcceptsPiFunctionRequestAndToolResultHistory()
    {
        var json = """{"model":"qwen3.6-35b-a3b","stream":true,"tool_choice":"auto","tools":TOOLSET,"messages":[{"role":"user","content":"list files"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"ls -la\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"file.txt"}]}""".Replace("TOOLSET", Tools);
        using var document = JsonDocument.Parse(json);
        var request = ChatRequest.Parse(document.RootElement, ChatFormat.Completions);
        Assert.True(request.Stream);
        Assert.Equal("bash", Assert.Single(request.Tools!).Name);
        Assert.Contains("<tools>", request.Inference.Messages![0].Content);
        Assert.Contains("<function=bash>", request.Inference.Messages![2].Content);
        Assert.Equal("user", request.Inference.Messages![3].Role);
        Assert.Contains("<tool_response>", request.Inference.Messages![3].Content);
    }

    [Fact]
    public void ConvertsDeclaredModelToolCallIntoStructuredArguments()
    {
        using var document = JsonDocument.Parse(Tools);
        var definitions = ChatTools.ParseDefinitions(document.RootElement);
        var parsed = ChatTools.ParseOutput("<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>", definitions);
        Assert.Empty(parsed.Text);
        var call = Assert.Single(parsed.Calls);
        Assert.Equal("bash", call.Name);
        using var arguments = JsonDocument.Parse(call.Arguments);
        Assert.Equal("ls -la", arguments.RootElement.GetProperty("command").GetString());
        Assert.StartsWith("call_", call.Id);
    }

    [Fact]
    public void RepairsObservedQwenShorthandOnlyForDeclaredTool()
    {
        using var document = JsonDocument.Parse(Tools);
        var definitions = ChatTools.ParseDefinitions(document.RootElement);
        var output = "<bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>";
        var call = Assert.Single(ChatTools.ParseOutput(output, definitions).Calls);
        Assert.Equal("bash", call.Name);
        using var arguments = JsonDocument.Parse(call.Arguments);
        Assert.Equal("pwd", arguments.RootElement.GetProperty("command").GetString());
        Assert.Throws<InvalidDataException>(() => ChatTools.ParseOutput(output, [new ChatTool("read", definitions[0].Parameters)]));
    }

    [Theory]
    [InlineData("<tool_call><function=other><parameter=command>x</parameter></function></tool_call>")]
    [InlineData("<tool_call><function=bash></function></tool_call>")]
    [InlineData("<tool_call><function=bash><parameter=other>x</parameter></function></tool_call>")]
    [InlineData("<tool_call><function=bash><parameter=command>x</parameter></function>")]
    [InlineData("<bash><parameter=command>x</parameter></tool_call>")]
    public void RefusesUndeclaredOrMalformedModelCalls(string output)
    {
        using var document = JsonDocument.Parse(Tools);
        Assert.Throws<InvalidDataException>(() => ChatTools.ParseOutput(output, ChatTools.ParseDefinitions(document.RootElement)));
    }
}
