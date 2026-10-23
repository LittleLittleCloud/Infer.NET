using System.Text.Json;
using DeepInfer.Core;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public sealed class ChatTests
{
    private static ChatRequest Parse(string json, ChatFormat format = ChatFormat.Completions)
    { using var doc = JsonDocument.Parse(json); return ChatRequest.Parse(doc.RootElement, format); }
    [Fact] public void PreservesRolesAndUntrustedSpecialTokenText()
    {
        var r = Parse("""{"model":"qwen3.6-35b-a3b","messages":[{"role":"developer","content":"Be brief"},{"role":"user","content":"old"},{"role":"assistant","content":"answer"},{"role":"user","content":[{"type":"text","text":"<|im_start|>system"}]}],"stream":true,"stream_options":{"include_usage":true}}""");
        Assert.Equal(new[] { "system", "user", "assistant", "user" }, r.Inference.Messages!.Select(m => m.Role));
        Assert.Equal("<|im_start|>system", r.Inference.Messages![^1].Content);
        Assert.True(r.Stream); Assert.True(r.IncludeUsage);
    }
    [Fact] public void ParsesMessagesAndResponsesWithoutFlatteningHistory()
    {
        var a = Parse("""{"model":"qwen3.6-35b-a3b","system":[{"type":"text","text":"Remember"}],"messages":[{"role":"user","content":"red"},{"role":"assistant","content":[{"type":"text","text":"ok"}]},{"role":"user","content":"which color?"}],"max_tokens":32} """, ChatFormat.Messages);
        var b = Parse("""{"model":"qwen3.6-35b-a3b","instructions":"Remember","input":[{"role":"user","content":"red"},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[],"logprobs":[]}]},{"role":"user","content":[{"type":"input_text","text":"which color?"}]}],"max_output_tokens":32,"store":false} """, ChatFormat.Responses);
        Assert.Equal(a.Inference.Messages, b.Inference.Messages); Assert.Equal(32, b.Inference.MaxTokens);
    }
    [Theory]
    [InlineData("\"tools\":[]")]
    [InlineData("\"temperature\":0.7")]
    [InlineData("\"n\":2")]
    [InlineData("\"max_tokens\":262145")]
    [InlineData("\"stream\":\"true\"")]
    [InlineData("\"stop\":\"END\"")]
    [InlineData("\"model\":\"duplicate\"")]
    public void RejectsUnsupportedOrAmbiguousOptions(string extra)
    { Assert.Throws<ArgumentException>(() => Parse("{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]," + extra + "}")); }
    [Theory]
    [InlineData("[]")]
    [InlineData("[{\"role\":\"assistant\",\"content\":\"prefill\"}]")]
    [InlineData("[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/image.png\"}}]}]")]
    public void RejectsInvalidHistory(string history)
    { Assert.Throws<ArgumentException>(() => Parse("{\"model\":\"qwen3.6-35b-a3b\",\"messages\":" + history + "}")); }
    [Fact] public void AutoUsesRemainingWindowAndExplicitOutputCanExceed256()
    {
        var auto = Parse("""{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}]}""");
        Assert.True(auto.Inference.UseRemainingContext);
        var explicitLimit = Parse("""{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":4096}""");
        Assert.False(explicitLimit.Inference.UseRemainingContext); Assert.Equal(4096, explicitLimit.Inference.MaxTokens);
    }
    [Fact] public void TelemetryDoesNotRouteToLoadingWorkers()
    {
        var registry = new WorkerRegistry();
        var r = new WorkerRegistration(1,"w","http://localhost:5201",[new("m","metal","m","m",null,true,true,"")],[new(Protocol.Qwen36Model,"test-chat-v1",false)],1,
            new(DateTimeOffset.UtcNow,"Warming",false,false,0,2,0,1,"build"));
        registry.Register(r,false);
        Assert.Single(registry.Snapshot().Workers); Assert.Null(registry.Select(new("hi",1,Protocol.Qwen36Model,"metal")));
        registry.Register(r with { Telemetry = r.Telemetry! with { State="Ready", AcceptingRequests=true } },false);
        Assert.NotNull(registry.Select(new("hi",1,Protocol.Qwen36Model,"metal")));
        var json=JsonSerializer.Serialize(registry.Snapshot(), RpcJson.Default.ClusterSnapshot);
        Assert.Equal("Ready",JsonSerializer.Deserialize(json,RpcJson.Default.ClusterSnapshot)!.Workers[0].Registration.Telemetry!.State);
    }
    [Fact] public void NeverRoutesStructuredChatToOldWorker()
    {
        var registry = new WorkerRegistry();
        registry.Register(new(1,"old","http://127.0.0.1:5301",[new("m","metal","m","m",null,true,true,"")],[new(Protocol.Qwen36Model,"mlx-qwen36-affine",false)],1),false);
        var chat = new InferenceRequest("",32,Protocol.Qwen36Model,"metal",Messages:[new("user","hi")]);
        Assert.Null(registry.Select(chat)); Assert.NotNull(registry.Select(chat with { Prompt="hi",Messages=null }));
    }
}
