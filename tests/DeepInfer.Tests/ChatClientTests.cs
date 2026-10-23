using System.Net;
using System.Text;
using System.Text.Json;
using DeepInfer.Runtime;
using Microsoft.Extensions.AI;
using Xunit;
namespace DeepInfer.Tests;

public sealed class ChatClientTests
{
    private sealed class Handler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> handle) : HttpMessageHandler
    { protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct) => handle(request, ct); }
    private static HttpResponseMessage Response(string text) => new(HttpStatusCode.OK) { Content = new StringContent(text, Encoding.UTF8, "text/event-stream") };
    private const string Chunk = """{"id":"chatcmpl_test","model":"qwen3.6-35b-a3b","choices":[{"delta":{"content":"你好"},"finish_reason":null}]}""";
    private const string Finish = """{"id":"chatcmpl_test","model":"qwen3.6-35b-a3b","choices":[{"delta":{},"finish_reason":"stop"}]}""";
    private const string Usage = """{"id":"chatcmpl_test","model":"qwen3.6-35b-a3b","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}""";
    [Fact] public async Task StreamingAdapterPreservesHistoryInstructionsUsageAndFinish()
    {
        using var http = new HttpClient(new Handler(async (r, ct) => {
            Assert.Equal("/v1/chat/completions",r.RequestUri!.AbsolutePath);
            using var json = JsonDocument.Parse(await r.Content!.ReadAsStringAsync(ct)); var messages = json.RootElement.GetProperty("messages");
            Assert.Equal("system", messages[0].GetProperty("role").GetString()); Assert.Equal("prior", messages[2].GetProperty("content").GetString());
            return Response($"data: {Chunk}\n\ndata: {Finish}\n\ndata: {Usage}\n\ndata: [DONE]\n\n");
        })) { BaseAddress = new("http://localhost:5200") };
        using IChatClient client = new InferChatClient(http);
        var updates = new List<ChatResponseUpdate>();
        await foreach(var u in client.GetStreamingResponseAsync([new(ChatRole.User,"old"),new(ChatRole.Assistant,"prior"),new(ChatRole.User,"next")],new() { Instructions="system",MaxOutputTokens=32 })) updates.Add(u);
        Assert.Equal("你好",string.Concat(updates.Select(u=>u.Text)));
        Assert.Equal(ChatFinishReason.Stop, updates[1].FinishReason);
        Assert.Equal(14, updates[2].Contents.OfType<UsageContent>().Single().Details.TotalTokenCount);
    }
    [Theory] [InlineData(false)] [InlineData(true)]
    public async Task TruncatedAndErrorStreamsDoNotComplete(bool error)
    {
        using var http = new HttpClient(new Handler((_,_)=>Task.FromResult(Response($"data: {Chunk}\n\n"+(error?"data: {\"error\":{\"message\":\"failed\"}}\n\n":""))))) { BaseAddress=new("http://localhost:5200") };
        using IChatClient client = new InferChatClient(http);
        await Assert.ThrowsAsync<IOException>(async()=> { await foreach(var _ in client.GetStreamingResponseAsync([new(ChatRole.User,"hi")])) {} });
    }
    [Fact] public async Task UnsupportedToolsNeverReachServer()
    {
        var sent=false; using var http = new HttpClient(new Handler((_,_)=> { sent=true;return Task.FromResult(Response("")); })) { BaseAddress=new("http://localhost:5200") };
        using IChatClient client=new InferChatClient(http);
        await Assert.ThrowsAsync<NotSupportedException>(()=>client.GetResponseAsync([new(ChatRole.User,"hi")],new(){ ConversationId="stored-id" })); Assert.False(sent);
    }
}
