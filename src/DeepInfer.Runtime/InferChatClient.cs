using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepInfer.Core;
using Microsoft.Extensions.AI;
namespace DeepInfer.Runtime;

// Uses the public HTTP surface, so playground and other IChatClient consumers share protocol behavior.
public sealed class InferChatClient : IChatClient
{
    private readonly HttpClient http;
    private readonly ChatClientMetadata metadata;
    private readonly bool ownsClient;
    public InferChatClient(string endpoint = "http://127.0.0.1:5200", string? token = null)
    {
        http = new(new SocketsHttpHandler { AllowAutoRedirect = false }) { BaseAddress = EndpointRules.Validate(endpoint, true), Timeout = TimeSpan.FromMinutes(10) };
        if (!string.IsNullOrEmpty(token)) http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        metadata = new("Infer.NET", http.BaseAddress, Protocol.Qwen36Model); ownsClient = true;
    }
    public InferChatClient(HttpClient client)
    {
        http = client; metadata = new("Infer.NET", EndpointRules.Validate(client.BaseAddress?.ToString() ?? "", true), Protocol.Qwen36Model);
    }
    public object? GetService(Type serviceType, object? serviceKey = null) => serviceKey != null ? null : serviceType == typeof(ChatClientMetadata) ? metadata : serviceType.IsInstanceOfType(this) ? this : null;
    private static StringContent Body(IEnumerable<ChatMessage> messages, ChatOptions? options, bool stream)
    {
        if (options is { Tools.Count: > 0 } || options?.ResponseFormat != null || options?.StopSequences is { Count: > 0 } || options?.ConversationId != null || options?.AdditionalProperties is { Count: > 0 } || options?.Seed != null || options?.FrequencyPenalty != null || options?.PresencePenalty != null || options?.TopK != null || options?.ToolMode != null || options?.AllowMultipleToolCalls != null || options?.RawRepresentationFactory != null || options?.Reasoning != null || options?.AllowBackgroundResponses != null || options?.ContinuationToken != null)
            throw new NotSupportedException("Only stateless text chat, model, output limit and greedy sampling options are supported.");
        var history = new JsonArray();
        if (options?.Instructions is { Length: > 0 } instructions) history.Add((JsonNode)new JsonObject { ["role"] = "system", ["content"] = instructions });
        foreach (var m in messages)
        {
            if (m.Contents.Any(c => c is not TextContent) || m.AuthorName != null) throw new NotSupportedException("Only unnamed text messages are supported.");
            history.Add((JsonNode)new JsonObject { ["role"] = m.Role.Value, ["content"] = m.Text });
        }
        return new(new JsonObject { ["model"] = options?.ModelId ?? Protocol.Qwen36Model, ["messages"] = history, ["max_completion_tokens"] = options?.MaxOutputTokens ?? 128,
            ["temperature"] = options?.Temperature ?? 0, ["top_p"] = options?.TopP ?? 1, ["stream"] = stream,
            ["stream_options"] = stream ? new JsonObject { ["include_usage"] = true } : null
        }.WithoutNulls().ToJsonString(), Encoding.UTF8, "application/json");
    }
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        using var content = Body(messages, options, false);
        using var response = await http.PostAsync("/v1/chat/completions", content, cancellationToken);
        await Check(response, cancellationToken);
        using var doc = await JsonDocument.ParseAsync(await response.Content.ReadAsStreamAsync(cancellationToken), cancellationToken: cancellationToken);
        var r = doc.RootElement; var choice = r.GetProperty("choices")[0];
        return new(new ChatMessage(ChatRole.Assistant, choice.GetProperty("message").GetProperty("content").GetString()))
        { ResponseId = r.GetProperty("id").GetString(), ModelId = r.GetProperty("model").GetString(), FinishReason = Finish(choice.GetProperty("finish_reason").GetString()), Usage = Usage(r.GetProperty("usage")) };
    }
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions") { Content = Body(messages, options, true) };
        using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        await Check(response, cancellationToken);
        using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(cancellationToken)); var done = false; var finished = false;
        while (await reader.ReadLineAsync(cancellationToken) is { } line)
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
            if (line[6..] == "[DONE]") { done = true; break; }
            using var doc = JsonDocument.Parse(line[6..]); var r = doc.RootElement;
            if (r.TryGetProperty("error", out var error)) throw new IOException(error.GetProperty("message").GetString());
            var update = new ChatResponseUpdate { Role = ChatRole.Assistant, ResponseId = r.GetProperty("id").GetString(), ModelId = r.GetProperty("model").GetString() };
            foreach (var choice in r.GetProperty("choices").EnumerateArray())
            {
                if (choice.GetProperty("delta").TryGetProperty("content", out var text) && text.ValueKind == JsonValueKind.String) update.Contents.Add(new TextContent(text.GetString()!));
                if (choice.GetProperty("finish_reason").ValueKind == JsonValueKind.String) { update.FinishReason = Finish(choice.GetProperty("finish_reason").GetString()); finished = true; }
            }
            if (r.TryGetProperty("usage", out var usage) && usage.ValueKind == JsonValueKind.Object) update.Contents.Add(new UsageContent(Usage(usage)));
            yield return update;
        }
        if (!done || !finished) throw new IOException("Incomplete chat stream; partial output must not be committed to conversation history.");
    }
    private static ChatFinishReason Finish(string? reason) => reason == "stop" ? ChatFinishReason.Stop : ChatFinishReason.Length;
    private static UsageDetails Usage(JsonElement u) => new() { InputTokenCount = u.GetProperty("prompt_tokens").GetInt64(), OutputTokenCount = u.GetProperty("completion_tokens").GetInt64(), TotalTokenCount = u.GetProperty("total_tokens").GetInt64() };
    private static async Task Check(HttpResponseMessage r, CancellationToken ct) { if (!r.IsSuccessStatusCode) throw new RpcException((int)r.StatusCode, await r.Content.ReadAsStringAsync(ct)); }
    public void Dispose() { if (ownsClient) http.Dispose(); }
}
internal static class ChatJsonExtensions
{
    public static JsonObject WithoutNulls(this JsonObject value) { foreach (var key in value.Where(p => p.Value == null).Select(p => p.Key).ToArray()) value.Remove(key); return value; }
}
