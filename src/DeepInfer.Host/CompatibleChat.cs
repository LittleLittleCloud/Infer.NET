using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepInfer.Core;
using DeepInfer.Runtime;
namespace DeepInfer.Host;

internal static class CompatibleChat
{
    public static bool IsPath(PathString path) => path == "/v1/chat/completions" || path == "/v1/messages" || path == "/v1/responses" || path == "/v1/models";
    public static void Map(WebApplication app, WorkerRegistry registry, string? token)
    {
        foreach (var (path, format) in new[] { ("/v1/chat/completions", ChatFormat.Completions), ("/v1/messages", ChatFormat.Messages), ("/v1/responses", ChatFormat.Responses) })
            app.MapPost(path, (HttpContext c) => Handle(c, registry, token, format));
        app.MapGet("/v1/models", async (HttpContext c) => {
            var models = registry.Snapshot().Workers.Where(w => w.Healthy && w.Registration.Telemetry?.AcceptingRequests != false).SelectMany(w => w.Registration.Models).Where(m => m.Format.EndsWith("-chat-v1", StringComparison.Ordinal)).Select(m => m.Id).Distinct();
            await Json(c, new JsonObject { ["object"] = "list", ["data"] = new JsonArray(models.Select(m => (JsonNode)new JsonObject { ["id"] = m, ["object"] = "model", ["created"] = 0, ["owned_by"] = "infer.net" }).ToArray()) });
        });
    }
    public static async Task Error(HttpContext c, ChatFormat format, int status, string message, JsonObject? usage = null, JsonObject? observedUsage = null, string? code = null)
    {
        var kind = status switch { 401 => "authentication_error", 429 => "rate_limit_error", >= 500 => "api_error", _ => "invalid_request_error" };
        var body = new JsonObject { ["error"] = new JsonObject { ["type"] = kind, ["message"] = message } };
        if (code != null) body["error"]!["code"] = code;
        if (usage != null) body["usage"] = usage;
        if (observedUsage != null) body["observed_usage"] = observedUsage;
        if (format == ChatFormat.Messages) body["type"] = "error";
        if (c.Response.HasStarted)
        {
            if (format == ChatFormat.Responses) body = new() { ["type"] = "error", ["sequence_number"] = c.Items["chat-sequence"] is int sequence ? sequence : 0, ["code"] = code ?? kind, ["message"] = message, ["param"] = null };
            await Sse(c, body, format == ChatFormat.Completions ? null : "error");
        }
        else { c.Response.StatusCode = status; await Json(c, body); }
    }
    private static async Task Json(HttpContext c, JsonObject value)
    { c.Response.ContentType = "application/json"; await c.Response.WriteAsync(value.ToJsonString(), c.RequestAborted); }
    private static async Task Sse(HttpContext c, JsonObject value, string? name = null)
    {
        if (!c.Response.HasStarted) { c.Response.ContentType = "text/event-stream"; c.Response.Headers.CacheControl = "no-cache"; }
        if (name != null) await c.Response.WriteAsync("event: " + name + "\n", c.RequestAborted);
        await c.Response.WriteAsync("data: " + value.ToJsonString() + "\n\n", c.RequestAborted);
        await c.Response.Body.FlushAsync(c.RequestAborted);
    }
    private static async Task Handle(HttpContext c, WorkerRegistry registry, string? token, ChatFormat format)
    {
        Writer? writer = null;
        var dispatched = false;
        try
        {
            var limit = c.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>();
            if (limit is { IsReadOnly: false }) limit.MaxRequestBodySize = Protocol.MaxRequestBytes;
            using var doc = await JsonDocument.ParseAsync(c.Request.Body, new JsonDocumentOptions { MaxDepth = 32 }, c.RequestAborted);
            var input = ChatRequest.Parse(doc.RootElement, format);
            writer = new Writer(c, format, input);
            var worker = registry.Select(input.Inference);
            if (worker == null) { await Error(c, format, 503, "No ready chat-v1 worker. Upgrade the worker and wait for readiness."); return; }
            using var http = new HttpClient { Timeout = TimeSpan.FromHours(24) };
            using var request = new HttpRequestMessage(HttpMethod.Post, worker.Endpoint.TrimEnd('/') + (input.Stream ? "/v1/infer/stream" : "/v1/infer")) { Content = JsonContent.Create(input.Inference, RpcJson.Default.InferenceRequest) };
            if (token != null) request.Headers.Add("X-DeepInfer-Token", token);
            dispatched = true;
            using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, c.RequestAborted);
            if (!response.IsSuccessStatusCode) { await Error(c, format, (int)response.StatusCode, await response.Content.ReadAsStringAsync(c.RequestAborted)); return; }
            if (!input.Stream)
            {
                var result = await response.Content.ReadFromJsonAsync(RpcJson.Default.InferenceResult, c.RequestAborted) ?? throw new IOException("Empty worker result.");
                writer.Record(result);
                await Json(c, writer.Complete(result)); return;
            }
            using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(c.RequestAborted));
            var completed = false; var started = false;
            while (await reader.ReadLineAsync(c.RequestAborted) is { } line)
            {
                if (!line.StartsWith("data: ", StringComparison.Ordinal)) continue;
                var update = JsonSerializer.Deserialize(line[6..], ServiceJson.Default.ServiceEvent) ?? throw new IOException("Invalid worker stream.");
                writer.Observe(update);
                if (update.Error != null) throw new IOException(update.Error.Message);
                if (!started) { await writer.Start(update.PromptTokens ?? update.Result?.PromptTokens ?? throw new IOException("Missing prompt usage metadata.")); started = true; }
                // Tool-enabled requests are buffered until the model has finished so
                // Qwen's XML call is never leaked as plain assistant text.
                if (input.Tools == null && update.Text is { Length: > 0 } text) await writer.Delta(text);
                if (update.Result is { } result) { await writer.End(result); completed = true; break; }
            }
            if (!completed) throw new IOException("Worker stream ended without completion; partial output is not retried.");
        }
        catch (OperationCanceledException) when (c.RequestAborted.IsCancellationRequested) { }
        catch (ModelOutputException)
        { await writer!.Fail("Invalid model output: malformed tool call.", "invalid_model_output"); }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException or BadHttpRequestException)
        {
            if (!dispatched) await Error(c, format, ex is BadHttpRequestException b ? b.StatusCode : 400, ex.Message);
            else await writer!.Fail("Invalid worker response.", "invalid_worker_response");
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or InvalidDataException or OperationCanceledException)
        {
            if (writer != null) await writer.Fail(ex is InvalidDataException ? "Invalid worker accounting or output metadata." : "Worker request failed: " + ex.Message,
                ex is InvalidDataException ? "invalid_worker_response" : "worker_request_failed");
            else await Error(c, format, 502, "Worker request failed.");
        }
    }
    private sealed class ModelOutputException(InvalidDataException inner) : Exception("Invalid model tool call.", inner);
    private sealed class Writer(HttpContext c, ChatFormat format, ChatRequest input)
    {
        private readonly string id = (format == ChatFormat.Completions ? "chatcmpl_" : format == ChatFormat.Messages ? "msg_" : "resp_") + Guid.NewGuid().ToString("N");
        private readonly string itemId = "msg_" + Guid.NewGuid().ToString("N");
        private readonly long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        private int sequence;
        private int? promptTokens;
        private int observedTokens;
        private InferenceResult? terminal;
        private string Model => input.Inference.Model;
        private static int Count(InferenceResult r) => r.TokenIds?.Length ?? r.Tokens.Length;
        private static JsonObject Usage(InferenceResult r, bool chat = false) => chat
            ? new() { ["prompt_tokens"] = r.PromptTokens, ["completion_tokens"] = Count(r), ["total_tokens"] = r.PromptTokens + Count(r) }
            : new() { ["input_tokens"] = r.PromptTokens, ["output_tokens"] = Count(r), ["total_tokens"] = r.PromptTokens + Count(r) };
        public void Observe(ServiceEvent update)
        {
            if (update.PromptTokens is { } count)
            {
                if (count <= 0 || count > Protocol.Qwen36Context || promptTokens is { } prior && prior != count)
                    throw new InvalidDataException("Inconsistent worker prompt usage.");
                promptTokens = count;
            }
            if (update.TokenId is { } token)
            {
                if (token < 0) throw new InvalidDataException("Invalid worker token.");
                observedTokens++;
            }
            if (update.Result is { } result) Record(result);
        }
        public void Record(InferenceResult result)
        {
            if (result.PromptTokens <= 0 || result.PromptTokens > Protocol.Qwen36Context ||
                result.TokenIds == null || result.TokenIds.Any(t => t < 0) ||
                result.TokenIds.Length > input.Inference.MaxTokens ||
                result.TokenIds.Length > Protocol.Qwen36Context - result.PromptTokens ||
                promptTokens is { } count && count != result.PromptTokens ||
                input.Stream && observedTokens != result.TokenIds.Length)
                throw new InvalidDataException("Invalid terminal worker accounting or finish reason.");
            terminal = result;
            if (result.Text == null || result.FinishReason is not ("eos" or "length"))
                throw new InvalidDataException("Invalid worker output metadata.");
        }
        public async Task Fail(string message, string code)
        {
            c.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("CompatibleChat")
                .LogWarning("Chat request failed ({Code}): {Message}", code, message);
            var usage = terminal == null ? null : Usage(terminal, format == ChatFormat.Completions);
            var observed = terminal == null ? new JsonObject { ["input_tokens"] = promptTokens,
                ["output_tokens_at_least"] = observedTokens, ["complete"] = false } : null;
            // Usage must precede the error: OpenAI clients throw as soon as they
            // consume an error event. No success finish_reason or [DONE] follows.
            if (c.Response.HasStarted && format == ChatFormat.Completions && input.IncludeUsage && usage != null)
                await UsageChunk(terminal!);
            if (c.Response.HasStarted && format == ChatFormat.Responses)
            {
                var failed = Response("failed");
                failed["error"] = new JsonObject { ["code"] = code, ["message"] = message };
                failed["usage"] = usage;
                failed["observed_usage"] = observed;
                await Event("response.failed", new() { ["response"] = failed });
                return;
            }
            await Error(c, format, 502, message, usage, observed, code);
        }
        private JsonObject Part(string text) => new() { ["type"] = "output_text", ["text"] = text, ["annotations"] = new JsonArray(), ["logprobs"] = new JsonArray() };
        private JsonObject Item(string text, string status) => new() { ["id"] = itemId, ["type"] = "message", ["role"] = "assistant", ["status"] = status, ["content"] = new JsonArray(Part(text)) };
        private JsonObject Response(string status, InferenceResult? r = null) => new() {
            ["id"] = id, ["object"] = "response", ["created_at"] = created, ["status"] = status, ["model"] = Model,
            ["error"] = null, ["incomplete_details"] = status == "incomplete" ? new JsonObject { ["reason"] = "max_output_tokens" } : null,
            ["output"] = r == null ? new JsonArray() : new JsonArray(Item(r.Text, status)), ["usage"] = r == null ? null : Usage(r),
            ["parallel_tool_calls"] = false, ["tool_choice"] = "none", ["tools"] = new JsonArray(), ["store"] = false,
            ["temperature"] = 0, ["top_p"] = 1, ["metadata"] = new JsonObject(), ["max_output_tokens"] = input.Inference.MaxTokens
        };
        private (string Text, ChatToolCall[] Calls) Parsed(InferenceResult r)
        {
            try { return input.Tools is { } tools ? ChatTools.ParseOutput(r.Text, tools) : (r.Text, []); }
            catch (InvalidDataException ex) { throw new ModelOutputException(ex); }
        }
        private static JsonArray ToolCalls(ChatToolCall[] calls) => new(calls.Select(call => (JsonNode)new JsonObject {
            ["id"] = call.Id, ["type"] = "function", ["function"] = new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments }
        }).ToArray());
        public JsonObject Complete(InferenceResult r)
        {
            var (text, calls) = Parsed(r);
            return format switch {
            ChatFormat.Completions => new() { ["id"] = id, ["object"] = "chat.completion", ["created"] = created, ["model"] = Model, ["choices"] = new JsonArray(new JsonObject { ["index"] = 0, ["message"] = new JsonObject { ["role"] = "assistant", ["content"] = calls.Length > 0 && text.Length == 0 ? null : text, ["tool_calls"] = calls.Length > 0 ? ToolCalls(calls) : null, ["refusal"] = null }, ["finish_reason"] = calls.Length > 0 ? "tool_calls" : r.FinishReason == "eos" ? "stop" : "length", ["logprobs"] = null }), ["usage"] = Usage(r, true) },
            ChatFormat.Messages => new() { ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = Model, ["content"] = new JsonArray(new JsonObject { ["type"] = "text", ["text"] = r.Text }), ["stop_reason"] = r.FinishReason == "eos" ? "end_turn" : "max_tokens", ["stop_sequence"] = null, ["usage"] = new JsonObject { ["input_tokens"] = r.PromptTokens, ["output_tokens"] = Count(r) } },
            _ => Response(r.FinishReason == "eos" ? "completed" : "incomplete", r)
            };
        }
        private Task Event(string name, JsonObject data)
        { data["type"] = name; if (format == ChatFormat.Responses) { data["sequence_number"] = sequence++; c.Items["chat-sequence"] = sequence; } return Sse(c, data, name); }
        private JsonObject Chunk(JsonObject delta, string? finish = null) => new() { ["id"] = id, ["object"] = "chat.completion.chunk", ["created"] = created, ["model"] = Model, ["choices"] = new JsonArray(new JsonObject { ["index"] = 0, ["delta"] = delta, ["finish_reason"] = finish, ["logprobs"] = null }) };
        private Task UsageChunk(InferenceResult result)
        {
            var chunk = Chunk(new());
            chunk["choices"] = new JsonArray();
            chunk["usage"] = Usage(result, true);
            return Sse(c, chunk);
        }
        public async Task Start(int promptTokens)
        {
            if (format == ChatFormat.Completions) await Sse(c, Chunk(new() { ["role"] = "assistant", ["content"] = "" }));
            else if (format == ChatFormat.Messages)
            {
                await Event("message_start", new() { ["message"] = new JsonObject { ["id"] = id, ["type"] = "message", ["role"] = "assistant", ["model"] = Model, ["content"] = new JsonArray(), ["stop_reason"] = null, ["stop_sequence"] = null, ["usage"] = new JsonObject { ["input_tokens"] = promptTokens, ["output_tokens"] = 0 } } });
                await Event("content_block_start", new() { ["index"] = 0, ["content_block"] = new JsonObject { ["type"] = "text", ["text"] = "" } });
            }
            else
            {
                await Event("response.created", new() { ["response"] = Response("in_progress") });
                await Event("response.in_progress", new() { ["response"] = Response("in_progress") });
                var item = Item("", "in_progress"); item["content"] = new JsonArray();
                await Event("response.output_item.added", new() { ["output_index"] = 0, ["item"] = item });
                await Event("response.content_part.added", new() { ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["part"] = Part("") });
            }
        }
        public Task Delta(string text) => format switch {
            ChatFormat.Completions => Sse(c, Chunk(new() { ["content"] = text })),
            ChatFormat.Messages => Event("content_block_delta", new() { ["index"] = 0, ["delta"] = new JsonObject { ["type"] = "text_delta", ["text"] = text } }),
            _ => Event("response.output_text.delta", new() { ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["delta"] = text, ["logprobs"] = new JsonArray() })
        };
        public async Task End(InferenceResult r)
        {
            if (format == ChatFormat.Completions)
            {
                var (text, calls) = Parsed(r);
                if (input.Tools != null)
                {
                    if (text.Length > 0) await Delta(text);
                    for (var index = 0; index < calls.Length; index++)
                    {
                        var call = calls[index];
                        await Sse(c, Chunk(new() { ["tool_calls"] = new JsonArray(new JsonObject {
                            ["index"] = index, ["id"] = call.Id, ["type"] = "function",
                            ["function"] = new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments }
                        }) }));
                    }
                }
                await Sse(c, Chunk(new(), calls.Length > 0 ? "tool_calls" : r.FinishReason == "eos" ? "stop" : "length"));
                if (input.IncludeUsage) await UsageChunk(r);
                await c.Response.WriteAsync("data: [DONE]\n\n", c.RequestAborted); await c.Response.Body.FlushAsync(c.RequestAborted);
            }
            else if (format == ChatFormat.Messages)
            {
                await Event("content_block_stop", new() { ["index"] = 0 });
                await Event("message_delta", new() { ["delta"] = new JsonObject { ["stop_reason"] = r.FinishReason == "eos" ? "end_turn" : "max_tokens", ["stop_sequence"] = null }, ["usage"] = new JsonObject { ["input_tokens"] = r.PromptTokens, ["output_tokens"] = Count(r) } });
                await Event("message_stop", new());
            }
            else
            {
                var status = r.FinishReason == "eos" ? "completed" : "incomplete";
                await Event("response.output_text.done", new() { ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["text"] = r.Text, ["logprobs"] = new JsonArray() });
                await Event("response.content_part.done", new() { ["item_id"] = itemId, ["output_index"] = 0, ["content_index"] = 0, ["part"] = Part(r.Text) });
                await Event("response.output_item.done", new() { ["output_index"] = 0, ["item"] = Item(r.Text, status) });
                await Event("response." + status, new() { ["response"] = Complete(r) });
            }
        }
    }
}
