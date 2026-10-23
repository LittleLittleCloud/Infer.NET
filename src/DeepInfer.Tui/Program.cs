using DeepInfer.Runtime;
using Microsoft.Extensions.AI;
using DeepInfer.Tui;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RazorConsole.Core;

var endpoint = args.Length == 0 ? "http://127.0.0.1:5200" : args[0];
if (Console.IsInputRedirected || Console.IsOutputRedirected)
{
    Console.Error.WriteLine("TUI requires an interactive terminal. Use 'deepinfer status' for JSON output.");
    return 1;
}
try
{
    // With no endpoint override, reuse the existing repository deployment configuration.
    var config = Path.GetFullPath("artifacts/deploy/private/mac/coordinator.json");
    string? configuredToken = null;
    if (args.Length == 0 && File.Exists(config))
    {
        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(config));
        endpoint = document.RootElement.GetProperty("listen").GetString()!;
        configuredToken = File.ReadAllText(document.RootElement.GetProperty("token-file").GetString()!).Trim();
    }
    var token = Environment.GetEnvironmentVariable("DEEPINFER_TOKEN_FILE") is { Length: > 0 } tokenFile ? File.ReadAllText(tokenFile).Trim() : Environment.GetEnvironmentVariable("DEEPINFER_TOKEN") ?? configuredToken;
    using var client = new ClusterClient(endpoint, token);
    using IChatClient chat = new InferChatClient(endpoint, token);
    var host = Host.CreateDefaultBuilder()
        .ConfigureServices(services => services.AddSingleton(client).AddSingleton(chat))
        .UseRazorConsole<Dashboard>().Build();
    await host.RunAsync();
    return 0;
}
catch (Exception ex) when (ex is ArgumentException or HttpRequestException or IOException or System.Text.Json.JsonException)
{ Console.Error.WriteLine(ex.Message); return 1; }
