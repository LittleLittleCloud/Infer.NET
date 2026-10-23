using DeepInfer.Runtime;
using DeepInfer.Tui;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using RazorConsole.Core;

var endpoint = args.Length == 0 ? "http://127.0.0.1:5100" : args[0];
if (Console.IsInputRedirected || Console.IsOutputRedirected)
{
    Console.Error.WriteLine("TUI requires an interactive terminal. Use 'deepinfer status' for JSON output.");
    return 1;
}
try
{
    using var client = new ClusterClient(endpoint, Environment.GetEnvironmentVariable("DEEPINFER_TOKEN_FILE") is { Length: > 0 } tokenFile ? File.ReadAllText(tokenFile).Trim() : Environment.GetEnvironmentVariable("DEEPINFER_TOKEN"));
    var host = Host.CreateDefaultBuilder()
        .ConfigureServices(services => services.AddSingleton(client))
        .UseRazorConsole<Dashboard>().Build();
    await host.RunAsync();
    return 0;
}
catch (Exception ex) when (ex is ArgumentException or HttpRequestException)
{ Console.Error.WriteLine(ex.Message); return 1; }
