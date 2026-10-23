namespace DeepInfer.Host;

internal static class WindowsLifetime
{
    public static async Task<int> Run(string[] args)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Windows service mode requires Windows.");
        var builder = Microsoft.Extensions.Hosting.Host.CreateApplicationBuilder();
        builder.Services.AddWindowsService(o => o.ServiceName = "InferNetCuda");
        builder.Services.AddSingleton(new Options(args));
        builder.Services.AddHostedService<Agent>();
        using var host = builder.Build(); await host.RunAsync(); return Environment.ExitCode;
    }
    private sealed class Agent(Options options, IHostApplicationLifetime lifetime, ILogger<Agent> logger) : BackgroundService
    {
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            try
            {
                if (options.Command != "split-agent") throw new ArgumentException("Windows service mode supports split-agent only.");
                await SplitService.RunAgent(options, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested) { }
            catch (Exception ex) { logger.LogError(ex, "CUDA service failed"); Environment.Exit(1); }
            finally { lifetime.StopApplication(); }
        }
    }
}
