using System.Runtime.InteropServices;
using DeepInfer.Host;
using DeepInfer.Runtime;
using var shutdown = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; shutdown.Cancel(); };
using var terminate = OperatingSystem.IsWindows() ? null :
    PosixSignalRegistration.Create(PosixSignal.SIGTERM, context => { context.Cancel = true; shutdown.Cancel(); });
try { return args.Contains("--windows-service") ? await WindowsLifetime.Run(args) : await Commands.RunAsync(args, shutdown.Token); }
catch (OperationCanceledException) { Console.Error.WriteLine("Cancelled."); return 130; }
catch (Exception ex) when (ex is ArgumentException or HttpRequestException or RpcException or InvalidOperationException or IOException or NotSupportedException or DllNotFoundException)
{ Console.Error.WriteLine($"Error: {ex.Message}"); return 1; }
