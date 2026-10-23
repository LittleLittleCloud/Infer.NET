using System.Runtime.InteropServices;
using DeepInfer.Host;
using DeepInfer.Runtime;
using var shutdown = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) => { e.Cancel = true; shutdown.Cancel(); };
using var terminate = OperatingSystem.IsWindows() ? null :
    PosixSignalRegistration.Create(PosixSignal.SIGTERM, context => { context.Cancel = true; shutdown.Cancel(); });
try
{
    var logPath = new Options(args).Get("log-file", "");
    using var log = logPath.Length == 0 ? null : new StreamWriter(new FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite)) { AutoFlush = true };
    var originalOut = Console.Out; var originalError = Console.Error;
    if (log != null) { Console.SetOut(log); Console.SetError(log); }
    try { return args.Contains("--windows-service") ? await WindowsLifetime.Run(args) : await Commands.RunAsync(args, shutdown.Token); }
    catch (Exception ex) { Console.Error.WriteLine(ex); throw; }
    finally { Console.SetOut(originalOut); Console.SetError(originalError); }
}
catch (OperationCanceledException) { Console.Error.WriteLine("Cancelled."); return 130; }
catch (Exception ex) when (ex is ArgumentException or HttpRequestException or RpcException or InvalidOperationException or IOException or NotSupportedException or DllNotFoundException)
{ Console.Error.WriteLine($"Error: {ex.Message}"); return 1; }
