using System.Diagnostics;
using DeepInfer.Core;
using DeepInfer.Runtime;
namespace DeepInfer.Host;

public static class Demo
{
    public static async Task RunAsync(Options options, CancellationToken ct)
    {
        var port = options.Number("base-port", 5100, 1024, 65000);
        var backend = options.Get("backend", "metal");
        var model = options.Get("model", backend == "simulation" ? Protocol.SyntheticModel : Protocol.Qwen36Model);
        var count = options.Number("workers", model == Protocol.Qwen36Model ? 1 : 2, 1, 8);
        var endpoint = $"http://127.0.0.1:{port}";
        var weights = options.Get("weights", "models/qwen3.6-35b-a3b-8bit-mlx");
        var tokenizer = options.Get("tokenizer", Path.Combine(weights, "tokenizer.json"));
        var children = new List<Process>();
        var logs = new List<Task>();
        try
        {
            Start("coordinator", "--listen", endpoint);
            using var client = new ClusterClient(endpoint, options.Token, TimeSpan.FromMinutes(5));
            await WaitFor(async () => (await client.HealthAsync(ct)).Role == "coordinator", ct);
            for (var i = 0; i < count; i++)
            {
                var workerArgs = new List<string> { "worker", "--id", $"demo-{i+1}", "--listen", $"http://127.0.0.1:{port+i+1}", "--coordinator", endpoint, "--backend", backend };
                if (backend != "simulation") workerArgs.AddRange(["--weights", weights, "--tokenizer", tokenizer, "--model", model]);
                Start(workerArgs.ToArray());
            }
            await WaitFor(async () => (await client.StatusAsync(ct)).Workers.Count(x => x.Healthy) == count, ct);
            Console.WriteLine($"Cluster ready: coordinator + {count} independent worker processes. Model={model}; backend={backend}; simulation={backend == "simulation"}");
            if (options.Flag("stay"))
            {
                Console.WriteLine($"Cluster stays online at {endpoint}. Ctrl+C stops all child processes.");
                while (true)
                {
                    await Task.Delay(1000, ct);
                    if (children.Any(p => p.HasExited)) throw new InvalidOperationException("A cluster child exited.");
                }
            }
            for (var i = 0; i < count * 2; i++)
            {
                var result = await client.InferAsync(new(options.Get("prompt", model == Protocol.Qwen36Model ? "用一句话介绍你自己。" : "Once upon a time"), backend == "simulation" ? 8 : 32, model, backend), ct);
                Console.WriteLine($"{result.WorkerId}: {result.Text}");
            }
            var benchmark = await Commands.BenchmarkAsync(client, backend == "simulation" ? 20 : 3, 1, 16, ct, model, backend);
            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(benchmark, RpcJson.Default.BenchmarkResult));
        }
        finally
        {
            foreach (var child in children.AsEnumerable().Reverse())
            {
                if (!child.HasExited) child.Kill(entireProcessTree: true);
                await child.WaitForExitAsync(CancellationToken.None);
                child.Dispose();
            }
            await Task.WhenAll(logs);
        }
        void Start(params string[] args)
        {
            var executable = Environment.ProcessPath ?? throw new InvalidOperationException("Cannot locate executable.");
            var start = new ProcessStartInfo(executable) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };
            // Supports dotnet-hosted DLL, generated apphost, and the Native AOT executable.
            if (Path.GetFileNameWithoutExtension(executable).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
                start.ArgumentList.Add(Path.Combine(AppContext.BaseDirectory, "deepinfer.dll"));
            foreach (var arg in args) start.ArgumentList.Add(arg);
            if (options.Token is not null) start.Environment["DEEPINFER_TOKEN"] = options.Token;
            var process = Process.Start(start) ?? throw new InvalidOperationException("Unable to launch child process.");
            children.Add(process);
            logs.Add(Drain(process.StandardOutput)); logs.Add(Drain(process.StandardError));
        }
        static async Task Drain(StreamReader reader)
        {
            while (await reader.ReadLineAsync() is { } line) Console.Error.WriteLine($"[child] {line}");
        }
        async Task WaitFor(Func<Task<bool>> predicate, CancellationToken token)
        {
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(token);
            deadline.CancelAfter(TimeSpan.FromMinutes(3));
            while (true)
            {
                deadline.Token.ThrowIfCancellationRequested();
                if (children.Any(p => p.HasExited)) throw new InvalidOperationException("A demo child exited; inspect child logs (possible port conflict).");
                try { if (await predicate()) return; }
                catch (Exception ex) when (ex is HttpRequestException or RpcException or OperationCanceledException)
                { token.ThrowIfCancellationRequested(); }
                await Task.Delay(100, deadline.Token);
            }
        }
    }
}
