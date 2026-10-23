using System.Diagnostics;
using System.Text.Json;
using DeepInfer.Backends;
using DeepInfer.Core;
using DeepInfer.Runtime;
namespace DeepInfer.Host;

public static class Commands
{
    public static async Task<int> RunAsync(string[] args, CancellationToken ct)
    {
        var options = new Options(args);
        switch (options.Command)
        {
            case "version": Console.WriteLine(typeof(Commands).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false).Cast<System.Reflection.AssemblyInformationalVersionAttribute>().Single().InformationalVersion); break;
            case "split-check-suffix": await SuffixChecks.Run(options, ct); break;
            case "split-check-bf16-prefix": await Bf16PrefixReference.Run(options, ct); break;
            case "split-serve": await SplitService.RunMac(options, ct); break;
            case "split-agent": await SplitService.RunAgent(options, ct); break;
            case "split-worker": await SplitInference.Worker(options, ct); break;
            case "split-infer": await SplitInference.Coordinator(options, ct); break;
            case "coordinator": await Server.RunAsync(options, false, ct); break;
            case "worker": await Server.RunAsync(options, true, ct); break;
            case "demo": await Demo.RunAsync(options, ct); break;
            case "verify-model": await ModelVerification.RunAsync(options, ct); break;
            case "verify-qwen": await QwenVerification.RunAsync(options, ct); break;
            case "verify-qwen36": await Qwen36Verification.RunAsync(options, ct); break;
            case "qwen-tokenize":
                var moeTokenizer = options.Get("model", "") == Protocol.Qwen36Model;
                var qwenTokenizer = new DeepInfer.Backends.Models.QwenTokenizer(options.Get("tokenizer", moeTokenizer ? "models/qwen3.6-35b-a3b-5bit-mlx/tokenizer.json" : "models/qwen2.5-3b-instruct/tokenizer.json"), moeTokenizer);
                var text = options.Get("prompt-file", "") is { Length: > 0 } file ? await File.ReadAllTextAsync(file, ct) : options.Get("prompt", "你好");
                Console.WriteLine(JsonSerializer.Serialize(qwenTokenizer.EncodePrompt(text, ct), RpcJson.Default.Int32Array));
                break;
            case "devices":
                Console.WriteLine(JsonSerializer.Serialize(BackendCatalog.DiscoverAll(), RpcJson.Default.DeviceInfoArray)); break;
            case "status":
                using (var client = Client(options)) Console.WriteLine(JsonSerializer.Serialize(await client.StatusAsync(ct), RpcJson.Default.ClusterSnapshot));
                break;
            case "infer":
                if (options.Flag("stream")) { await ServiceStreaming.Client(options, ct); break; }
                using (var client = Client(options))
                {
                    var result = await client.InferAsync(new(options.Get("prompt", "用一句话介绍你自己。"),
                        options.Number("max-tokens", 48, 1, Protocol.MaxTokens), options.Get("model", options.Get("backend", "metal") == "simulation" ? Protocol.SyntheticModel : Protocol.Qwen3BModel), options.Get("backend", "metal")), ct);
                    Console.WriteLine(options.Flag("json") ? JsonSerializer.Serialize(result, RpcJson.Default.InferenceResult) : $"{result.Text}\nworker={result.WorkerId} backend={result.Backend} compute={result.ComputeMs:F3}ms");
                }
                break;
            case "bench":
                using (var client = Client(options))
                {
                    var result = await BenchmarkAsync(client, options.Number("requests", 100, 1, 100000),
                        options.Number("warmup", 5, 0, 1000), options.Number("max-tokens", 48, 1, Protocol.MaxTokens), ct,
                        options.Get("model", options.Get("backend", "metal") == "simulation" ? Protocol.SyntheticModel : Protocol.Qwen3BModel), options.Get("backend", "metal"),
                        options.Get("prompt", "请用中文详细解释什么是机器学习，至少写五百字。"));
                    var json = JsonSerializer.Serialize(result, RpcJson.Default.BenchmarkResult);
                    Console.WriteLine(json);
                    var output = options.Get("output", "");
                    if (output.Length > 0) await File.WriteAllTextAsync(output, json + Environment.NewLine, ct);
                }
                break;
            case "help": case "--help": case "-h":
                Console.WriteLine("""
                DeepInfer.NET — distributed Native AOT inference (Qwen 3B / 7B / Qwen3.6 MoE)
                  demo         [--workers 2] [--base-port 5100] [--stay]   launch, exercise, benchmark, stop
                  coordinator  [--listen http://127.0.0.1:5100]
                  worker       [--listen http://127.0.0.1:5101] [--id worker-1]
                               [--coordinator http://127.0.0.1:5100] [--backend simulation|cpu|metal|cuda]
                               [--capacity 1] [--advertise http://127.0.0.1:5101]
                               [--weights models/stories15M.bin --tokenizer models/tokenizer.bin --model stories15m]
                  split-serve  --config PRIVATE_MAC_WORKER_JSON   persistent Metal worker + API
                  split-agent  --config PRIVATE_WINDOWS_JSON      persistent CUDA agent
                  infer        --endpoint http://127.0.0.1:5200 --token-file FILE --stream
                  version      print build version and source commit
                  split-infer  --weights MODEL_DIR --listen tcp://MAC_INTERFACE:5210 --allow-remote
                               [--verify] [--max-tokens 8] [--output RUN_DIRECTORY]
                  split-worker --weights PREFIX_BUNDLE --endpoint tcp://MAC_INTERFACE:5210
                               experimental resident 20/20 handoff, <=2048 total tokens
                  devices      probe hardware and available compute backends
                  verify-qwen  [--weights models/qwen2.5-3b-instruct] [--reference artifacts/qwen-reference.json]
                  verify-qwen36 [--weights models/qwen3.6-35b-a3b-5bit-mlx]
                                [--reference artifacts/reference/qwen36/reference.json]
                  verify-model --weights models/stories15M.bin --tokenizer models/tokenizer.bin
                               compare CPU/Metal GEMV, full logits, and greedy generation
                  status       [--endpoint http://127.0.0.1:5100]
                  infer        [--endpoint ...] [--prompt "Hello"] [--max-tokens 48] [--json]
                               [--backend metal] [--model qwen2.5-3b-instruct]
                  bench        [--endpoint ...] [--requests 100] [--warmup 5] [--max-tokens 48]
                               [--output results.json] [--model stories15m --backend metal]
                Shared authentication: DEEPINFER_TOKEN environment variable or --token VALUE.
                Remote endpoints require --allow-remote on servers and a token. Use a trusted network.
                TUI: dotnet run --project src/DeepInfer.Tui -- http://127.0.0.1:5100
                """); break;
            default: throw new ArgumentException($"Unknown command '{options.Command}'. Run 'deepinfer help'.");
        }
        return 0;
    }
    private static ClusterClient Client(Options options) => new(options.Get("endpoint", "http://127.0.0.1:5100"), options.Token);
    public static async Task<BenchmarkResult> BenchmarkAsync(ClusterClient client, int requests, int warmup, int units, CancellationToken ct, string model = Protocol.SyntheticModel, string backend = "simulation", string prompt = "Once upon a time")
    {
        var request = new InferenceRequest(prompt, units, model, backend);
        for (var i = 0; i < warmup; i++) await client.InferAsync(request, ct);
        var samples = new double[requests];
        var workers = new HashSet<string>();
        var totalUnits = 0; var simulation = true; long dispatches = 0, submissions = 0; double gpuMs = 0;
        var timings = new List<InferenceTimings>();
        var elapsed = Stopwatch.StartNew();
        for (var i = 0; i < requests; i++)
        {
            var timer = Stopwatch.StartNew();
            var result = await client.InferAsync(request, ct);
            samples[i] = timer.Elapsed.TotalMilliseconds;
            simulation = result.IsSimulation;
            totalUnits += result.TokenIds?.Length ?? result.Tokens.Length; workers.Add(result.WorkerId);
            dispatches += result.GpuDispatches; submissions += result.GpuSubmissions; gpuMs += result.GpuComputeMs;
            if (result.Timings is not null) timings.Add(result.Timings);
        }
        elapsed.Stop(); Array.Sort(samples);
        double Percentile(double p) => samples[Math.Clamp((int)Math.Ceiling(p * requests) - 1, 0, requests - 1)];
        var decodeTokens = timings.Sum(t => t.DecodeTokens);
        var decodeMs = timings.Sum(t => t.DecodeMs);
        var firstTokens = timings.Select(t => t.FirstTokenMs).Order().ToArray();
        return new(simulation ? "simulation-rpc-sequential; NOT LLM tokens/sec" : $"{backend}-llm-rpc-sequential; includes prefill", requests, warmup, totalUnits,
            elapsed.Elapsed.TotalMilliseconds, Percentile(.5), Percentile(.95), requests / elapsed.Elapsed.TotalSeconds,
            simulation ? totalUnits / elapsed.Elapsed.TotalSeconds : 0, workers.Order(StringComparer.Ordinal).ToArray(), simulation,
            simulation ? 0 : totalUnits / elapsed.Elapsed.TotalSeconds, dispatches, gpuMs,
            timings.Count == 0 ? 0 : timings.Average(t => t.PrefillMs),
            firstTokens.Length == 0 ? 0 : firstTokens[(int)Math.Ceiling(firstTokens.Length * .5) - 1],
            decodeTokens, decodeMs, decodeMs == 0 ? 0 : decodeTokens * 1000d / decodeMs, submissions);
    }
}
