namespace DeepInfer.Host;

public sealed class Options
{
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    private static readonly HashSet<string> Flags = ["verify", "allow-remote", "json", "stay", "trace-all", "prefill", "windows-service", "stream", "ignore-eos", "poison-suffix"];
    private static readonly HashSet<string> Keys = ["listen", "coordinator", "backend", "id", "capacity", "advertise",
        "config", "log-file", "decode-attention", "kv-transfer", "cuda-send-overlap", "suffix-chunk-tokens", "metal-scratch-tokens", "cuda-prefill", "cuda-readback", "short-prefill-tokens", "metal-residency", "residency-refresh-ms", "data-listen", "token-file", "tls-cert", "tls-password-file", "tls-peer-sha256", "lock-file", "deadline-seconds", "queue-size", "kv", "token", "endpoint", "prompt", "prompt-file", "max-tokens", "model", "requests", "warmup", "workers", "base-port", "output", "weights", "tokenizer", "reference", "route-trace",
        "speculative-synthetic", "speculative-block-tokens", "token-ids", "output-lengths", "patterns", "blocks", "warm-runs", "formal-runs", "checkpoint-tokens"];
    public string Command { get; }
    public Options(string[] args)
    {
        Command = args.Length == 0 ? "help" : args[0];
        var configIndex = Array.IndexOf(args, "--config");
        if (configIndex >= 0)
        {
            if (configIndex + 1 >= args.Length) throw new ArgumentException("--config needs a path.");
            using var config = System.Text.Json.JsonDocument.Parse(File.ReadAllText(args[configIndex + 1]));
            foreach (var entry in config.RootElement.EnumerateObject())
            {
                if (!Keys.Contains(entry.Name) && !Flags.Contains(entry.Name)) throw new ArgumentException("Unknown configuration key: " + entry.Name);
                if (entry.Name == "config" || entry.Name == "token") throw new ArgumentException("Configuration must use token-file; nested config is not supported.");
                if (Flags.Contains(entry.Name)) { if(entry.Value.GetBoolean()) values.Add(entry.Name, "true"); }
                else values.Add(entry.Name, entry.Value.GetString() ?? throw new ArgumentException("Configuration values must be strings."));
            }
        }
        var supplied = new HashSet<string>(StringComparer.Ordinal);
        for (var i = 1; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Expected an option, received '{args[i]}'.");
            var key = args[i][2..];
            if (!Flags.Contains(key) && !Keys.Contains(key)) throw new ArgumentException($"Unknown option --{key}.");
            if (!supplied.Add(key)) throw new ArgumentException($"Duplicate option --{key}.");
            if (Flags.Contains(key)) values[key] = "true";
            else
            {
                if (++i >= args.Length || args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"--{key} needs a value.");
                values[key] = args[i];
            }
        }
    }
    public string? Token => values.TryGetValue("token-file", out var path) ? File.ReadAllText(path).Trim() : values.GetValueOrDefault("token") ?? Environment.GetEnvironmentVariable("DEEPINFER_TOKEN");
    public string Get(string key, string fallback) => values.GetValueOrDefault(key) ?? fallback;
    public bool Flag(string key) => values.ContainsKey(key);
    public int Number(string key, int fallback, int min, int max)
    {
        if (!int.TryParse(Get(key, fallback.ToString(System.Globalization.CultureInfo.InvariantCulture)), out var value) || value < min || value > max)
            throw new ArgumentException($"--{key} must be {min}..{max}.");
        return value;
    }
}
