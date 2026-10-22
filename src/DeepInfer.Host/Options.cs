namespace DeepInfer.Host;

public sealed class Options
{
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    private static readonly HashSet<string> Flags = ["allow-remote", "json", "stay", "trace-all", "prefill"];
    private static readonly HashSet<string> Keys = ["listen", "coordinator", "backend", "id", "capacity", "advertise",
        "token", "endpoint", "prompt", "prompt-file", "max-tokens", "model", "requests", "warmup", "workers", "base-port", "output", "weights", "tokenizer", "reference", "route-trace"];
    public string Command { get; }
    public Options(string[] args)
    {
        Command = args.Length == 0 ? "help" : args[0];
        for (var i = 1; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Expected an option, received '{args[i]}'.");
            var key = args[i][2..];
            if (!Flags.Contains(key) && !Keys.Contains(key)) throw new ArgumentException($"Unknown option --{key}.");
            if (values.ContainsKey(key)) throw new ArgumentException($"Duplicate option --{key}.");
            if (Flags.Contains(key)) values[key] = "true";
            else
            {
                if (++i >= args.Length || args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"--{key} needs a value.");
                values[key] = args[i];
            }
        }
    }
    public string? Token => values.GetValueOrDefault("token") ?? Environment.GetEnvironmentVariable("DEEPINFER_TOKEN");
    public string Get(string key, string fallback) => values.GetValueOrDefault(key) ?? fallback;
    public bool Flag(string key) => values.ContainsKey(key);
    public int Number(string key, int fallback, int min, int max)
    {
        if (!int.TryParse(Get(key, fallback.ToString(System.Globalization.CultureInfo.InvariantCulture)), out var value) || value < min || value > max)
            throw new ArgumentException($"--{key} must be {min}..{max}.");
        return value;
    }
}
