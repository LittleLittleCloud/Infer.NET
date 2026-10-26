using System.Diagnostics;
using Xunit;
namespace DeepInfer.Tests;

// Process tests of the real CLI: every case fails inside Dflash2HostConfig.Parse, before any lock, model, weight or GPU work.
public class Dflash2HostOptionProcessTests
{
    private static string HostDll()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        var configuration = directory.Parent!.Name;
        var root = directory; while (root is not null && !Directory.Exists(Path.Combine(root.FullName, "src", "DeepInfer.Host"))) root = root.Parent;
        Assert.NotNull(root);
        var dll = Path.Combine(root!.FullName, "src", "DeepInfer.Host", "bin", configuration, directory.Name, "deepinfer.dll");
        Assert.True(File.Exists(dll), $"Host build output missing: {dll}");
        return dll;
    }

    private static readonly string[] Valid =
        ["--dflash2", "8", "--dflash2-model", "/nonexistent-draft", "--decode-attention", "fused-fp32", "--kv", "fp32", "--metal-scratch-tokens", "128",
         "--suffix-chunk-tokens", "128", "--short-prefill-tokens", "32768", "--metal-residency", "on"];

    private static string[] With(params (string Key, string? Value)[] overrides)
    {
        var values = new Dictionary<string, string?>();
        for (var i = 0; i < Valid.Length; i += 2) values[Valid[i]] = Valid[i + 1];
        foreach (var (key, value) in overrides) values[key] = value;
        var args = new List<string> { "split-serve" };
        foreach (var (key, value) in values) { if (value is null) continue; args.Add(key); args.Add(value); }
        return [.. args];
    }

    private static (int Exit, string Output) Run(string[] args)
    {
        var start = new ProcessStartInfo("dotnet") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        start.ArgumentList.Add(HostDll()); foreach (var a in args) start.ArgumentList.Add(a);
        using var process = Process.Start(start)!;
        var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
        if (!process.WaitForExit(120_000)) { process.Kill(true); Assert.Fail("Host option validation did not exit."); }
        return (process.ExitCode, stdout.Result + stderr.Result);
    }

    [Theory]
    [InlineData("--dflash2", "3", "--dflash2 must be off, 4 or 8")]
    [InlineData("--dflash2", "16", "--dflash2 must be off, 4 or 8")]
    [InlineData("--dflash2-model", null, "--dflash2 requires --dflash2-model")]
    [InlineData("--decode-attention", "legacy", "fused-fp32")]
    [InlineData("--kv", "bf16", "FP32 KV")]
    [InlineData("--metal-scratch-tokens", "512", "scratch-tokens 128")]
    [InlineData("--suffix-chunk-tokens", "32", "suffix-chunk-tokens 128")]
    [InlineData("--short-prefill-tokens", "128", "short-prefill-tokens >= 32768")]
    [InlineData("--metal-residency", "off", "residency on")]
    public void UnsupportedEnabledConfigurationsAreRejectedBeforeAnyWork(string key, string? value, string message)
    {
        var (exit, output) = Run(With((key, value)));
        Assert.Equal(1, exit);
        Assert.Contains(message, output);
    }

    [Fact]
    public void ReplayModeMustBeBatchedOrSerial()
    {
        var (exit, output) = Run([.. With(), "--dflash2-replay", "fast"]);
        Assert.Equal(1, exit); Assert.Contains("batched or serial", output);
    }

    [Fact]
    public void ModelAndReplayWithoutEnablingDflash2AreRejectedSoOffStaysExplicit()
    {
        var (exit, output) = Run(["split-serve", "--dflash2-model", "/nonexistent-draft"]);
        Assert.Equal(1, exit); Assert.Contains("require --dflash2 4 or --dflash2 8", output);
        (exit, output) = Run(["split-serve", "--dflash2", "off", "--dflash2-replay", "serial"]);
        Assert.Equal(1, exit); Assert.Contains("require --dflash2 4 or --dflash2 8", output);
    }
}
