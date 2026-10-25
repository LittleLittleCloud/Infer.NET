using System.Diagnostics;
using DeepInfer.Backends.Mtp;
using Xunit;
namespace DeepInfer.Tests;

public class MtpLimitTests
{
    [Fact]
    public void ContextAndOutputBoundsAreEnforcedAtLimitAndLimitPlusOne()
    {
        Assert.Null(MtpLimits.Check(32768 - 4096, 4096));
        Assert.Null(MtpLimits.Check(1, 1));
        Assert.NotNull(MtpLimits.Check(32768 - 4096, 4097));
        Assert.NotNull(MtpLimits.Check(32768 - 4095, 4096));
        Assert.NotNull(MtpLimits.Check(32768, 1));
        Assert.NotNull(MtpLimits.Check(10, 0));
        Assert.Contains("32768", MtpLimits.Check(30000, 4000)!);
        Assert.Contains("4096", MtpLimits.Check(10, 5000)!);
    }

    [Fact]
    public void PreTokenizationOutputCheckRejectsOversizeAndRemainingContext()
    {
        Assert.Null(MtpLimits.CheckOutput(4096, false));
        Assert.NotNull(MtpLimits.CheckOutput(4097, false));
        Assert.NotNull(MtpLimits.CheckOutput(16, true));
    }
}

// Process tests of the real CLI: every case fails inside MtpHostConfig.Parse, before any lock, model, weight or GPU work.
public class MtpHostOptionProcessTests
{
    private static string HostDll()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        var configuration = directory.Parent!.Name; // bin/<configuration>/<tfm>/
        var root = directory; while (root is not null && !Directory.Exists(Path.Combine(root.FullName, "src", "DeepInfer.Host"))) root = root.Parent;
        Assert.NotNull(root);
        var dll = Path.Combine(root!.FullName, "src", "DeepInfer.Host", "bin", configuration, directory.Name, "deepinfer.dll");
        Assert.True(File.Exists(dll), $"Host build output missing: {dll}");
        return dll;
    }

    private static readonly string[] Valid =
        ["--mtp", "1", "--mtp-sidecar", "/nonexistent-sidecar", "--decode-attention", "fused-fp32", "--kv", "fp32", "--metal-scratch-tokens", "128",
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
    [InlineData("--mtp", "3", "--mtp must be off, 1 or 2")]
    [InlineData("--mtp", "0", "--mtp must be off, 1 or 2")]
    [InlineData("--mtp-sidecar", null, "--mtp requires --mtp-sidecar")]
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
        var args = With().Concat(["--mtp-replay", "fast"]).ToArray();
        var (exit, output) = Run(args);
        Assert.Equal(1, exit); Assert.Contains("batched or serial", output);
    }

    [Fact]
    public void SidecarAndReplayWithoutEnablingMtpAreRejectedSoOffStaysExplicit()
    {
        var (exit, output) = Run(["split-serve", "--mtp-sidecar", "/nonexistent-sidecar"]);
        Assert.Equal(1, exit); Assert.Contains("require --mtp 1 or --mtp 2", output);
        (exit, output) = Run(["split-serve", "--mtp", "off", "--mtp-replay", "serial"]);
        Assert.Equal(1, exit); Assert.Contains("require --mtp 1 or --mtp 2", output);
    }
}
