using DeepInfer.Backends.Mtp;
using DeepInfer.Core;
namespace DeepInfer.Host;

// Default-off MTP experiment options. Any unsupported combination is rejected up front; there is no fallback.
public sealed record MtpHostConfig(int Width, string SidecarDirectory, MtpReplayMode Replay)
{
    public static MtpHostConfig? Parse(Options o)
    {
        var mode = o.Get("mtp", "off");
        if (mode == "off")
        {
            if (o.Get("mtp-sidecar", "") != "" || o.Get("mtp-replay", "batched") != "batched") throw new ArgumentException("--mtp-sidecar/--mtp-replay require --mtp 1 or --mtp 2.");
            return null;
        }
        if (mode is not ("1" or "2")) throw new ArgumentException("--mtp must be off, 1 or 2.");
        var sidecar = o.Get("mtp-sidecar", "");
        if (sidecar == "") throw new ArgumentException("--mtp requires --mtp-sidecar <converted official sidecar directory>.");
        if (o.Get("decode-attention", "legacy") != "fused-fp32") throw new ArgumentException("--mtp requires --decode-attention fused-fp32.");
        if (o.Get("kv", "fp32") != "fp32") throw new ArgumentException("--mtp requires FP32 KV.");
        if (o.Get("metal-scratch-tokens", "512") != "128") throw new ArgumentException("--mtp requires --metal-scratch-tokens 128.");
        if (o.Get("suffix-chunk-tokens", "128") != "128") throw new ArgumentException("--mtp requires --suffix-chunk-tokens 128.");
        if (o.Number("short-prefill-tokens", 128, 0, Protocol.Qwen36Context) < 32768)
            throw new ArgumentException("--mtp requires --short-prefill-tokens >= 32768 (fully local prefill; prompts above the limit are rejected, split prefill is unsupported).");
        if (o.Get("metal-residency", "on") != "on") throw new ArgumentException("--mtp requires --metal-residency on.");
        var replay = o.Get("mtp-replay", "batched") switch
        {
            "batched" => MtpReplayMode.Batched,
            "serial" => MtpReplayMode.Serial,
            _ => throw new ArgumentException("--mtp-replay must be batched or serial (serial is a diagnostic oracle, never a performance mode).")
        };
        return new(int.Parse(mode), Path.GetFullPath(sidecar), replay);
    }
}
