using DeepInfer.Backends.Dflash2;
using DeepInfer.Core;
namespace DeepInfer.Host;

// Default-off DFlash2 experiment options. Any unsupported combination is rejected up front; there is no fallback.
public sealed record Dflash2HostConfig(int Block, string DraftDirectory, Dflash2ReplayMode Replay)
{
    public const long AllocationBudgetBytes = (long)(38.5 * 1024 * 1024 * 1024);
    public static Dflash2HostConfig? Parse(Options o)
    {
        var mode = o.Get("dflash2", "off");
        if (mode == "off")
        {
            if (o.Get("dflash2-model", "") != "" || o.Get("dflash2-replay", "batched") != "batched") throw new ArgumentException("--dflash2-model/--dflash2-replay require --dflash2 4 or --dflash2 8.");
            return null;
        }
        if (mode is not ("4" or "8")) throw new ArgumentException("--dflash2 must be off, 4 or 8.");
        var model = o.Get("dflash2-model", "");
        if (model == "") throw new ArgumentException("--dflash2 requires --dflash2-model <pinned DFlash2 draft directory>.");
        if (o.Get("decode-attention", "legacy") != "fused-fp32") throw new ArgumentException("--dflash2 requires --decode-attention fused-fp32.");
        if (o.Get("kv", "fp32") != "fp32") throw new ArgumentException("--dflash2 requires FP32 KV.");
        if (o.Get("metal-scratch-tokens", "512") != "128") throw new ArgumentException("--dflash2 requires --metal-scratch-tokens 128.");
        if (o.Get("suffix-chunk-tokens", "128") != "128") throw new ArgumentException("--dflash2 requires --suffix-chunk-tokens 128.");
        if (o.Number("short-prefill-tokens", 128, 0, Protocol.Qwen36Context) < 32768)
            throw new ArgumentException("--dflash2 requires --short-prefill-tokens >= 32768 (fully local prefill; prompts above the limit are rejected, split prefill is unsupported).");
        if (o.Get("metal-residency", "on") != "on") throw new ArgumentException("--dflash2 requires --metal-residency on.");
        var replay = o.Get("dflash2-replay", "batched") switch
        {
            "batched" => Dflash2ReplayMode.Batched,
            "serial" => Dflash2ReplayMode.Serial,
            "scalar" => Dflash2ReplayMode.Scalar,
            "native" when mode == "8" => Dflash2ReplayMode.Native,
            _ => throw new ArgumentException("--dflash2-replay must be batched or serial, scalar, or native (native requires block8; scalar is the diagnostic reference).")
        };
        return new(int.Parse(mode), Path.GetFullPath(model), replay);
    }
}
