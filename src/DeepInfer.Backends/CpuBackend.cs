using DeepInfer.Core;
using DeepInfer.Backends.Models;
namespace DeepInfer.Backends;

public sealed class CpuBackend(string? weightsPath = null, string? tokenizerPath = null) : IComputeBackend
{
    public string Kind => "cpu";
    public IReadOnlyList<DeviceInfo> DiscoverDevices() => [new("cpu:0", Kind, "CPU reference transformer",
        System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(), null, true, true,
        "llama2c-f32 / hf-qwen2 inference; SIMD GEMV and CPU attention")];
    public ValueTask<IInferenceModel> LoadModelAsync(ModelInfo model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model.Format is not ("llama2c-f32" or "hf-qwen2") || model.IsSimulation) throw new NotSupportedException("CPU model requires llama2c-f32 or hf-qwen2.");
        if (string.IsNullOrWhiteSpace(weightsPath) || string.IsNullOrWhiteSpace(tokenizerPath))
            throw new ArgumentException("Real inference requires --weights and --tokenizer paths.");
        return ValueTask.FromResult<IInferenceModel>(new LlamaModel(model, weightsPath, tokenizerPath, Kind));
    }
}
