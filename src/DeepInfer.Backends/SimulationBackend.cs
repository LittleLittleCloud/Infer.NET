using System.Security.Cryptography;
using System.Text;
using DeepInfer.Core;
namespace DeepInfer.Backends;

// Deliberately not called CPU inference: no neural network, weights, tokenizer or GPU work.
public sealed class SimulationBackend : IComputeBackend
{
    public string Kind => "simulation";
    public IReadOnlyList<DeviceInfo> DiscoverDevices() => [new("simulation:0", Kind,
        "Deterministic synthetic generator", System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
        null, true, true, "SIMULATION ONLY — no model weights or GPU inference")];
    public ValueTask<IInferenceModel> LoadModelAsync(ModelInfo model, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (model.Id != Protocol.SyntheticModel || !model.IsSimulation || model.Format != "synthetic")
            throw new NotSupportedException("Only synthetic-v1 is implemented.");
        return ValueTask.FromResult<IInferenceModel>(new SyntheticModel());
    }

    private sealed class SyntheticModel : IInferenceModel
    {
        public ModelInfo Info => new(Protocol.SyntheticModel, "synthetic", true);
        public ValueTask<ModelOutput> GenerateAsync(InferenceRequest request, CancellationToken cancellationToken)
        {
            var error = RequestValidation.Error(request);
            if (error != null) throw new ArgumentException(error);
            if (request.Model != Info.Id || request.Backend != "simulation")
                throw new NotSupportedException("Synthetic model/backend mismatch.");
            string[] vocabulary = ["distributed", "worker", "rpc", "tensor", "route", "model", "device", "native"];
            var hash = SHA256.HashData(Encoding.UTF8.GetBytes(request.Prompt));
            var output = new string[request.MaxTokens];
            for (var i = 0; i < output.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                output[i] = vocabulary[(hash[i % hash.Length] + i) % vocabulary.Length];
            }
            return ValueTask.FromResult(new ModelOutput(output, true));
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
