using System.Security.Cryptography;
namespace DeepInfer.Host;

internal static class PinnedSplitModel
{
    // Checksums from the pinned repository revision, not an untrusted supplied manifest.
    private static readonly Dictionary<string, string> Files = new()
    {
        ["config.json"] = "5da49c3a84c5a4a751b720b62b9f8703bdb8a0796b5fc55df3278f8b9d6ce296",
        ["model-00001-of-00008.safetensors"] = "ae1ced7360792820cfc12f2e0c8e1ca4d95c712c122e08ebb474b197dee3edaf",
        ["model-00002-of-00008.safetensors"] = "b8fdc3700f7339dc29d0e8504d15bc61e9beda7dc69eea037084cfa4067f1b26",
        ["model-00003-of-00008.safetensors"] = "8a7851d9152dd39369026ab0ac7c369e4e62ce67ea1e3836862ff0c651176675",
        ["model-00004-of-00008.safetensors"] = "3ad6623bb7ff4ddf39ee3854d473d66543ea8fdffc48f03ae4a1dc866da34b36",
        ["model-00005-of-00008.safetensors"] = "ec9d987a3c499146e6e8edd92afb3943dfd35c908a3de0685346c1ea6901e383",
        ["model-00006-of-00008.safetensors"] = "9a843214e1e72bfbae1d2c61283e25ee5930b5ccdbe15dccc7d72fba3a5812c4",
        ["model-00007-of-00008.safetensors"] = "0a1ed4b82de64955eefb212b2eda57acfa1c2330a4258448e5c7f0f8aaf3649a",
        ["model-00008-of-00008.safetensors"] = "41a1d955de1a9a97de3abebc40b5519b02d542adf266fefbd33b24dc674aef39",
        ["model.safetensors.index.json"] = "3db12edeebeb65cab9a6eeb63cd74be4e0c74139a75f672701290b98230501cf",
        ["tokenizer.json"] = "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
    };
    public static async Task Verify(string directory, CancellationToken ct)
    {
        foreach (var item in Files)
        {
            await using var file = File.OpenRead(Path.Combine(directory, item.Key));
            if (Convert.ToHexStringLower(await SHA256.HashDataAsync(file, ct)) != item.Value)
                throw new InvalidDataException("Pinned model file hash mismatch: " + item.Key);
        }
    }
}
