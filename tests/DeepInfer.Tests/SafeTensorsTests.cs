using System.Text;
using System.Text.Json;
using DeepInfer.Backends.Compute;
using DeepInfer.Backends.Models;
using Xunit;
namespace DeepInfer.Tests;

public sealed class SafeTensorsTests
{
    [Fact] public void PackedUnsignedWeightsAndUnusedVisionShapesCanBeReadWithoutFloatConversion()
    {
        byte[] bytes = [1, 2, 3, 4, 255, 254, 253, 252];
        using var fixture = new Fixture("U32", [1, 1, 1, 1, 2], [0, 8], bytes);
        using var file = new SafeTensors(fixture.Path);
        var tensor = file.Get("test", 1, 1, 1, 1, 2);
        var result = new byte[8]; file.ReadInto(tensor, result);
        Assert.Equal(bytes, result);
        Assert.Throws<InvalidDataException>(() => file.ReadInto(tensor, new byte[7]));
        Assert.Throws<InvalidDataException>(() => file.ReadFloat(tensor));
    }
    [Fact] public void BFloat16TensorValuesAndShapeAreValidated()
    {
        using var fixture = new Fixture("BF16", [2,2], [0,8], [0,0x3f,0,0x40,0x80,0x3f,0x80,0xbf]);
        using var file = new SafeTensors(fixture.Path);
        var tensor = file.Get("test",2,2);
        Assert.Equal(new[] { .5f, 2f, 1f, -1f },file.ReadFloat(tensor));
        Assert.Equal(new ushort[] { 0x3f00,0x4000,0x3f80,0xbf80 },file.ReadBFloat16(tensor));
        Assert.Throws<InvalidDataException>(() => file.Get("test",4));
        Assert.Throws<InvalidDataException>(() => file.Get("missing",2,2));
    }
    [Theory]
    [InlineData(0,10)] [InlineData(-1,7)] [InlineData(4,12)]
    public void InvalidOffsetsAreRejected(long start,long end)
    {
        using var fixture = new Fixture("BF16",[2,2],[start,end],new byte[8]);
        Assert.Throws<InvalidDataException>(() => new SafeTensors(fixture.Path));
    }
    [Fact] public void UnsupportedDtypeAndOverlappingTensorsAreRejected()
    {
        using var unsupported = new Fixture("F16",[2,2],[0,8],new byte[8]);
        Assert.Throws<NotSupportedException>(() => new SafeTensors(unsupported.Path));
        using var overlap = new Fixture("BF16",[2,2],[0,8],new byte[8],true);
        Assert.Throws<InvalidDataException>(() => new SafeTensors(overlap.Path));
    }
    [Fact] public void TruncatedHeaderAndNonFiniteFloatValuesAreRejected()
    {
        using var fixture = new Fixture("F32",[1],[0,4],BitConverter.GetBytes(float.NaN));
        using (var file = new SafeTensors(fixture.Path)) Assert.Throws<InvalidDataException>(() => file.ReadFloat(file.Get("test",1)));
        File.WriteAllBytes(fixture.Path,BitConverter.GetBytes(ulong.MaxValue));
        Assert.Throws<InvalidDataException>(() => new SafeTensors(fixture.Path));
    }
    [Fact] public void CpuBFloat16GemvUsesOriginalWeightValues()
    {
        using var device = new CpuMatVecDevice(); using var matrix = device.UploadBFloat16([0x3f80,0x4000,0xbf80,0x3f00],2,2);
        var output = new float[2];matrix.Multiply(new float[] {3,4},output);
        Assert.Equal(new float[] {11,-1},output);
    }
    private sealed class Fixture : IDisposable
    {
        public string Path {get;} = System.IO.Path.Combine(System.IO.Path.GetTempPath(),"deepinfer-st-"+Guid.NewGuid().ToString("N"));
        public Fixture(string dtype,int[] shape,long[] offsets,byte[] values,bool duplicate=false)
        {
            var entry = new {dtype,shape,data_offsets=offsets};
            var header = new Dictionary<string,object> {{"test",entry}};
            if(duplicate)header["overlap"]=entry;
            var json=Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
            using var writer=new BinaryWriter(File.Create(Path));writer.Write((ulong)json.Length);writer.Write(json);writer.Write(values);
        }
        public void Dispose()=>File.Delete(Path);
    }
}
