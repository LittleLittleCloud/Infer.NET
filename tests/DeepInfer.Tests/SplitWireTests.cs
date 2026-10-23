using System.Net;
using System.Net.Sockets;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;
public sealed class SplitWireTests
{
    [Theory] [InlineData(2)] [InlineData(4)] public async Task AuthenticatesPeersAndTransfersRealTensorBytes(int kvBytes)
    {
        using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(5));var ct=timeout.Token;
        using var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();
        using var client=new TcpClient();var connect=client.ConnectAsync(IPAddress.Loopback,((IPEndPoint)listener.LocalEndpoint).Port,ct);
        using var peer=await listener.AcceptTcpClientAsync(ct);await connect;
        var id=Guid.NewGuid();var token=new string('a',64);
        var server=SplitWire.ConnectAsync(peer.GetStream(),token,true,Guid.Empty,ct,kvBytes);
        var sender=await SplitWire.ConnectAsync(client.GetStream(),token,false,id,ct,kvBytes);var receiver=await server;
        var frame=new SplitWire.Frame(SplitWire.Hidden,20,0,32,2,new byte[2*2048*4]);frame.Data[123]=42;
        await sender.SendAsync(frame,ct);var actual=await receiver.ReceiveAsync(ct);
        Assert.Equal(id,receiver.Request);Assert.Equal(frame.Data,actual.Data);Assert.Equal(32,actual.Position);
    }
    [Theory] [InlineData(false)] [InlineData(true)]
    public async Task RejectsWrongSecretOrKvProfile(bool wrongProfile)
    {
        using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(5));var ct=timeout.Token;
        using var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();
        using var client=new TcpClient();var connect=client.ConnectAsync(IPAddress.Loopback,((IPEndPoint)listener.LocalEndpoint).Port,ct);
        using var peer=await listener.AcceptTcpClientAsync(ct);await connect;
        var server=SplitWire.ConnectAsync(peer.GetStream(),new string('a',64),true,Guid.Empty,ct,wrongProfile?4:2);
        var sender=SplitWire.ConnectAsync(client.GetStream(),new string(wrongProfile?'a':'b',64),false,Guid.NewGuid(),ct,2);
        await Assert.ThrowsAsync<InvalidDataException>(()=>server);peer.Dispose();
        await Assert.ThrowsAnyAsync<IOException>(()=>sender);
    }
    [Theory] [InlineData(0)] [InlineData(8)] [InlineData(24)] [InlineData(52)] [InlineData(100)]
    public async Task RejectsHeaderAndPayloadTampering(int index)
    {
        var key=new byte[32];var id=Guid.NewGuid();using var stream=new MemoryStream();
        var sender=new SplitWire(stream,key,id);await sender.SendAsync(new(SplitWire.Hidden,20,0,0,1,new byte[8192]),default);
        var bytes=stream.ToArray();bytes[index]^=1;using var bad=new MemoryStream(bytes);
        await Assert.ThrowsAsync<InvalidDataException>(()=>new SplitWire(bad,key,id).ReceiveAsync(default));
    }
    [Fact] public async Task RejectsImpossiblePayloadSizeBeforeReadingPayload()
    {
        var id=Guid.NewGuid();using var output=new MemoryStream();
        await new SplitWire(output,new byte[32],id).SendAsync(new(SplitWire.End,20,0,1,0,[]),default);
        var header=output.ToArray()[..92];
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(48,4),512*1024*1024);
        using var bad=new MemoryStream(header);
        await Assert.ThrowsAsync<InvalidDataException>(()=>new SplitWire(bad,new byte[32],id).ReceiveAsync(default));
    }
    [Fact] public async Task RejectsDuplicateFrameAndTruncation()
    {
        var id=Guid.NewGuid();using var output=new MemoryStream();await new SplitWire(output,new byte[32],id).SendAsync(new(SplitWire.End,20,0,1,0,[]),default);
        var bytes=output.ToArray();using var duplicate=new MemoryStream(bytes.Concat(bytes).ToArray());var reader=new SplitWire(duplicate,new byte[32],id);
        await reader.ReceiveAsync(default);await Assert.ThrowsAsync<InvalidDataException>(()=>reader.ReceiveAsync(default));
        using var truncated=new MemoryStream(bytes[..^1]);await Assert.ThrowsAsync<EndOfStreamException>(()=>new SplitWire(truncated,new byte[32],id).ReceiveAsync(default));
    }
    [Fact] public void RejectsWrongStateShapeAndUnsupportedContext()
    {
        Assert.Throws<InvalidDataException>(()=>SplitWire.Validate(new(SplitWire.State,3,0,128,1,new byte[128*512*4])));
        Assert.Throws<InvalidDataException>(()=>SplitWire.Validate(new(SplitWire.Start,20,0,0,262145,new byte[262145*4])));
        SplitWire.Validate(new(SplitWire.Start,20,0,0,8192,new byte[8192*4]));
        SplitWire.Validate(new(SplitWire.State,3,0,4097,1,new byte[4097*512*4]),4);
        Assert.Throws<InvalidDataException>(()=>SplitWire.Validate(new(SplitWire.Hidden,20,0,int.MaxValue,32,new byte[32*8192])));
        SplitWire.Validate(new(SplitWire.State,0,0,128,1,new byte[32*128*128*4]));
        SplitWire.Validate(new(SplitWire.State,3,1,128,1,new byte[128*512*2]));
        SplitWire.Validate(new(SplitWire.State,3,1,128,1,new byte[128*512*4]),4);
        Assert.Throws<InvalidDataException>(()=>SplitWire.Validate(new(SplitWire.State,3,1,128,1,new byte[128*512*4]),2));
    }
}
