using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
namespace DeepInfer.Runtime;

// Version 2 fixes geometry, dtype, little-endian layouts and model revision.
// Frames authenticate both metadata and payload; nonce-derived keys prevent replay.
// Trusted LAN transport: authenticated/integrity-protected, NOT encrypted.
public sealed class SplitWire(Stream stream, byte[] key, Guid request)
{
    public const string Revision = "e06a74e6236a60c8367e1a3214e83d8b61b637b0";
    public const int Start=1, Hidden=2, State=3, End=4, Trace=5;
    private int sent, received;
    public Guid Request => request;
    public sealed record Frame(int Kind, int Layer, int Tensor, int Position, int Count, byte[] Data);
    public static async Task<SplitWire> ConnectAsync(Stream stream, string token, bool server, Guid request, CancellationToken ct)
    {
        if (token.Length < 32) throw new ArgumentException("Split transport requires a token of at least 32 characters.");
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Split protocol uses little-endian tensors.");
        var secret=Encoding.UTF8.GetBytes(token); var a=new byte[32]; var b=new byte[32]; var id=new byte[16];
        if(server){RandomNumberGenerator.Fill(a);await stream.WriteAsync(a,ct);await stream.ReadExactlyAsync(b,ct);await stream.ReadExactlyAsync(id,ct);request=new Guid(id);}
        else {await stream.ReadExactlyAsync(a,ct);RandomNumberGenerator.Fill(b);id=request.ToByteArray();await stream.WriteAsync(b,ct);await stream.WriteAsync(id,ct);}
        var challenge=a.Concat(b).Concat(id).Concat(Encoding.ASCII.GetBytes("Infer.NET/split/v2/"+Revision)).ToArray();
        var key=HMACSHA256.HashData(secret,challenge); var clientTag=HMACSHA256.HashData(key,"client"u8); var serverTag=HMACSHA256.HashData(key,"server"u8);
        var actual=new byte[32];
        if(server){await stream.ReadExactlyAsync(actual,ct);if(!CryptographicOperations.FixedTimeEquals(actual,clientTag))throw new InvalidDataException("Split authentication failed.");await stream.WriteAsync(serverTag,ct);}
        else {await stream.WriteAsync(clientTag,ct);await stream.ReadExactlyAsync(actual,ct);if(!CryptographicOperations.FixedTimeEquals(actual,serverTag))throw new InvalidDataException("Split authentication failed.");}
        return new(stream,key,request);
    }
    public static void Validate(Frame f)
    {
        var valid=f.Kind switch
        {
            Start => f.Layer==20 && f.Tensor is 0 or 1 && f.Position==0 && f.Count is >=1 and <=2048 && f.Data.Length==f.Count*4,
            Hidden => f.Layer==20 && f.Tensor==0 && f.Position>=0 && f.Count is >=1 and <=32 && f.Position+f.Count<=2048 && f.Data.Length==f.Count*2048*4,
            Trace => f.Layer==0 && f.Tensor==0 && f.Position>=0 && f.Count is >=1 and <=32 && f.Position+f.Count<=2048 && f.Data.Length==f.Count*20*2048*4,
            State => f.Layer is >=0 and <20 && f.Tensor is 0 or 1 && f.Position is >=1 and <=2048 && f.Count==1 && f.Data.Length==((f.Layer+1)%4==0?f.Position*512*2:f.Tensor==0?32*128*128*4:3*8192*4),
            End => f.Layer==20 && f.Tensor==0 && f.Position is >=1 and <=2048 && f.Count==0 && f.Data.Length==0,
            _ => false
        };
        if(!valid)throw new InvalidDataException("Invalid split tensor descriptor/shape/layout.");
    }
    public async Task SendAsync(Frame f,CancellationToken ct)
    {
        Validate(f); var header=new byte[92];
        Write(header,0,0x53504944);Write(header,4,2);request.TryWriteBytes(header.AsSpan(8,16));
        int[] fields=[sent++,f.Kind,f.Layer,f.Tensor,f.Position,f.Count,f.Data.Length];for(var i=0;i<fields.Length;i++)Write(header,24+i*4,fields[i]);
        Encoding.ASCII.GetBytes(Revision,header.AsSpan(52));
        using var mac=IncrementalHash.CreateHMAC(HashAlgorithmName.SHA256,key);mac.AppendData(header);mac.AppendData(f.Data);var tag=mac.GetHashAndReset();
        await stream.WriteAsync(header,ct);await stream.WriteAsync(f.Data,ct);await stream.WriteAsync(tag,ct);
    }
    public async Task<Frame> ReceiveAsync(CancellationToken ct)
    {
        var h=new byte[92];await stream.ReadExactlyAsync(h,ct);
        if(Read(h,0)!=0x53504944||Read(h,4)!=2||new Guid(h.AsSpan(8,16))!=request||Read(h,24)!=received++||Encoding.ASCII.GetString(h,52,40)!=Revision)throw new InvalidDataException("Split protocol/request/sequence/revision mismatch.");
        var size=Read(h,48);if(size is <0 or >32*20*2048*4)throw new InvalidDataException("Split frame exceeds bounded allocation.");
        var data=new byte[size];var tag=new byte[32];await stream.ReadExactlyAsync(data,ct);await stream.ReadExactlyAsync(tag,ct);
        using var mac=IncrementalHash.CreateHMAC(HashAlgorithmName.SHA256,key);mac.AppendData(h);mac.AppendData(data);
        if(!CryptographicOperations.FixedTimeEquals(mac.GetHashAndReset(),tag))throw new InvalidDataException("Split frame integrity check failed.");
        var f=new Frame(Read(h,28),Read(h,32),Read(h,36),Read(h,40),Read(h,44),data);Validate(f);return f;
    }
    private static void Write(byte[] b,int at,int value)=>BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(at,4),value);
    private static int Read(byte[] b,int at)=>BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(at,4));
}
