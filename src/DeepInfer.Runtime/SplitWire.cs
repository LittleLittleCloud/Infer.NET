using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
namespace DeepInfer.Runtime;

// Version 5 permits up to 2048 tokens per hidden/KV frame; older peers fail authentication.
// Frames authenticate both metadata and payload; nonce-derived keys prevent replay.
// Trusted LAN transport: authenticated/integrity-protected, NOT encrypted.
public sealed class SplitWire(Stream stream, byte[] key, Guid request, int kvBytes = 2)
{
    private const int WireVersion = 5;
    public const string Revision = "e06a74e6236a60c8367e1a3214e83d8b61b637b0";
    public const int Start=1, Hidden=2, State=3, End=4, Trace=5, KvRange=6;
    private int sent, received;
    public Guid Request => request;
    public sealed record Frame(int Kind, int Layer, int Tensor, int Position, int Count, byte[] Data);
    public static async Task<SplitWire> ConnectAsync(Stream stream, string token, bool server, Guid request, CancellationToken ct, int kvBytes = 2)
    {
        if (kvBytes is not (2 or 4)) throw new ArgumentException("Unsupported KV profile.");
        if (token.Length < 32) throw new ArgumentException("Split transport requires a token of at least 32 characters.");
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Split protocol uses little-endian tensors.");
        var secret=Encoding.UTF8.GetBytes(token); var a=new byte[32]; var b=new byte[32]; var id=new byte[16];
        if(server){RandomNumberGenerator.Fill(a);await stream.WriteAsync(a,ct);await stream.ReadExactlyAsync(b,ct);await stream.ReadExactlyAsync(id,ct);request=new Guid(id);}
        else {await stream.ReadExactlyAsync(a,ct);RandomNumberGenerator.Fill(b);id=request.ToByteArray();await stream.WriteAsync(b,ct);await stream.WriteAsync(id,ct);}
        var challenge=a.Concat(b).Concat(id).Concat(Encoding.ASCII.GetBytes("Infer.NET/split/v5/kv"+kvBytes+"/"+Revision)).ToArray();
        var key=HMACSHA256.HashData(secret,challenge); var clientTag=HMACSHA256.HashData(key,"client"u8); var serverTag=HMACSHA256.HashData(key,"server"u8);
        var actual=new byte[32];
        if(server){await stream.ReadExactlyAsync(actual,ct);if(!CryptographicOperations.FixedTimeEquals(actual,clientTag))throw new InvalidDataException("Split authentication failed.");await stream.WriteAsync(serverTag,ct);}
        else {await stream.WriteAsync(clientTag,ct);await stream.ReadExactlyAsync(actual,ct);if(!CryptographicOperations.FixedTimeEquals(actual,serverTag))throw new InvalidDataException("Split authentication failed.");}
        return new(stream,key,request,kvBytes);
    }
    public static void Validate(Frame f, int kvBytes = 2)
        => ValidateDescriptor(f.Kind, f.Layer, f.Tensor, f.Position, f.Count, f.Data.Length, kvBytes);
    private static void ValidateDescriptor(int kind, int layer, int tensor, int position, int count, int size, int kvBytes)
    {
        if(kvBytes is not (2 or 4)) throw new InvalidDataException("Unsupported KV profile.");
        var valid=kind switch
        {
            Start => layer==20 && tensor is >=0 and <=3 && position==0 && count is >=1 and <=DeepInfer.Core.Protocol.Qwen36Context && size==count*4,
            Hidden => layer==20 && tensor==0 && position>=0 && count is >=1 and <=2048 && position<=DeepInfer.Core.Protocol.Qwen36Context-count && size==count*2048*4,
            Trace => layer==0 && tensor==0 && position>=0 && count is >=1 and <=32 && position<=DeepInfer.Core.Protocol.Qwen36Context-count && size==count*20*2048*4,
            KvRange => layer is >=0 and <20 && (layer+1)%4==0 && tensor is 0 or 1 && position>=0 && count is >=1 and <=2048 && position<=DeepInfer.Core.Protocol.Qwen36Context-count && size==count*512*kvBytes,
            State => layer is >=0 and <20 && tensor is 0 or 1 && position is >=1 and <=DeepInfer.Core.Protocol.Qwen36Context && count==1 && size==((layer+1)%4==0?position*512*kvBytes:tensor==0?32*128*128*4:3*8192*4),
            End => layer==20 && tensor==0 && position is >=1 and <=DeepInfer.Core.Protocol.Qwen36Context && count==0 && size is 0 or PrefixAgentTiming.PayloadBytes,
            _ => false
        };
        if(!valid)throw new InvalidDataException("Invalid split tensor descriptor/shape/layout.");
    }
    public async Task SendAsync(Frame f,CancellationToken ct)
    {
        Validate(f,kvBytes); var header=new byte[92];
        Write(header,0,0x53504944);Write(header,4,WireVersion);request.TryWriteBytes(header.AsSpan(8,16));
        int[] fields=[sent++,f.Kind,f.Layer,f.Tensor,f.Position,f.Count,f.Data.Length];for(var i=0;i<fields.Length;i++)Write(header,24+i*4,fields[i]);
        Encoding.ASCII.GetBytes(Revision,header.AsSpan(52));
        using var mac=IncrementalHash.CreateHMAC(HashAlgorithmName.SHA256,key);mac.AppendData(header);mac.AppendData(f.Data);var tag=mac.GetHashAndReset();
        await stream.WriteAsync(header,ct);
        // End has no payload. Do not emit an empty TLS application-data record:
        // Schannel and Apple TLS do not consistently interoperate on empty writes.
        if (f.Data.Length != 0) await stream.WriteAsync(f.Data,ct);
        await stream.WriteAsync(tag,ct);
    }
    public async Task<Frame> ReceiveAsync(CancellationToken ct)
    {
        var h=new byte[92];await stream.ReadExactlyAsync(h,ct);
        if(Read(h,0)!=0x53504944||Read(h,4)!=WireVersion||new Guid(h.AsSpan(8,16))!=request||Read(h,24)!=received++||Encoding.ASCII.GetString(h,52,40)!=Revision)throw new InvalidDataException("Split protocol/request/sequence/revision mismatch.");
        var size=Read(h,48);if(size is <0 or >DeepInfer.Core.Protocol.Qwen36Context*512*4)throw new InvalidDataException("Split frame exceeds bounded allocation.");
        ValidateDescriptor(Read(h,28),Read(h,32),Read(h,36),Read(h,40),Read(h,44),size,kvBytes);
        var data=new byte[size];var tag=new byte[32];await stream.ReadExactlyAsync(data,ct);await stream.ReadExactlyAsync(tag,ct);
        using var mac=IncrementalHash.CreateHMAC(HashAlgorithmName.SHA256,key);mac.AppendData(h);mac.AppendData(data);
        if(!CryptographicOperations.FixedTimeEquals(mac.GetHashAndReset(),tag))throw new InvalidDataException("Split frame integrity check failed.");
        var f=new Frame(Read(h,28),Read(h,32),Read(h,36),Read(h,40),Read(h,44),data);Validate(f,kvBytes);return f;
    }
    private static void Write(byte[] b,int at,int value)=>BinaryPrimitives.WriteInt32LittleEndian(b.AsSpan(at,4),value);
    private static int Read(byte[] b,int at)=>BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(at,4));
}

// Optional timings on the agent's final End frame. Old agents send an empty End.
// These are durations on the agent's own monotonic clock, never timestamps to
// subtract from Mac timestamps. CUDA production and socket writes can overlap.
public sealed record PrefixAgentTiming(double ComputeMs, double StateExportMs, double SendMs,
    double ElapsedMs, long SentBytes)
{
    public const int PayloadBytes = 48;
    private const int Magic = 0x314D5450; // PTM1
    public byte[] ToPayload()
    {
        var data = new byte[PayloadBytes];
        BinaryPrimitives.WriteInt32LittleEndian(data, Magic);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(4), 1);
        WriteDouble(data, 8, ComputeMs); WriteDouble(data, 16, StateExportMs);
        WriteDouble(data, 24, SendMs); WriteDouble(data, 32, ElapsedMs);
        BinaryPrimitives.WriteInt64LittleEndian(data.AsSpan(40), SentBytes);
        return data;
    }
    public static PrefixAgentTiming? FromPayload(ReadOnlySpan<byte> data)
    {
        if (data.IsEmpty) return null;
        if (data.Length != PayloadBytes || BinaryPrimitives.ReadInt32LittleEndian(data) != Magic ||
            BinaryPrimitives.ReadInt32LittleEndian(data[4..]) != 1)
            throw new InvalidDataException("Invalid prefix agent timing payload.");
        var result = new PrefixAgentTiming(ReadDouble(data, 8), ReadDouble(data, 16),
            ReadDouble(data, 24), ReadDouble(data, 32), BinaryPrimitives.ReadInt64LittleEndian(data[40..]));
        if (!Finite(result.ComputeMs) || !Finite(result.StateExportMs) || !Finite(result.SendMs) ||
            !Finite(result.ElapsedMs) || result.SentBytes < 0)
            throw new InvalidDataException("Invalid prefix agent timing values.");
        return result;
    }
    private static bool Finite(double value) => double.IsFinite(value) && value >= 0;
    private static void WriteDouble(Span<byte> data, int offset, double value) =>
        BinaryPrimitives.WriteInt64LittleEndian(data[offset..], BitConverter.DoubleToInt64Bits(value));
    private static double ReadDouble(ReadOnlySpan<byte> data, int offset) =>
        BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(data[offset..]));
}
