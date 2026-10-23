using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using DeepInfer.Backends.Split;
using DeepInfer.Runtime;
namespace DeepInfer.Host;

public static class SplitInference
{
    // Deterministic export of the pinned checkpoint; recompute on both hosts.
    private const string PrefixHash="ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77";
    public static async Task Worker(Options options,CancellationToken ct)
    {
        var token=RequireToken(options);var endpoint=new Uri(options.Get("endpoint","tcp://127.0.0.1:5210"));
        var bundle=options.Get("weights","artifacts/split/prefix.bin");
        Console.Error.WriteLine("Verifying pinned prefix bundle SHA256...");
        await using(var f=File.OpenRead(bundle))
            if(!Convert.ToHexStringLower(await SHA256.HashDataAsync(f,ct)).Equals(PrefixHash,StringComparison.Ordinal))throw new InvalidDataException("Pinned prefix bundle hash mismatch.");
        Console.Error.WriteLine("Loading resident CUDA prefix...");
        using var gpu=new CudaPrefixSession(bundle,2048);
        using var client=new TcpClient{NoDelay=true};await client.ConnectAsync(endpoint.Host,endpoint.Port,ct);
        var wire=await SplitWire.ConnectAsync(client.GetStream(),token,false,Guid.NewGuid(),ct);
        var start=await wire.ReceiveAsync(ct);if(start.Kind!=SplitWire.Start)throw new InvalidDataException("Expected split request.");
        var tokens=Ints(start.Data);var verify=start.Tensor==1;var timer=Stopwatch.StartNew();
        for(var offset=0;offset<tokens.Length;offset+=32)
        {
            ct.ThrowIfCancellationRequested();var count=Math.Min(32,tokens.Length-offset);
            var trace=verify?new float[count*20*2048]:null;
            var hidden=gpu.Run(tokens.AsSpan(offset,count).ToArray(),offset,trace);
            await wire.SendAsync(new(SplitWire.Hidden,20,0,offset,count,Bytes(hidden)),ct);
            if(trace is not null)await wire.SendAsync(new(SplitWire.Trace,0,0,offset,count,Bytes(trace)),ct);
            Console.Error.WriteLine($"PREFIX_PROGRESS tokens={offset+count}/{tokens.Length} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
        }
        for(var layer=0;layer<20;layer++)for(var tensor=0;tensor<2;tensor++)
            await wire.SendAsync(new(SplitWire.State,layer,tensor,tokens.Length,1,gpu.State(layer,tensor,tokens.Length)),ct);
        await wire.SendAsync(new(SplitWire.End,20,0,tokens.Length,0,[]),ct);
        Console.Error.WriteLine($"PREFIX_FINISHED elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
    }
    public static async Task Coordinator(Options options,CancellationToken ct)
    {
        var token=RequireToken(options);var uri=new Uri(options.Get("listen","tcp://127.0.0.1:5210"));var ip=IPAddress.Parse(uri.Host);
        if(!IPAddress.IsLoopback(ip)&&!options.Flag("allow-remote"))throw new ArgumentException("Remote split listener requires --allow-remote.");
        var verify=options.Flag("verify");var max=options.Number("max-tokens",8,1,256);
        var output=Path.GetFullPath(options.Get("output","artifacts/split/run-"+DateTime.UtcNow.ToString("yyyyMMddTHHmmss")));
        Directory.CreateDirectory(output);
        Console.Error.WriteLine("Loading full resident Metal model...");
        using var gpu=new MetalSplitSession(options.Get("weights","models/qwen3.6-35b-a3b-8bit-mlx"));
        var prompt=options.Get("prompt-file","") is {Length:>0} file?await File.ReadAllTextAsync(file,ct):options.Get("prompt","用一句话介绍你自己。");
        var tokens=gpu.Tokenize(prompt,ct);
        if(tokens.Length>2048||tokens.Length+max>2048)throw new ArgumentException("Initial split executor supports prompt + output <=2048; no long-context performance claim.");
        if(verify&&tokens.Length>128)throw new ArgumentException("Layer-by-layer verification is bounded to 128 prompt tokens; use --verify only for short correctness cases.");
        var expected=new List<int>();var checks=new List<SplitCheck>();float[]? referenceTrace=null;
        var logits=new float[248320];
        if(verify)
        {
            referenceTrace=new float[tokens.Length*20*2048];gpu.Reset(tokens.Length+max);
            var next=gpu.Reference(tokens,ct,logits,(p,l,x)=>{if(l<20)x.CopyTo(referenceTrace,(p*20+l)*2048);});
            await File.WriteAllBytesAsync(Path.Combine(output,"reference-logits-0.f32"),Bytes(logits),ct);
            for(var l=0;l<20;l++)for(var t=0;t<2;t++)
            {var data=new byte[CudaPrefixSession.StateBytes(l,t,tokens.Length)];gpu.State(l,t,tokens.Length,data,false);await File.WriteAllBytesAsync(Path.Combine(output,$"reference-state-{l}-{t}.bin"),data,ct);}
            for(var i=0;i<max;i++)
            {expected.Add(next);if(i+1<max){next=gpu.Forward(next,tokens.Length+i,logits);await File.WriteAllBytesAsync(Path.Combine(output,$"reference-logits-{i+1}.f32"),Bytes(logits),ct);}}
            await File.WriteAllBytesAsync(Path.Combine(output,"reference-prefix-trace.f32"),Bytes(referenceTrace),ct);
            Console.Error.WriteLine($"REFERENCE_READY promptTokens={tokens.Length} greedy={string.Join(',',expected)}");
        }
        gpu.Reset(tokens.Length+max);
        using var listener=new TcpListener(ip,uri.Port);listener.Start(1);
        Console.Error.WriteLine($"SPLIT_LISTENING promptTokens={tokens.Length}");
        using var client=await listener.AcceptTcpClientAsync(ct);client.NoDelay=true;listener.Stop();
        var wire=await SplitWire.ConnectAsync(client.GetStream(),token,true,Guid.Empty,ct);
        var timer=Stopwatch.StartNew();await wire.SendAsync(new(SplitWire.Start,20,verify?1:0,0,tokens.Length,Bytes(tokens)),ct);
        var nextToken=-1;
        for(var offset=0;offset<tokens.Length;offset+=32)
        {
            var count=Math.Min(32,tokens.Length-offset);var f=await wire.ReceiveAsync(ct);
            if(f.Kind!=SplitWire.Hidden||f.Position!=offset||f.Count!=count)throw new InvalidDataException("Noncontiguous hidden stream.");
            var hidden=Floats(f.Data);if(hidden.Any(x=>!float.IsFinite(x)))throw new InvalidDataException("Nonfinite hidden tensor.");
            nextToken=gpu.Suffix(tokens.AsSpan(offset,count).ToArray(),hidden,offset+count==tokens.Length,verify?logits:null);
            if(verify)
            {
                var trace=await wire.ReceiveAsync(ct);if(trace.Kind!=SplitWire.Trace||trace.Position!=offset||trace.Count!=count)throw new InvalidDataException("Incorrect trace identity.");
                await File.WriteAllBytesAsync(Path.Combine(output,$"cuda-trace-{offset}.f32"),trace.Data,ct);
                var actual=Floats(trace.Data);
                for(var p=0;p<count;p++)for(var l=0;l<20;l++)checks.Add(Compare($"hidden/p{offset+p}/l{l}",referenceTrace!.AsSpan(((offset+p)*20+l)*2048,2048),actual.AsSpan((p*20+l)*2048,2048)));
            }
            Console.Error.WriteLine($"SUFFIX_PROGRESS tokens={offset+count}/{tokens.Length} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
        }
        var hiddenReadyMs=timer.Elapsed.TotalMilliseconds;
        for(var l=0;l<20;l++)for(var t=0;t<2;t++)
        {
            var f=await wire.ReceiveAsync(ct);if(f.Kind!=SplitWire.State||f.Layer!=l||f.Tensor!=t||f.Position!=tokens.Length)throw new InvalidDataException("Missing, duplicate or stale prefix state.");
            if(verify)
            {
                var reference=await File.ReadAllBytesAsync(Path.Combine(output,$"reference-state-{l}-{t}.bin"),ct);
                await File.WriteAllBytesAsync(Path.Combine(output,$"cuda-state-{l}-{t}.bin"),f.Data,ct);
                checks.Add(Compare($"state/l{l}/t{t}",StateFloats(reference,l),StateFloats(f.Data,l)));
            }
            gpu.State(l,t,tokens.Length,f.Data,true);
        }
        var end=await wire.ReceiveAsync(ct);if(end.Kind!=SplitWire.End||end.Position!=tokens.Length)throw new InvalidDataException("Incomplete prefix handoff.");
        var readyMs=timer.Elapsed.TotalMilliseconds;var ids=new List<int>();
        for(var i=0;i<max;i++)
        {
            ct.ThrowIfCancellationRequested();ids.Add(nextToken);
            if(verify){await File.WriteAllBytesAsync(Path.Combine(output,$"split-logits-{i}.f32"),Bytes(logits),ct);checks.Add(Compare($"logits/{i}",Floats(await File.ReadAllBytesAsync(Path.Combine(output,$"reference-logits-{i}.f32"),ct)),logits));}
            if(i+1<max)nextToken=gpu.Forward(nextToken,tokens.Length+i,verify?logits:null);
        }
        var elapsed=timer.Elapsed.TotalMilliseconds;
        var result=new SplitResult(wire.Request.ToString(),SplitWire.Revision,tokens.Length,ids.ToArray(),gpu.Decode(ids.ToArray()),expected.ToArray(),
            verify&&ids.SequenceEqual(expected)&&checks.All(x=>x.Pass),verify,hiddenReadyMs,readyMs,elapsed,checks.ToArray(),"CUDA serial-token affine8 prefix; Metal <=32-token suffix; BF16 GEMM operands and KV; FP32 accumulation/recurrent state; authenticated direct LAN; no prefix hit");
        var json=JsonSerializer.Serialize(result,SplitJson.Default.SplitResult);await File.WriteAllTextAsync(Path.Combine(output,"result.json"),json+Environment.NewLine,ct);Console.WriteLine(json);
        if(verify&&!result.Verified)throw new InvalidOperationException("Split correctness gate failed; see result.json. Do not use these timings as accepted performance.");
    }
    private static string RequireToken(Options o)=>o.Token is {Length:>=32} t?t:throw new ArgumentException("Set DEEPINFER_TOKEN to a random secret of at least 32 characters.");
    private static byte[] Bytes(float[] x)=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private static byte[] Bytes(int[] x)=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private static int[] Ints(byte[] x)=>MemoryMarshal.Cast<byte,int>(x).ToArray();
    private static float[] Floats(byte[] x)=>MemoryMarshal.Cast<byte,float>(x).ToArray();
    private static float[] StateFloats(byte[] data,int layer)=>(layer+1)%4!=0?Floats(data):MemoryMarshal.Cast<byte,ushort>(data).ToArray().Select(x=>BitConverter.Int32BitsToSingle(x<<16)).ToArray();
    private static SplitCheck Compare(string name,ReadOnlySpan<float> expected,ReadOnlySpan<float> actual)
    {
        double error=0,norm=0,max=0;for(var i=0;i<expected.Length;i++){var d=(double)expected[i]-actual[i];error+=d*d;norm+=(double)expected[i]*expected[i];max=Math.Max(max,Math.Abs(d));}
        var relative=Math.Sqrt(error/Math.Max(norm,1e-30));return new(name,relative,max,double.IsFinite(relative)&&relative<1e-4&&max<.01);
    }
}
public sealed record SplitCheck(string Tensor,double RelativeL2,double MaxAbs,bool Pass);
public sealed record SplitResult(string Request,string Revision,int PromptTokens,int[] Tokens,string Text,int[] ReferenceTokens,bool Verified,bool VerificationEnabled,double HiddenReadyMs,double DecodeReadyMs,double ElapsedMs,SplitCheck[] Checks,string Execution);
[JsonSerializable(typeof(SplitResult))]
[JsonSourceGenerationOptions(WriteIndented=true)]
internal partial class SplitJson:JsonSerializerContext;
