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
    internal const string PrefixHash="ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77";
    public static async Task Worker(Options options,CancellationToken ct)
    {
        var token=RequireToken(options);var kvBytes=KvBytes(options);var endpoint=new Uri(options.Get("endpoint","tcp://127.0.0.1:5210"));
        var bundle=options.Get("weights","artifacts/split/prefix.bin");
        Console.Error.WriteLine("Verifying pinned prefix bundle SHA256...");
        await using(var f=File.OpenRead(bundle))
            if(!Convert.ToHexStringLower(await SHA256.HashDataAsync(f,ct)).Equals(PrefixHash,StringComparison.Ordinal))throw new InvalidDataException("Pinned prefix bundle hash mismatch.");
        Console.Error.WriteLine("Loading resident CUDA prefix...");
        if(options.Get("cuda-prefill", "batch-fp32") is not ("serial" or "batch-fp32" or "batch-fp32-packed" or "batch-fp32-fast")) throw new ArgumentException("Invalid cuda-prefill mode.");
        using var gpu=new CudaPrefixSession(bundle,2048,kvBytes,layerBatch:options.Get("cuda-prefill", "batch-fp32")!="serial",packedWeights:options.Get("cuda-prefill", "batch-fp32")=="batch-fp32-packed",fastFp32:options.Get("cuda-prefill", "batch-fp32")=="batch-fp32-fast");
        using var client=new TcpClient{NoDelay=true};await client.ConnectAsync(endpoint.Host,endpoint.Port,ct);
        var wire=await SplitWire.ConnectAsync(client.GetStream(),token,false,Guid.NewGuid(),ct,kvBytes);
        await ServePrefix(gpu, wire, ct);
    }
    internal static async Task ServePrefix(CudaPrefixSession gpu, SplitWire wire, CancellationToken ct, SplitWire.Frame? first = null, bool overlapSend = false)
    {
        var start=first ?? await wire.ReceiveAsync(ct);if(start.Kind!=SplitWire.Start)throw new InvalidDataException("Expected split request.");
        var tokens=Ints(start.Data);gpu.Reserve(tokens.Length);var verify=(start.Tensor&1)!=0;var incremental=(start.Tensor&2)!=0;var timer=Stopwatch.StartNew();
        double computeMs=0, stateExportMs=0, sendMs=0;
        long sentBytes=0;
        IEnumerable<SplitWire.Frame[]> Chunks()
        {
            for(var offset=0;offset<tokens.Length;offset+=32)
            {
                ct.ThrowIfCancellationRequested();var count=Math.Min(32,tokens.Length-offset);
                var trace=verify?new float[count*20*2048]:null;
                var began=Stopwatch.GetTimestamp();
                var hidden=gpu.Run(tokens.AsSpan(offset,count).ToArray(),offset,trace);
                var frames=new List<SplitWire.Frame> { new(SplitWire.Hidden,20,0,offset,count,Bytes(hidden)) };
                if(trace is not null)frames.Add(new(SplitWire.Trace,0,0,offset,count,Bytes(trace)));
                if(incremental)for(var layer=3;layer<20;layer+=4)for(var kind=0;kind<2;kind++)
                    frames.Add(new(SplitWire.KvRange,layer,kind,offset,count,gpu.KvRange(layer,kind,offset,count)));
                computeMs+=Stopwatch.GetElapsedTime(began).TotalMilliseconds;
                yield return frames.ToArray();
            }
        }
        async ValueTask Send(SplitWire.Frame frame,CancellationToken cancel)
        {
            var began=Stopwatch.GetTimestamp();await wire.SendAsync(frame,cancel);
            sendMs+=Stopwatch.GetElapsedTime(began).TotalMilliseconds;
            sentBytes+=frame.Data.Length;
            if(frame.Kind==SplitWire.Hidden)Console.Error.WriteLine($"PREFIX_PROGRESS tokens={frame.Position+frame.Count}/{tokens.Length} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
        }
        if(overlapSend)await SplitFramePump.RunBatches(Chunks(),Send,ct);
        else foreach(var batch in Chunks())foreach(var frame in batch)await Send(frame,ct);
        Console.Error.WriteLine($"PREFIX_CHUNKS overlap={overlapSend} incrementalKv={incremental} computeMs={computeMs:F3} sendMs={sendMs:F3} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
        for(var layer=0;layer<20;layer++)for(var tensor=0;tensor<2;tensor++)
            if(!incremental||(layer+1)%4!=0)
            {
                var began=Stopwatch.GetTimestamp();
                var data=gpu.State(layer,tensor,tokens.Length);
                stateExportMs+=Stopwatch.GetElapsedTime(began).TotalMilliseconds;
                await Send(new(SplitWire.State,layer,tensor,tokens.Length,1,data),ct);
            }
        var timing=new PrefixAgentTiming(computeMs,stateExportMs,sendMs,timer.Elapsed.TotalMilliseconds,sentBytes);
        await wire.SendAsync(new(SplitWire.End,20,0,tokens.Length,0,timing.ToPayload()),ct);
        Console.Error.WriteLine($"PREFIX_FINISHED computeMs={computeMs:F3} stateExportMs={stateExportMs:F3} sendMs={sendMs:F3} sentBytes={sentBytes} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
    }
    public static async Task Coordinator(Options options,CancellationToken ct)
    {
        var token=RequireToken(options);var kvBytes=KvBytes(options);var uri=new Uri(options.Get("listen","tcp://127.0.0.1:5210"));var ip=IPAddress.Parse(uri.Host);
        if(!IPAddress.IsLoopback(ip)&&!options.Flag("allow-remote"))throw new ArgumentException("Remote split listener requires --allow-remote.");
        var incremental=IncrementalKv(options);var verify=options.Flag("verify");var max=options.Number("max-tokens",8,1,DeepInfer.Core.Protocol.Qwen36Context);
        var output=Path.GetFullPath(options.Get("output","artifacts/split/run-"+DateTime.UtcNow.ToString("yyyyMMddTHHmmss")));
        Directory.CreateDirectory(output);
        Console.Error.WriteLine("Loading full resident Metal model...");
        using var gpu=new MetalSplitSession(options.Get("weights","models/qwen3.6-35b-a3b-8bit-mlx"),kvBytes);
        var prompt=options.Get("prompt-file","") is {Length:>0} file?await File.ReadAllTextAsync(file,ct):options.Get("prompt","用一句话介绍你自己。");
        var tokens=gpu.Tokenize(prompt,ct);
        if((long)tokens.Length+max>DeepInfer.Core.Protocol.Qwen36Context)throw new ArgumentException("Prompt + output exceeds the model's 262144-token window.");
        if(verify&&tokens.Length>128)throw new ArgumentException("Layer-by-layer verification is bounded to 128 prompt tokens; use --verify only for short correctness cases.");
        var expected=new List<int>();var checks=new List<SplitCheck>();float[]? referenceTrace=null;
        var logits=new float[248320];
        if(verify)
        {
            referenceTrace=new float[tokens.Length*20*2048];gpu.Reset(tokens.Length+max);
            var next=gpu.Reference(tokens,ct,logits,(p,l,x)=>{if(l<20)x.CopyTo(referenceTrace,(p*20+l)*2048);});
            await File.WriteAllBytesAsync(Path.Combine(output,"reference-logits-0.f32"),Bytes(logits),ct);
            for(var l=0;l<20;l++)for(var t=0;t<2;t++)
            {var data=new byte[CudaPrefixSession.StateBytes(l,t,tokens.Length,kvBytes)];gpu.State(l,t,tokens.Length,data,false);await File.WriteAllBytesAsync(Path.Combine(output,$"reference-state-{l}-{t}.bin"),data,ct);}
            for(var i=0;i<max;i++)
            {expected.Add(next);if(i+1<max){next=gpu.Forward(next,tokens.Length+i,logits);await File.WriteAllBytesAsync(Path.Combine(output,$"reference-logits-{i+1}.f32"),Bytes(logits),ct);}}
            await File.WriteAllBytesAsync(Path.Combine(output,"reference-prefix-trace.f32"),Bytes(referenceTrace),ct);
            Console.Error.WriteLine($"REFERENCE_READY promptTokens={tokens.Length} greedy={string.Join(',',expected)}");
        }
        gpu.Reset(tokens.Length+max);
        gpu.ConfigureFusedFp32Decode(FusedDecode(options));
        using var listener=new TcpListener(ip,uri.Port);listener.Start(1);
        Console.Error.WriteLine($"SPLIT_LISTENING promptTokens={tokens.Length}");
        using var client=await listener.AcceptTcpClientAsync(ct);client.NoDelay=true;listener.Stop();
        var wire=await SplitWire.ConnectAsync(client.GetStream(),token,true,Guid.Empty,ct,kvBytes);
        var timer=Stopwatch.StartNew();await wire.SendAsync(new(SplitWire.Start,20,(verify?1:0)|(incremental?2:0),0,tokens.Length,Bytes(tokens)),ct);
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
            if(incremental)for(var l=3;l<20;l+=4)for(var t=0;t<2;t++)
            {
                var range=await wire.ReceiveAsync(ct);
                if(range.Kind!=SplitWire.KvRange||range.Layer!=l||range.Tensor!=t||range.Position!=offset||range.Count!=count)
                    throw new InvalidDataException("Missing, duplicate or stale KV range.");
                gpu.KvRange(l,t,offset,count,tokens.Length,range.Data);
                if(verify)
                {
                    var reference=await File.ReadAllBytesAsync(Path.Combine(output,$"reference-state-{l}-{t}.bin"),ct);
                    var slice=reference.AsSpan(offset*512*kvBytes,count*512*kvBytes).ToArray();
                    checks.Add(Compare($"kv/p{offset}/l{l}/t{t}",StateFloats(slice,l,kvBytes),StateFloats(range.Data,l,kvBytes)));
                    await using var stateFile=new FileStream(Path.Combine(output,$"cuda-state-{l}-{t}.bin"),offset==0?FileMode.Create:FileMode.Append);
                    await stateFile.WriteAsync(range.Data,ct);
                }
            }
            Console.Error.WriteLine($"SUFFIX_PROGRESS tokens={offset+count}/{tokens.Length} elapsedMs={timer.Elapsed.TotalMilliseconds:F3}");
        }
        var hiddenReadyMs=timer.Elapsed.TotalMilliseconds;
        var guards=new List<string>();
        void Reject(Action operation,string guard)
        {
            try { operation(); } catch(InvalidOperationException) { guards.Add(guard); return; }
            throw new InvalidOperationException("Handoff safety check failed: "+guard);
        }
        if(verify)Reject(()=>gpu.Forward(nextToken,tokens.Length),"decode-before-state");
        for(var l=0;l<20;l++)for(var t=0;t<2;t++)
        {
            if(incremental&&(l+1)%4==0)continue;
            var f=await wire.ReceiveAsync(ct);if(f.Kind!=SplitWire.State||f.Layer!=l||f.Tensor!=t||f.Position!=tokens.Length)throw new InvalidDataException("Missing, duplicate or stale prefix state.");
            if(verify)
            {
                var reference=await File.ReadAllBytesAsync(Path.Combine(output,$"reference-state-{l}-{t}.bin"),ct);
                await File.WriteAllBytesAsync(Path.Combine(output,$"cuda-state-{l}-{t}.bin"),f.Data,ct);
                checks.Add(Compare($"state/l{l}/t{t}",StateFloats(reference,l,kvBytes),StateFloats(f.Data,l,kvBytes)));
            }
            if(verify&&l==0&&t==0)
            {
                Reject(()=>gpu.State(l,t,tokens.Length+1,f.Data,true),"wrong-state-position");
                Reject(()=>gpu.State(l,t,tokens.Length,f.Data[..^1],true),"wrong-state-size");
            }
            if(verify&&l==(incremental?18:19)&&t==1)Reject(()=>gpu.Forward(nextToken,tokens.Length),"decode-before-last-state");
            gpu.State(l,t,tokens.Length,f.Data,true);
            if(verify&&l==0&&t==0)Reject(()=>gpu.State(l,t,tokens.Length,f.Data,true),"duplicate-state");
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
            verify&&ids.SequenceEqual(expected)&&checks.All(x=>x.Pass),verify,hiddenReadyMs,readyMs,elapsed,checks.ToArray(),guards.ToArray(),$"CUDA layer-major affine8 prefix; Metal <=32-token suffix; incrementalKV={incremental}; decodeAttention={options.Get("decode-attention","legacy")}; KV={(kvBytes==2?"BF16":"FP32 diagnostic")}; FP32 GEMV/recurrent state; authenticated direct LAN; no prefix hit");
        var json=JsonSerializer.Serialize(result,SplitJson.Default.SplitResult);await File.WriteAllTextAsync(Path.Combine(output,"result.json"),json+Environment.NewLine,ct);Console.WriteLine(json);
        if(verify&&!result.Verified)throw new InvalidOperationException("Split correctness gate failed; see result.json. Do not use these timings as accepted performance.");
    }
    internal static bool FusedDecode(Options o)=>o.Get("decode-attention","legacy") switch
    { "fused-fp32"=>true,"legacy"=>false,_=>throw new ArgumentException("--decode-attention must be legacy or fused-fp32.") };
    internal static bool IncrementalKv(Options o)=>o.Get("kv-transfer","final") switch
    { "incremental"=>true,"final"=>false,_=>throw new ArgumentException("--kv-transfer must be final or incremental.") };
    private static int KvBytes(Options o)=>o.Get("kv","fp32") switch { "bf16"=>2,"fp32"=>4,_=>throw new ArgumentException("--kv must be fp32 or bf16.") };
    private static string RequireToken(Options o)=>o.Token is {Length:>=32} t?t:throw new ArgumentException("Set DEEPINFER_TOKEN to a random secret of at least 32 characters.");
    private static byte[] Bytes(float[] x)=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private static byte[] Bytes(int[] x)=>MemoryMarshal.AsBytes(x.AsSpan()).ToArray();
    private static int[] Ints(byte[] x)=>MemoryMarshal.Cast<byte,int>(x).ToArray();
    private static float[] Floats(byte[] x)=>MemoryMarshal.Cast<byte,float>(x).ToArray();
    private static float[] StateFloats(byte[] data,int layer,int kvBytes)=>(layer+1)%4!=0||kvBytes==4?Floats(data):MemoryMarshal.Cast<byte,ushort>(data).ToArray().Select(x=>BitConverter.Int32BitsToSingle(x<<16)).ToArray();
    private static SplitCheck Compare(string name,ReadOnlySpan<float> expected,ReadOnlySpan<float> actual)
    {
        double error=0,norm=0,max=0;for(var i=0;i<expected.Length;i++){var d=(double)expected[i]-actual[i];error+=d*d;norm+=(double)expected[i]*expected[i];max=Math.Max(max,Math.Abs(d));}
        var relative=Math.Sqrt(error/Math.Max(norm,1e-30));return new(name,relative,max,double.IsFinite(relative)&&relative<1e-4&&max<.01);
    }
}
public sealed record SplitCheck(string Tensor,double RelativeL2,double MaxAbs,bool Pass);
public sealed record SplitResult(string Request,string Revision,int PromptTokens,int[] Tokens,string Text,int[] ReferenceTokens,bool Verified,bool VerificationEnabled,double HiddenReadyMs,double DecodeReadyMs,double ElapsedMs,SplitCheck[] Checks,string[] PassedSafetyChecks,string Execution);
[JsonSerializable(typeof(SplitResult))]
[JsonSourceGenerationOptions(WriteIndented=true)]
internal partial class SplitJson:JsonSerializerContext;
