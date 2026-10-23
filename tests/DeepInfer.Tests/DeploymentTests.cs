using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using DeepInfer.Runtime;
using Xunit;
namespace DeepInfer.Tests;

public sealed class DeploymentTests
{
    private static X509Certificate2 Cert(string name)
    {
        using var key = RSA.Create(2048);
        var request = new CertificateRequest("CN=" + name, key, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        using var cert = request.CreateSelfSigned(DateTimeOffset.UtcNow.AddMinutes(-1), DateTimeOffset.UtcNow.AddHours(1));
        return X509CertificateLoader.LoadPkcs12(cert.Export(X509ContentType.Pkcs12), null, OperatingSystem.IsWindows() ? X509KeyStorageFlags.MachineKeySet : X509KeyStorageFlags.EphemeralKeySet);
    }
    private static string Pin(X509Certificate2 cert) => Convert.ToHexString(SHA256.HashData(cert.RawData));
    [Fact]
    public async Task MutualTlsCarriesAuthenticatedFrames()
    {
        using var serverCert = Cert("mac"); using var clientCert = Cert("windows");
        using var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10)); var ct = deadline.Token;
        var receive = Task.Run(async () =>
        {
            using var accepted = await listener.AcceptTcpClientAsync(ct);
            await using var tls = await SplitTls.Authenticate(accepted.GetStream(), serverCert, Pin(clientCert), true, ct);
            var wire = await SplitWire.ConnectAsync(tls, new string('a', 64), true, Guid.Empty, ct, 4);
            var frame = await wire.ReceiveAsync(ct); Assert.Equal(SplitWire.Start, frame.Kind);
            await wire.SendAsync(new(SplitWire.End, 20, 0, 1, 0, []), ct);
        }, ct);
        using var client = new TcpClient(); await client.ConnectAsync((IPEndPoint)listener.LocalEndpoint, ct);
        await using var stream = await SplitTls.Authenticate(client.GetStream(), clientCert, Pin(serverCert), false, ct);
        Assert.True(stream.IsMutuallyAuthenticated); Assert.True(stream.IsEncrypted);
        var sender = await SplitWire.ConnectAsync(stream, new string('a', 64), false, Guid.NewGuid(), ct, 4);
        await sender.SendAsync(new(SplitWire.Start, 20, 0, 0, 1, BitConverter.GetBytes(42)), ct);
        Assert.Equal(SplitWire.End, (await sender.ReceiveAsync(ct)).Kind); await receive;
    }
    [Fact]
    public async Task WrongPeerCertificateFailsTls()
    {
        using var serverCert = Cert("mac"); using var clientCert = Cert("windows"); using var wrong = Cert("other");
        using var listener = new TcpListener(IPAddress.Loopback, 0); listener.Start();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5)); var ct = deadline.Token;
        var receive = Task.Run(async () =>
        {
            using var accepted = await listener.AcceptTcpClientAsync(ct);
            try { await using var tls = await SplitTls.Authenticate(accepted.GetStream(), serverCert, Pin(clientCert), true, ct); }
            catch (Exception ex) when (ex is System.Security.Authentication.AuthenticationException or IOException) { }
        }, ct);
        using var client = new TcpClient(); await client.ConnectAsync((IPEndPoint)listener.LocalEndpoint, ct);
        var failure = await Record.ExceptionAsync(() => SplitTls.Authenticate(client.GetStream(), clientCert, Pin(wrong), false, ct));
        Assert.True(failure is System.Security.Authentication.AuthenticationException or IOException, failure?.ToString() ?? "Unexpected TLS success");
        await receive;
    }
    [Fact]
    public void PinsRequireExactCertificateAndValidity()
    {
        using var cert = Cert("one"); using var other = Cert("two");
        Assert.True(SplitTls.Matches(cert, Pin(cert))); Assert.False(SplitTls.Matches(other, Pin(cert)));
        Assert.False(SplitTls.Matches(cert, "invalid")); Assert.False(SplitTls.Matches(null, Pin(cert)));
    }
    [Fact]
    public async Task QueueBoundsAndCancellationReleaseAdmission()
    {
        using var queue = new InferenceQueue(1);
        using var running = await queue.Enter(CancellationToken.None); Assert.NotNull(running);
        using var cancel = new CancellationTokenSource(); var waiting = queue.Enter(cancel.Token);
        Assert.Null(await queue.Enter(CancellationToken.None)); cancel.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
        var replacement = queue.Enter(CancellationToken.None); Assert.False(replacement.IsCompleted);
        running.Dispose(); using var next = await replacement; Assert.NotNull(next);
    }
    [Fact]
    public async Task NoQueueRejectsOverlappingRequest()
    {
        using var queue = new InferenceQueue(0);
        using var first = await queue.Enter(CancellationToken.None);
        Assert.Null(await queue.Enter(CancellationToken.None)); first!.Dispose();
        using var next = await queue.Enter(CancellationToken.None); Assert.NotNull(next);
    }
}
