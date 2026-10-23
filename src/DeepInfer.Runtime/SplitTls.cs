using System.Net.Security;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
namespace DeepInfer.Runtime;

// Explicit certificate pins identify the two private cluster members. No trust-all fallback.
public static class SplitTls
{
    // TcpClient.Client becomes null after Dispose; telemetry races with request teardown.
    public static bool IsConnected(TcpClient client)
    {
        try
        {
            var socket = client.Client;
            return socket is not null && socket.Connected &&
                !(socket.Poll(0, SelectMode.SelectRead) && socket.Available == 0);
        }
        catch (Exception e) when (e is ObjectDisposedException or SocketException) { return false; }
    }
    public static bool Matches(X509Certificate? certificate, string pin)
    {
        if (certificate is null || pin.Length != 64 || !pin.All(Uri.IsHexDigit)) return false;
        using var cert = X509CertificateLoader.LoadCertificate(certificate.GetRawCertData());
        return DateTime.UtcNow >= cert.NotBefore.ToUniversalTime() && DateTime.UtcNow <= cert.NotAfter.ToUniversalTime() &&
            CryptographicOperations.FixedTimeEquals(SHA256.HashData(cert.RawData), Convert.FromHexString(pin));
    }
    public static async Task<SslStream> Authenticate(Stream stream, X509Certificate2 certificate, string peerPin, bool server, CancellationToken ct)
    {
        if (!certificate.HasPrivateKey || peerPin.Length != 64 || !peerPin.All(Uri.IsHexDigit))
            throw new ArgumentException("A private TLS certificate and a SHA256 peer certificate pin are required.");
        var tls = new SslStream(stream, false, (_, cert, _, _) => Matches(cert, peerPin));
        try
        {
            if (server) await tls.AuthenticateAsServerAsync(new SslServerAuthenticationOptions
            {
                ServerCertificate = certificate, ClientCertificateRequired = true,
                EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
                CertificateRevocationCheckMode = X509RevocationMode.NoCheck
            }, ct);
            else await tls.AuthenticateAsClientAsync(new SslClientAuthenticationOptions
            {
                TargetHost = "infer-split", ClientCertificates = new X509CertificateCollection { certificate },
                EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
                CertificateRevocationCheckMode = X509RevocationMode.NoCheck
            }, ct);
            return tls;
        }
        catch { await tls.DisposeAsync(); throw; }
    }
}

// One GPU executor plus a bounded number of waiting requests. Cancellation never leaks a slot.
public sealed class InferenceQueue(int waitingCapacity) : IDisposable
{
    private readonly SemaphoreSlim admitted = new(checked(waitingCapacity + 1));
    private readonly SemaphoreSlim execution = new(1);
    public bool Busy => execution.CurrentCount == 0;
    public int WaitingCapacity => waitingCapacity;
    public int WaitingCount => Math.Max(0, waitingCapacity + 1 - admitted.CurrentCount - (Busy ? 1 : 0));
    public async Task<IDisposable?> Enter(CancellationToken ct)
    {
        if (!await admitted.WaitAsync(0, ct)) return null;
        try { await execution.WaitAsync(ct); return new Lease(this); }
        catch { admitted.Release(); throw; }
    }
    private sealed class Lease(InferenceQueue queue) : IDisposable
    {
        private int disposed;
        public void Dispose() { if (Interlocked.Exchange(ref disposed, 1) == 0) { queue.execution.Release(); queue.admitted.Release(); } }
    }
    public void Dispose() { admitted.Dispose(); execution.Dispose(); }
}
