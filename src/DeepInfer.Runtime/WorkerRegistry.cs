using DeepInfer.Core;
namespace DeepInfer.Runtime;

public sealed class WorkerRegistry(TimeProvider? timeProvider = null, TimeSpan? lease = null)
{
    private readonly TimeProvider clock = timeProvider ?? TimeProvider.System;
    private readonly TimeSpan ttl = lease ?? TimeSpan.FromSeconds(8);
    private readonly Dictionary<string, WorkerSnapshot> workers = new(StringComparer.Ordinal);
    private readonly object gate = new();
    private long cursor;

    public void Register(WorkerRegistration registration, bool allowRemote)
    {
        if (registration.ProtocolVersion != Protocol.Version) throw new ArgumentException("Unsupported protocol version.");
        if (string.IsNullOrWhiteSpace(registration.WorkerId) || registration.WorkerId.Length > 100)
            throw new ArgumentException("Worker ID must have 1..100 characters.");
        if (registration.Capacity is < 1 or > 64) throw new ArgumentException("Capacity must be 1..64.");
        if (registration.Devices is null || registration.Models is null || registration.Devices.Length > 32 || registration.Models.Length > 32)
            throw new ArgumentException("Invalid device/model advertisement.");
        if (registration.Devices.Any(d => d is null || string.IsNullOrWhiteSpace(d.Id) || string.IsNullOrWhiteSpace(d.Backend)) ||
            registration.Models.Any(m => m is null || string.IsNullOrWhiteSpace(m.Id)))
            throw new ArgumentException("Device and model entries must be non-null and have identifiers.");
        EndpointRules.Validate(registration.Endpoint, allowRemote);
        lock (gate)
        {
            // Bound registry size and reclaim expired registrations.
            foreach (var id in workers.Where(x => clock.GetUtcNow() - x.Value.LastSeenUtc >= ttl).Select(x => x.Key).ToArray())
                workers.Remove(id);
            if (!workers.ContainsKey(registration.WorkerId) && workers.Count >= 1024)
                throw new ArgumentException("Registry is full.");
            workers[registration.WorkerId] = new(registration, clock.GetUtcNow(), true);
        }
    }
    public ClusterSnapshot Snapshot()
    {
        lock (gate) return new(Protocol.Version, workers.Values.OrderBy(x => x.Registration.WorkerId, StringComparer.Ordinal)
            .Select(x => x with { Healthy = clock.GetUtcNow() - x.LastSeenUtc < ttl }).ToArray());
    }
    public WorkerRegistration? Select(InferenceRequest request)
    {
        lock (gate)
        {
            var eligible = Snapshot().Workers.Where(x => x.Healthy &&
                x.Registration.Devices.Any(d => d.SupportsInference && d.Backend == request.Backend) &&
                x.Registration.Models.Any(m => m.Id == request.Model && (request.Messages == null || m.Format.EndsWith("-chat-v1", StringComparison.Ordinal)))).ToArray();
            if (eligible.Length == 0) return null;
            return eligible[(int)((ulong)cursor++ % (ulong)eligible.Length)].Registration;
        }
    }
}

public static class EndpointRules
{
    public static Uri Validate(string endpoint, bool allowRemote)
    {
        if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var uri) ||
            (uri.Scheme != "http" && uri.Scheme != "https") || uri.AbsolutePath != "/" ||
            uri.Query.Length != 0 || uri.Fragment.Length != 0 || uri.UserInfo.Length != 0)
            throw new ArgumentException("Endpoint must be an HTTP(S) origin without a path, query or credentials.");
        if (!allowRemote && !IsLoopback(uri)) throw new ArgumentException("Remote endpoints require --allow-remote.");
        return uri;
    }
    public static bool IsLoopback(Uri uri) => uri.Host.Equals("localhost", StringComparison.OrdinalIgnoreCase) ||
        (System.Net.IPAddress.TryParse(uri.Host.Trim('[', ']'), out var ip) && System.Net.IPAddress.IsLoopback(ip));
}
