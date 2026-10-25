namespace DeepInfer.Backends.Mtp;

// Experimental envelope of the Issue #76 official-MTP trial (context 32768, output cap 4096). Enforced
// before any tokenization-dependent allocation; larger requests are rejected, never truncated or sent
// down a non-MTP path.
public static class MtpLimits
{
    public const int MaxContextTokens = 32768;
    public const int MaxOutputTokens = 4096;

    // Output-length check that needs no tokenization.
    public static string? CheckOutput(int maxTokens, bool useRemainingContext)
    {
        if (useRemainingContext) return $"MTP mode does not support remaining-context outputs; request at most {MaxOutputTokens} output tokens.";
        return maxTokens > MaxOutputTokens ? $"MTP mode supports at most {MaxOutputTokens} output tokens (requested {maxTokens})." : null;
    }

    public static string? Check(int promptTokens, int maxTokens)
    {
        if (maxTokens < 1) return "MTP mode requires at least one output token.";
        if (maxTokens > MaxOutputTokens) return $"MTP mode supports at most {MaxOutputTokens} output tokens (requested {maxTokens}).";
        if ((long)promptTokens + maxTokens > MaxContextTokens) return $"MTP mode supports at most {MaxContextTokens} prompt+output tokens (requested {(long)promptTokens + maxTokens}).";
        return null;
    }
}
