#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// KV profile 1 is block32 Q8, not one byte per logical component.
inline size_t di_kv_token_bytes(int profile) { return profile == 1 ? 544 : size_t(512) * profile; }
inline bool di_q8_valid(const void *data, size_t bytes) {
    if (!data || bytes % 34) return false;
    auto *p = static_cast<const unsigned char *>(data);
    for (size_t at = 0; at < bytes; at += 34) {
        uint16_t scale; std::memcpy(&scale, p + at, 2);
        if ((scale & 0x8000) || (scale & 0x7c00) == 0x7c00) return false;
        for (int j = 0; j < 32; ++j)
            if (p[at + 2 + j] == 128 || (!scale && p[at + 2 + j])) return false;
    }
    return true;
}
