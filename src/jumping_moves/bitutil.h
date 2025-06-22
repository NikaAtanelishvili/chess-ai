#pragma once
#include <stdint.h>

static inline int lsb_index(uint64_t bb) {
    return __builtin_ctzll(bb);
}
