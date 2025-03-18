#ifndef SLIDING_MOVES_H
#define SLIDING_MOVES_H

#include <stdint.h>

static inline int bit_scan_forward(uint64_t bb) {
    return __builtin_ctzll(bb);  // Count trailing zeros
}

void generate_moves(uint64_t pieces, uint64_t occupancy, uint64_t own_occ,
                    const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts,
                    uint64_t **attacks,
                    uint64_t *moves, int *move_count);

#endif