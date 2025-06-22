#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define FLAG_KNIGHT_MOVE 7

static uint64_t knight_attacks[64];

void initialize_knight_attacks(const uint64_t *attacks) {
    memcpy(knight_attacks, attacks, 64 * sizeof(uint64_t));
}


static inline int lsb_index(uint64_t bb) {
    return __builtin_ctzll(bb); // GCC/Clang intrinsic
}


size_t generate_knight_moves(uint64_t own_occ,
                             uint64_t knights_bb,
                             int color,
                             uint32_t *moves_out) {
    size_t count = 0;

    while (knights_bb) {
        uint64_t bit = knights_bb & -knights_bb;
        int from_sq = lsb_index(bit);
        knights_bb &= knights_bb - 1;

        uint64_t targets = knight_attacks[from_sq] & ~own_occ;
        while (targets) {
            uint64_t tbit = targets & -targets;
            int to_sq = lsb_index(tbit);
            targets &= targets - 1;

            uint32_t base = (from_sq << 6) | to_sq | (FLAG_KNIGHT_MOVE << 16);
            moves_out[count++] = base;
        }
    }

    return count;
}