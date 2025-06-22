#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define FLAG_KING_SINGLE      4
#define FLAG_CASTLE_KINGSIDE  5
#define FLAG_CASTLE_QUEENSIDE 6

static uint64_t king_attacks[64];


void initialize_king_attacks(const uint64_t *attacks) {
    memcpy(king_attacks, attacks, 64 * sizeof(uint64_t));
}


static inline int lsb_index(uint64_t bb) {
    return __builtin_ctzll(bb); // GCC/Clang intrinsic
}


size_t generate_king_moves(uint64_t occupancy, uint64_t own_occ, int from_sq,
                                  int can_ks, int can_qs, int color, uint32_t *moves) {
    size_t count = 0;

    uint64_t targets = king_attacks[from_sq] & ~own_occ;

    // Single-step
    while (targets) {
        uint64_t bit = targets & -targets;
        int to_sq = __builtin_ctzll(bit);
        targets &= targets - 1;

        uint32_t code = (from_sq << 6) | to_sq | (FLAG_KING_SINGLE << 16);
        moves[count++] = code;
    }

    // Castling
    if (color) {
        // White
        // Kingside: f1 (5), g1 (6)
        if (can_ks && !(occupancy & ((1ULL << 5) | (1ULL << 6)))) {
            moves[count++] = (4 << 6) | 6 | (FLAG_CASTLE_KINGSIDE << 16);
        }
        // Queenside: d1 (3), c1 (2)
        if (can_qs && !(occupancy & ((1ULL << 3) | (1ULL << 2)))) {
            moves[count++] = (4 << 6) | 2 | (FLAG_CASTLE_QUEENSIDE << 16);
        }
    } else {
        // Black
        // Kingside: f8 (61), g8 (62)
        if (can_ks && !(occupancy & ((1ULL << 61) | (1ULL << 62)))) {
            moves[count++] = (60 << 6) | 62 | (FLAG_CASTLE_KINGSIDE << 16);
        }
        // Queenside: d8 (59), c8 (58)
        if (can_qs && !(occupancy & ((1ULL << 59) | (1ULL << 58)))) {
            moves[count++] = (60 << 6) | 58 | (FLAG_CASTLE_QUEENSIDE << 16);
        }
    }

    return count;
}



