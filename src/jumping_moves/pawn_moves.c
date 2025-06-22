#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define BB_ALL    0xFFFFFFFFFFFFFFFFULL
// rank 3 squares a3–h3 are bits 16–23:
#define BB_RANK_3 0x0000000000FF0000ULL
// rank 6 squares a6–h6 are bits 40–47:
#define BB_RANK_6 0x0000FF0000000000ULL


#define PROMO_QUEEN  5
#define PROMO_ROOK   4
#define PROMO_BISHOP 3
#define PROMO_KNIGHT 2

#define FLAG_SINGLE  0
#define FLAG_DOUBLE  1
#define FLAG_CAPTURE 2
#define FLAG_EP      3

static uint64_t pawn_white_attacks[64];
static uint64_t pawn_black_attacks[64];


static inline int lsb_index(uint64_t bb) {
    return __builtin_ctzll(bb); // GCC/Clang intrinsic
}


static inline int is_backrank(int sq, int color) {
    return (color && sq >= 56) || (!color && sq < 8);
}


void initialize_pawn_attacks(const uint64_t *white_attacks, const uint64_t *black_attacks) {
    memcpy(pawn_white_attacks, white_attacks, 64 * sizeof(uint64_t));
    memcpy(pawn_black_attacks, black_attacks, 64 * sizeof(uint64_t));
}


size_t generate_pawn_moves(uint64_t pieces, uint64_t occupancy, uint64_t opp_occ,
                                   int color, int ep_sq, uint32_t* moves) {
    uint64_t empty = ~occupancy & BB_ALL;
    uint64_t one, two;
    int delta1, delta2;

    size_t count = 0;

    one = color ? (pieces << 8) & empty : (pieces >> 8) & empty;
    two = color ? ((one & BB_RANK_3) << 8) & empty : ((one & BB_RANK_6) >> 8) & empty;
    delta1 =  color ? 8 : -8;
    delta2 = color ? 16 : -16;

    // single pushes
    while (one) {
        uint64_t bit = one & -one;
        int to_sq = lsb_index(bit);
        one &= one - 1;
        int from_sq = to_sq - delta1;
        uint32_t base = (from_sq << 6) | to_sq | (FLAG_SINGLE << 16);
        if (is_backrank(to_sq, color)) {
            for (int promo = PROMO_QUEEN; promo >= PROMO_KNIGHT; --promo)
                moves[count++] = base | (promo << 12);
        } else {
            moves[count++] = base;
        }
    }

    // double pushes
    while (two) {
        uint64_t bit = two & -two;
        int to_sq = lsb_index(bit);
        two &= two - 1;
        int from_sq = to_sq - delta2;
        uint32_t base = (from_sq << 6) | to_sq | (FLAG_DOUBLE << 16);
        moves[count++] = base;
    }

    // captures & en passant
    while (pieces) {
        uint64_t bit = pieces & -pieces;
        int from_sq = lsb_index(bit);
        pieces &= pieces - 1;

        uint64_t cap_targets = color ? pawn_white_attacks[from_sq] : pawn_black_attacks[from_sq];
        uint64_t caps = cap_targets & opp_occ;

        while (caps) {
            uint64_t cb = caps & -caps;
            int to_sq = lsb_index(cb);
            caps &= caps - 1;
            uint32_t base = (from_sq << 6) | to_sq | (FLAG_CAPTURE << 16);
            if (is_backrank(to_sq, color)) {
                for (int promo = PROMO_QUEEN; promo >= PROMO_KNIGHT; --promo)
                    moves[count++] = base | (promo << 12);
            } else {
                moves[count++] = base;
            }
        }

        // en passant
        if (ep_sq >= 0 && (cap_targets & (1ULL << ep_sq))) {
            uint32_t base = (from_sq << 6) | ep_sq | (FLAG_EP << 16);
            if (is_backrank(ep_sq, color)) {
                for (int promo = PROMO_QUEEN; promo >= PROMO_KNIGHT; --promo)
                    moves[count++] = base | (promo << 12);
            } else {
                moves[count++] = base;
            }
        }
    }

    return count;
}

