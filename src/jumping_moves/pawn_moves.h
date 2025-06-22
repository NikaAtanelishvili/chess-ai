#pragma once

#include <stdint.h>
#include <stddef.h>
#include "bitutil.h"


// Initializes the internal pawn capture tables (64 entries each)
void initialize_pawn_attacks(const uint64_t *white_attacks, const uint64_t *black_attacks);

// Generates pseudo-legal pawn moves and writes them into `moves_out`.
// Returns the number of moves written.
// `color` should be 0 for black, 1 for white.
// `ep_sq` should be -1 if there is no en-passant square.
size_t generate_pawn_moves(uint64_t pieces,
                                  uint64_t occupancy,
                                  uint64_t opp_occ,
                                  int color,
                                  int ep_sq,
                                  uint32_t* moves);

//// Utility functions
//static inline int lsb_index(uint64_t bb) {
//    return __builtin_ctzll(bb);
//}

static inline int is_backrank(int sq, int color) {
    return (color && sq >= 56) || (!color && sq < 8);
}

