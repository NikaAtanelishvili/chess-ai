#pragma once

#include <stdint.h>
#include <stddef.h>
#include "bitutil.h"

// Initializes the internal king capture table
void initialize_king_attacks(const uint64_t *attacks);

// Generates pseudo-legal king moves and writes them into `moves_out`.
// Returns the number of moves written.
// `color` should be 0 for black, 1 for white.
// `can_ks, can_qs` are boolians whether king can castle on king side/queen side
size_t generate_king_moves(uint64_t occupancy, uint64_t own_occ, int from_sq,
                                  int can_ks, int can_qs, int color, uint32_t *moves);

//
//static inline int lsb_index(uint64_t bb) {
//    return __builtin_ctzll(bb);
//}

