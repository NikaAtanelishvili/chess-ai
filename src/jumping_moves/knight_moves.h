#pragma once

#include <stdint.h>
#include <stddef.h>
#include "bitutil.h"


// Initializes the internal knight capture table
void initialize_knight_attacks(const uint64_t *attacks);


// Generates pseudo-legal knight moves and writes them into `moves`.
// Returns the number of moves written.
size_t generate_knight_moves(uint64_t pieces,
                             uint64_t own_occ,
                             int color,
                             uint32_t *moves_out);


//static inline int lsb_index(uint64_t bb) {
//    return __builtin_ctzll(bb);
//}

