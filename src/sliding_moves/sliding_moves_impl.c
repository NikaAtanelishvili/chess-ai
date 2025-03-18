#include <stdint.h>


static inline int bit_scan_forward(uint64_t bb) {
    return __builtin_ctzll(bb);  // Count trailing zeros (GCC/Clang)
}

void generate_moves(uint64_t pieces, uint64_t occupancy, uint64_t own_occ,
                            const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts,
                            uint64_t **attacks,
                            uint64_t *moves, int *move_count) {
    *move_count = 0;

    while (pieces) {
        int square = bit_scan_forward(pieces);
        pieces &= pieces - 1;  // Clear the scanned bit

        uint64_t relevant_occ = occupancy & masks[square];
        uint64_t magic = magics[square];
        uint8_t shift = shifts[square];
        uint64_t index = (relevant_occ * magic) >> shift;
        uint64_t valid_attacks = attacks[square][index] & ~own_occ;

        while (valid_attacks) {
            int target = bit_scan_forward(valid_attacks);
            valid_attacks &= valid_attacks - 1;
            moves[*move_count] = ((uint64_t)square << 6) | target;
            (*move_count)++;
        }
    }
}

