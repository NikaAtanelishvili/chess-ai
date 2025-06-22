#include <stdint.h>
#include <string.h>

#define MAX_ENTRIES 4096

// Static global variables for Rook
static uint64_t rook_attacks[64 * MAX_ENTRIES];
static uint64_t rook_magics[64];
static uint64_t rook_masks[64];
static uint8_t rook_shifts[64];

// Static global variables for Bishop
static uint64_t bishop_attacks[64 * MAX_ENTRIES];
static uint64_t bishop_magics[64];
static uint64_t bishop_masks[64];
static uint8_t bishop_shifts[64];

static int rook_initialized = 0;
static int bishop_initialized = 0;


// Initialize rook tables
void init_rook_tables(const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts, const uint64_t *attacks) {
    memcpy(rook_magics, magics, sizeof(uint64_t) * 64);
    memcpy(rook_masks, masks, sizeof(uint64_t) * 64);
    memcpy(rook_shifts, shifts, sizeof(uint8_t) * 64);
    memcpy(rook_attacks, attacks, sizeof(uint64_t) * 64 * MAX_ENTRIES);
    rook_initialized = 1;
}


// Initialize bishop tables
void init_bishop_tables(const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts, const uint64_t *attacks) {
    memcpy(bishop_magics, magics, sizeof(uint64_t) * 64);
    memcpy(bishop_masks, masks, sizeof(uint64_t) * 64);
    memcpy(bishop_shifts, shifts, sizeof(uint8_t) * 64);
    memcpy(bishop_attacks, attacks, sizeof(uint64_t) * 64 * MAX_ENTRIES);
    bishop_initialized = 1;
}


static inline int bit_scan_forward(uint64_t bb) {
    return __builtin_ctzll(bb);  // Count trailing zeros (GCC/Clang)
}


void generate_moves(uint64_t pieces, uint64_t occupancy, uint64_t own_occ,
                    uint16_t *moves, int *move_count, int is_rook) {
    *move_count = 0;
    if (is_rook && !rook_initialized) return;
    if (!is_rook && !bishop_initialized) return;

    const uint64_t *magics  = is_rook ? rook_magics  : bishop_magics;
    const uint64_t *masks   = is_rook ? rook_masks   : bishop_masks;
    const uint8_t  *shifts  = is_rook ? rook_shifts  : bishop_shifts;
    const uint64_t *attacks = is_rook ? rook_attacks : bishop_attacks;


    while (pieces) {
        int square = bit_scan_forward(pieces);
        pieces &= pieces - 1;

        uint64_t relevant_occ = occupancy & masks[square];
        uint64_t magic_index = (relevant_occ * magics[square]) >> shifts[square];

        uint64_t valid_attacks = attacks[square * MAX_ENTRIES + magic_index] & ~own_occ;

        while (valid_attacks) {
            int target = bit_scan_forward(valid_attacks);
            valid_attacks &= valid_attacks - 1;
            moves[*move_count] = ((uint16_t)square << 6) | target;
            (*move_count)++;
        }
    }
}
