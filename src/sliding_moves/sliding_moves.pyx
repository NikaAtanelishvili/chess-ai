from libc.stdint cimport uint64_t, uint8_t, uint16_t


cdef extern from "sliding_moves_impl.h":
    void init_rook_tables(const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts, const uint64_t *attacks)

    void init_bishop_tables(const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts,
                            const uint64_t *attacks)

    void generate_moves(uint64_t pieces, uint64_t occupancy, uint64_t own_cc,
                        uint16_t *moves, int *move_count, int is_rook)


def initialize_rook_tables(rook_magics, rook_masks, rook_shifts, rook_attacks):
    """Initialize rook tables from Python."""
    cdef uint64_t[:] rook_magics_view = rook_magics
    cdef uint64_t[:] rook_masks_view = rook_masks
    cdef uint8_t[:] rook_shifts_view = rook_shifts
    cdef uint64_t[:] rook_attacks_view = rook_attacks.flatten()
    init_rook_tables(&rook_magics_view[0], &rook_masks_view[0], &rook_shifts_view[0], &rook_attacks_view[0])

def initialize_bishop_tables(bishop_magics, bishop_masks, bishop_shifts, bishop_attacks):
    """Initialize bishop tables from Python."""
    cdef uint64_t[:] bishop_magics_view = bishop_magics
    cdef uint64_t[:] bishop_masks_view = bishop_masks
    cdef uint8_t[:] bishop_shifts_view = bishop_shifts
    cdef uint64_t[:] bishop_attacks_view = bishop_attacks.flatten()

    init_bishop_tables(&bishop_magics_view[0], &bishop_masks_view[0], &bishop_shifts_view[0], &bishop_attacks_view[0])


def generate_moves_wrapper(uint64_t pieces, uint64_t occupancy, uint64_t own_occ, int is_rook):
    cdef int move_count = 0
    cdef uint16_t[1024] moves  # Adjust size as needed

    generate_moves(pieces, occupancy, own_occ, &moves[0], &move_count, is_rook)

    return [moves[i] for i in range(move_count)]
