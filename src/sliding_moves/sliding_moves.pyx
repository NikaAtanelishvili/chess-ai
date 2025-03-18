from libc.stdint cimport uint64_t, uint8_t

cdef extern from "sliding_moves_impl.h":
    void generate_moves(uint64_t pieces, uint64_t occupancy, uint64_t own_occ,
                        const uint64_t *magics, const uint64_t *masks, const uint8_t *shifts,
                        uint64_t **attacks,  # Simplified for Cython compatibility
                        uint64_t *moves, int *move_count)


def generate_moves_wrapper(uint64_t pieces, uint64_t occupancy, uint64_t own_occ,
                   uint64_t[:] magics, uint64_t[:] masks, uint8_t[:] shifts,
                   uint64_t[:, :] attacks):  # 2D memoryview for ATTACKS

    cdef int move_count = 0
    cdef uint64_t[1024] moves  # Adjust size as needed

    # Convert 2D memoryview to array of pointers
    cdef uint64_t *attacks_ptrs[64]
    for i in range(64):
        attacks_ptrs[i] = &attacks[i, 0]

    generate_moves(pieces, occupancy, own_occ,
                           &magics[0], &masks[0], &shifts[0],
                           attacks_ptrs,
                           moves, &move_count)

    return [moves[i] for i in range(move_count)]