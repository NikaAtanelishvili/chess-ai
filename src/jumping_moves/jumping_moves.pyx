from libc.stdint cimport uint64_t, uint32_t

cdef extern from "pawn_moves.h":
    void initialize_pawn_attacks(const uint64_t *white_attacks, const uint64_t *black_attacks)
    size_t generate_pawn_moves(uint64_t pieces, uint64_t occupancy, uint64_t opp_occ,
                                      int color, int ep_sq, uint32_t* moves)

cdef extern  from "king_moves.h":
    void initialize_king_attacks(const uint64_t *attacks)
    size_t generate_king_moves(uint64_t occupancy, uint64_t own_occ,int from_sq,
                                       int can_ks, int can_qs, int color, uint32_t *moves)

cdef extern from "knight_moves.h":
    void initialize_knight_attacks(const uint64_t *attacks)
    size_t generate_knight_moves(uint64_t pieces, uint64_t own_occ,
                                 int color, uint32_t *moves)

# ATTACK TABLE INITIALIZATIONS
def initialize_pawn_tables( white_attacks, black_attacks):

    cdef uint64_t[:] white_attacks_view = white_attacks

    cdef uint64_t[:] black_attacks_view = black_attacks

    initialize_pawn_attacks(&white_attacks_view[0], &black_attacks_view[0])


def initialize_king_table(attacks):

    cdef uint64_t[:] attacks_view = attacks

    initialize_king_attacks(&attacks_view[0])


def initialize_knight_table(attacks):

    cdef uint64_t[:] attacks_view = attacks

    initialize_knight_attacks(&attacks_view[0])

# WRAPPERS
def generate_pawn_moves_wrapper(uint64_t pieces,
                                uint64_t occupancy,
                                uint64_t opp_occ,
                                int color,
                                int ep_sq):
    cdef uint32_t moves[256]  # buffer to hold generated moves

    cdef size_t n = generate_pawn_moves(pieces, occupancy, opp_occ, color, ep_sq, &moves[0])

    return [moves[i] for i in range(n)]


def generate_king_moves_wrapper(uint64_t occupancy,
                                uint64_t own_occ,
                                int from_sq,
                                int can_ks,
                                int can_qs,
                                int color):
    cdef uint32_t moves[64]  # King can max move 8 + 2 castle = 10

    cdef size_t n = generate_king_moves(occupancy, own_occ, from_sq, can_ks, can_qs, color, &moves[0])

    return [moves[i] for i in range(n)]


def generate_knight_moves_wrapper(uint64_t own_occ,
                                  uint64_t pieces,
                                  int color):
    cdef uint32_t moves[64]

    cdef size_t n = generate_knight_moves(pieces, own_occ, color, &moves[0])

    return [moves[i] for i in range(n)]


