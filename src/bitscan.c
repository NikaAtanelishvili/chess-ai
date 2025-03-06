#include <stdint.h>

int bit_scan(uint64_t bitboard, int* restrict squares) {
    int count = 0;
    while (bitboard) {
        squares[count++] = __builtin_ctzll(bitboard); // Get least significant set bit
        bitboard &= bitboard - 1;  // Clear the least significant set bit
    }
    return count;
}
