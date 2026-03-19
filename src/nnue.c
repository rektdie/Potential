#include "nnue.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bit_manipulation.h"
#include "board_constants.h"
#include "incbin.h"
#include "values.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef EVALFILE
#define EVALFILE "../nnue.bin"
#endif

#if !defined(_MSC_VER) || defined(__clang__)
INCBIN(EVAL, EVALFILE);
#endif

#define NNUE_PATH_MAX 4096

typedef struct {
    int16_t accumulator_weights[NNUE_INPUT_SIZE * NNUE_HL_SIZE];
    int16_t accumulator_biases[NNUE_HL_SIZE];
    int16_t output_weights[NNUE_OUTPUT_BUCKETS][2 * NNUE_HL_SIZE];
    int16_t output_bias[NNUE_OUTPUT_BUCKETS];
} NNUENetwork;

static NNUENetwork g_nnue_network;
static bool g_nnue_loaded = false;
static char g_nnue_path[NNUE_PATH_MAX];

static size_t nnue_required_bytes(void) {
    return (sizeof(g_nnue_network.accumulator_weights) +
            sizeof(g_nnue_network.accumulator_biases) +
            sizeof(g_nnue_network.output_weights) +
            sizeof(g_nnue_network.output_bias));
}

static bool nnue_load_from_memory(const unsigned char *data, size_t size, const char *label) {
    if (data == NULL || size < nnue_required_bytes()) {
        return false;
    }

    const unsigned char *cursor = data;

    memcpy(g_nnue_network.accumulator_weights, cursor, sizeof(g_nnue_network.accumulator_weights));
    cursor += sizeof(g_nnue_network.accumulator_weights);

    memcpy(g_nnue_network.accumulator_biases, cursor, sizeof(g_nnue_network.accumulator_biases));
    cursor += sizeof(g_nnue_network.accumulator_biases);

    memcpy(g_nnue_network.output_weights, cursor, sizeof(g_nnue_network.output_weights));
    cursor += sizeof(g_nnue_network.output_weights);

    memcpy(g_nnue_network.output_bias, cursor, sizeof(g_nnue_network.output_bias));

    snprintf(g_nnue_path, sizeof(g_nnue_path), "%s", label == NULL ? "<memory>" : label);
    g_nnue_loaded = true;
    return true;
}

static bool nnue_has_enough_data(FILE *file) {
    if (fseek(file, 0, SEEK_END) != 0) {
        return false;
    }

    const long file_size = ftell(file);
    if (file_size < 0) {
        return false;
    }

    if (fseek(file, 0, SEEK_SET) != 0) {
        return false;
    }

    return (size_t)file_size >= nnue_required_bytes();
}

bool nnue_load(const char *path) {
    if (path == NULL || *path == '\0') {
        return false;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return false;
    }

    if (!nnue_has_enough_data(file)) {
        fclose(file);
        return false;
    }

    const size_t acc_weight_count = NNUE_INPUT_SIZE * NNUE_HL_SIZE;
    const size_t acc_bias_count = NNUE_HL_SIZE;
    const size_t output_weight_count = NNUE_OUTPUT_BUCKETS * 2 * NNUE_HL_SIZE;
    const size_t output_bias_count = NNUE_OUTPUT_BUCKETS;

    const bool ok =
        fread(g_nnue_network.accumulator_weights, sizeof(int16_t), acc_weight_count, file) == acc_weight_count &&
        fread(g_nnue_network.accumulator_biases, sizeof(int16_t), acc_bias_count, file) == acc_bias_count &&
        fread(g_nnue_network.output_weights, sizeof(int16_t), output_weight_count, file) == output_weight_count &&
        fread(g_nnue_network.output_bias, sizeof(int16_t), output_bias_count, file) == output_bias_count;

    fclose(file);

    if (!ok) {
        return false;
    }

    snprintf(g_nnue_path, sizeof(g_nnue_path), "%s", path);
    g_nnue_loaded = true;
    return true;
}

bool nnue_is_loaded(void) {
    return g_nnue_loaded;
}

const char *nnue_current_path(void) {
    return g_nnue_loaded ? g_nnue_path : "";
}

static bool try_load_candidate(const char *path) {
    return nnue_load(path);
}

static void dirname_from_path(const char *path, char *buffer, size_t buffer_size) {
    if (buffer_size == 0) {
        return;
    }

    buffer[0] = '\0';
    if (path == NULL || *path == '\0') {
        return;
    }

    snprintf(buffer, buffer_size, "%s", path);

    char *last_forward = strrchr(buffer, '/');
    char *last_backward = strrchr(buffer, '\\');
    char *last_sep = last_forward;
    if (last_backward != NULL && (last_sep == NULL || last_backward > last_sep)) {
        last_sep = last_backward;
    }

    if (last_sep == NULL) {
        buffer[0] = '\0';
        return;
    }

    *last_sep = '\0';
}

static bool try_relative_to_dir(const char *dir, const char *suffix) {
    if (dir == NULL || *dir == '\0' || suffix == NULL || *suffix == '\0') {
        return false;
    }

    char candidate[NNUE_PATH_MAX];
    snprintf(candidate, sizeof(candidate), "%s/%s", dir, suffix);
    return try_load_candidate(candidate);
}

void nnue_init(const char *argv0) {
#if !defined(_MSC_VER) || defined(__clang__)
    if (nnue_load_from_memory(gEVALData, gEVALSize, "<embedded>")) {
        printf("info string loaded embedded NNUE from %s\n", EVALFILE);
        return;
    }
#endif

    if (try_load_candidate(EVALFILE)) {
        printf("info string loaded NNUE from %s\n", g_nnue_path);
        return;
    }

    const char *env_path = getenv("POTENTIAL_NNUE_FILE");
    if (env_path != NULL && try_load_candidate(env_path)) {
        printf("info string loaded NNUE from %s\n", g_nnue_path);
        return;
    }

    static const char *cwd_candidates[] = {
        "nnue.bin",
        "../nnue.bin",
        "../../nnue.bin"
    };

    for (size_t i = 0; i < sizeof(cwd_candidates) / sizeof(cwd_candidates[0]); ++i) {
        if (try_load_candidate(cwd_candidates[i])) {
            printf("info string loaded NNUE from %s\n", g_nnue_path);
            return;
        }
    }

    char exe_dir[NNUE_PATH_MAX];
    dirname_from_path(argv0, exe_dir, sizeof(exe_dir));

    if (try_relative_to_dir(exe_dir, "nnue.bin") ||
        try_relative_to_dir(exe_dir, "../nnue.bin") ||
        try_relative_to_dir(exe_dir, "../../nnue.bin")) {
        printf("info string loaded NNUE from %s\n", g_nnue_path);
        return;
    }

    printf("info string NNUE file not found, using classical evaluation\n");
}

static int potential_square_to_eleanor_square(int square) {
    return square ^ 56;
}

static int piece_type_index(int piece) {
    return piece % 6;
}

static int calculate_feature_index(int perspective, int side, int piece_type, int square, bool mirrored) {
    if (perspective == black) {
        side ^= 1;
        square ^= 56;
    }

    if (mirrored) {
        square ^= 7;
    }

    return side * 64 * 6 + piece_type * 64 + square;
}

static int feature_index(const board *position, int perspective, int side, int piece, int square) {
    const bool mirrored = perspective == white ? position->nnue.mirroredWhite : position->nnue.mirroredBlack;
    return calculate_feature_index(perspective, side, piece_type_index(piece), potential_square_to_eleanor_square(square), mirrored);
}

#if defined(__AVX2__)
static inline void add_feature_row(int16_t *accumulator, const int16_t *weights) {
    for (int i = 0; i < NNUE_HL_SIZE; i += 16) {
        const __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        const __m256i row = _mm256_loadu_si256((const __m256i *)(weights + i));
        const __m256i sum = _mm256_add_epi16(acc, row);
        _mm256_storeu_si256((__m256i *)(accumulator + i), sum);
    }
}

static inline void add_sub_feature_rows(int16_t *accumulator, const int16_t *add_weights, const int16_t *sub_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; i += 16) {
        const __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        const __m256i add = _mm256_loadu_si256((const __m256i *)(add_weights + i));
        const __m256i sub = _mm256_loadu_si256((const __m256i *)(sub_weights + i));
        const __m256i sum = _mm256_sub_epi16(_mm256_add_epi16(acc, add), sub);
        _mm256_storeu_si256((__m256i *)(accumulator + i), sum);
    }
}

static inline void add_sub_sub_feature_rows(int16_t *accumulator, const int16_t *add_weights,
                                            const int16_t *sub1_weights, const int16_t *sub2_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; i += 16) {
        const __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        const __m256i add = _mm256_loadu_si256((const __m256i *)(add_weights + i));
        const __m256i sub1 = _mm256_loadu_si256((const __m256i *)(sub1_weights + i));
        const __m256i sub2 = _mm256_loadu_si256((const __m256i *)(sub2_weights + i));
        const __m256i sum = _mm256_sub_epi16(_mm256_sub_epi16(_mm256_add_epi16(acc, add), sub1), sub2);
        _mm256_storeu_si256((__m256i *)(accumulator + i), sum);
    }
}

static inline void add_add_sub_sub_feature_rows(int16_t *accumulator, const int16_t *add1_weights, const int16_t *add2_weights,
                                                const int16_t *sub1_weights, const int16_t *sub2_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; i += 16) {
        const __m256i acc = _mm256_loadu_si256((const __m256i *)(accumulator + i));
        const __m256i add1 = _mm256_loadu_si256((const __m256i *)(add1_weights + i));
        const __m256i add2 = _mm256_loadu_si256((const __m256i *)(add2_weights + i));
        const __m256i sub1 = _mm256_loadu_si256((const __m256i *)(sub1_weights + i));
        const __m256i sub2 = _mm256_loadu_si256((const __m256i *)(sub2_weights + i));
        const __m256i sum = _mm256_sub_epi16(
            _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(acc, add1), add2), sub1),
            sub2);
        _mm256_storeu_si256((__m256i *)(accumulator + i), sum);
    }
}
#else
static inline void add_feature_row(int16_t *accumulator, const int16_t *weights) {
    for (int i = 0; i < NNUE_HL_SIZE; ++i) {
        accumulator[i] = (int16_t)(accumulator[i] + weights[i]);
    }
}

static inline void add_sub_feature_rows(int16_t *accumulator, const int16_t *add_weights, const int16_t *sub_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; ++i) {
        accumulator[i] = (int16_t)(accumulator[i] + add_weights[i] - sub_weights[i]);
    }
}

static inline void add_sub_sub_feature_rows(int16_t *accumulator, const int16_t *add_weights,
                                            const int16_t *sub1_weights, const int16_t *sub2_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; ++i) {
        accumulator[i] = (int16_t)(accumulator[i] + add_weights[i] - sub1_weights[i] - sub2_weights[i]);
    }
}

static inline void add_add_sub_sub_feature_rows(int16_t *accumulator, const int16_t *add1_weights, const int16_t *add2_weights,
                                                const int16_t *sub1_weights, const int16_t *sub2_weights) {
    for (int i = 0; i < NNUE_HL_SIZE; ++i) {
        accumulator[i] = (int16_t)(accumulator[i] + add1_weights[i] + add2_weights[i] - sub1_weights[i] - sub2_weights[i]);
    }
}
#endif

void nnue_refresh_accumulators(board *position) {
    if (!g_nnue_loaded) {
        return;
    }

    const int white_king_square = position->bitboards[K] ? getLS1BIndex(position->bitboards[K]) : e1;
    const int black_king_square = position->bitboards[k] ? getLS1BIndex(position->bitboards[k]) : e8;
    position->nnue.mirroredWhite = (white_king_square % 8) > 3;
    position->nnue.mirroredBlack = (black_king_square % 8) > 3;

    memcpy(position->nnue.white, g_nnue_network.accumulator_biases, sizeof(g_nnue_network.accumulator_biases));
    memcpy(position->nnue.black, g_nnue_network.accumulator_biases, sizeof(g_nnue_network.accumulator_biases));

    for (int piece = P; piece <= k; ++piece) {
        U64 bitboard = position->bitboards[piece];
        const int side = pieceColor(piece);

        while (bitboard) {
            const int square = getLS1BIndex(bitboard);
            const int white_input = feature_index(position, white, side, piece, square);
            const int black_input = feature_index(position, black, side, piece, square);

            add_feature_row(position->nnue.white, &g_nnue_network.accumulator_weights[white_input * NNUE_HL_SIZE]);
            add_feature_row(position->nnue.black, &g_nnue_network.accumulator_weights[black_input * NNUE_HL_SIZE]);

            popBit(bitboard, square);
        }
    }
}

void nnue_acc_add_sub(board *position, int stm, int add_square, int add_piece, int sub_square, int sub_piece) {
    if (!g_nnue_loaded) {
        return;
    }

    const int addW = feature_index(position, white, stm, add_piece, add_square);
    const int addB = feature_index(position, black, stm, add_piece, add_square);
    const int subW = feature_index(position, white, stm, sub_piece, sub_square);
    const int subB = feature_index(position, black, stm, sub_piece, sub_square);

    add_sub_feature_rows(position->nnue.white,
                         &g_nnue_network.accumulator_weights[addW * NNUE_HL_SIZE],
                         &g_nnue_network.accumulator_weights[subW * NNUE_HL_SIZE]);
    add_sub_feature_rows(position->nnue.black,
                         &g_nnue_network.accumulator_weights[addB * NNUE_HL_SIZE],
                         &g_nnue_network.accumulator_weights[subB * NNUE_HL_SIZE]);
}

void nnue_acc_add_sub_sub(board *position, int stm, int add_square, int add_piece,
                          int sub1_square, int sub1_piece, int sub2_square, int sub2_piece) {
    if (!g_nnue_loaded) {
        return;
    }

    const int addW = feature_index(position, white, stm, add_piece, add_square);
    const int addB = feature_index(position, black, stm, add_piece, add_square);
    const int sub1W = feature_index(position, white, stm, sub1_piece, sub1_square);
    const int sub1B = feature_index(position, black, stm, sub1_piece, sub1_square);
    const int sub2W = feature_index(position, white, stm ^ 1, sub2_piece, sub2_square);
    const int sub2B = feature_index(position, black, stm ^ 1, sub2_piece, sub2_square);

    add_sub_sub_feature_rows(position->nnue.white,
                             &g_nnue_network.accumulator_weights[addW * NNUE_HL_SIZE],
                             &g_nnue_network.accumulator_weights[sub1W * NNUE_HL_SIZE],
                             &g_nnue_network.accumulator_weights[sub2W * NNUE_HL_SIZE]);
    add_sub_sub_feature_rows(position->nnue.black,
                             &g_nnue_network.accumulator_weights[addB * NNUE_HL_SIZE],
                             &g_nnue_network.accumulator_weights[sub1B * NNUE_HL_SIZE],
                             &g_nnue_network.accumulator_weights[sub2B * NNUE_HL_SIZE]);
}

void nnue_acc_add_add_sub_sub(board *position, int stm,
                              int add1_square, int add1_piece, int add2_square, int add2_piece,
                              int sub1_square, int sub1_piece, int sub2_square, int sub2_piece) {
    if (!g_nnue_loaded) {
        return;
    }

    const int add1W = feature_index(position, white, stm, add1_piece, add1_square);
    const int add1B = feature_index(position, black, stm, add1_piece, add1_square);
    const int add2W = feature_index(position, white, stm, add2_piece, add2_square);
    const int add2B = feature_index(position, black, stm, add2_piece, add2_square);
    const int sub1W = feature_index(position, white, stm, sub1_piece, sub1_square);
    const int sub1B = feature_index(position, black, stm, sub1_piece, sub1_square);
    const int sub2W = feature_index(position, white, stm, sub2_piece, sub2_square);
    const int sub2B = feature_index(position, black, stm, sub2_piece, sub2_square);

    add_add_sub_sub_feature_rows(position->nnue.white,
                                 &g_nnue_network.accumulator_weights[add1W * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[add2W * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[sub1W * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[sub2W * NNUE_HL_SIZE]);
    add_add_sub_sub_feature_rows(position->nnue.black,
                                 &g_nnue_network.accumulator_weights[add1B * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[add2B * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[sub1B * NNUE_HL_SIZE],
                                 &g_nnue_network.accumulator_weights[sub2B * NNUE_HL_SIZE]);
}

#if defined(__AVX2__)
static int reduce_epi32(__m256i vec) {
    __m128i hi = _mm256_extracti128_si256(vec, 1);
    __m128i lo = _mm256_castsi256_si128(vec);
    lo = _mm_add_epi32(lo, hi);
    hi = _mm_shuffle_epi32(lo, 0xEE);
    lo = _mm_add_epi32(lo, hi);
    hi = _mm_shuffle_epi32(lo, 0x55);
    lo = _mm_add_epi32(lo, hi);
    return _mm_cvtsi128_si32(lo);
}

static int32_t screlu_forward_avx2(const int16_t *stm_acc, const int16_t *nstm_acc, int bucket) {
    const __m256i vec_qa = _mm256_set1_epi16(NNUE_QA);
    const __m256i vec_zero = _mm256_setzero_si256();
    __m256i acc = _mm256_setzero_si256();

    for (int i = 0; i < NNUE_HL_SIZE; i += 16) {
        const __m256i stm_values = _mm256_loadu_si256((const __m256i *)(stm_acc + i));
        const __m256i nstm_values = _mm256_loadu_si256((const __m256i *)(nstm_acc + i));

        const __m256i stm_clamped = _mm256_min_epi16(vec_qa, _mm256_max_epi16(stm_values, vec_zero));
        const __m256i nstm_clamped = _mm256_min_epi16(vec_qa, _mm256_max_epi16(nstm_values, vec_zero));

        const __m256i stm_weights = _mm256_loadu_si256((const __m256i *)(g_nnue_network.output_weights[bucket] + i));
        const __m256i nstm_weights = _mm256_loadu_si256((const __m256i *)(g_nnue_network.output_weights[bucket] + NNUE_HL_SIZE + i));

        const __m256i stm_mul = _mm256_mullo_epi16(stm_clamped, stm_weights);
        const __m256i nstm_mul = _mm256_mullo_epi16(nstm_clamped, nstm_weights);

        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(stm_clamped, stm_mul));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(nstm_clamped, nstm_mul));
    }

    return reduce_epi32(acc);
}
#endif

#if !defined(__AVX2__)
static int32_t screlu_forward_scalar(const int16_t *stm_acc, const int16_t *nstm_acc, int bucket) {
    int32_t sum = 0;

    for (int i = 0; i < NNUE_HL_SIZE; ++i) {
        int32_t stm = stm_acc[i];
        int32_t nstm = nstm_acc[i];

        if (stm < 0) {
            stm = 0;
        } else if (stm > NNUE_QA) {
            stm = NNUE_QA;
        }

        if (nstm < 0) {
            nstm = 0;
        } else if (nstm > NNUE_QA) {
            nstm = NNUE_QA;
        }

        sum += stm * stm * g_nnue_network.output_weights[bucket][i];
        sum += nstm * nstm * g_nnue_network.output_weights[bucket][NNUE_HL_SIZE + i];
    }

    return sum;
}
#endif

int nnue_evaluate(const board *position) {
    const int piece_count = countBits(position->occupancies[both]);
    int bucket = (piece_count - 2) / (32 / NNUE_OUTPUT_BUCKETS);
    if (bucket < 0) {
        bucket = 0;
    } else if (bucket >= NNUE_OUTPUT_BUCKETS) {
        bucket = NNUE_OUTPUT_BUCKETS - 1;
    }

    const int16_t *stm_acc = position->side == white ? position->nnue.white : position->nnue.black;
    const int16_t *nstm_acc = position->side == white ? position->nnue.black : position->nnue.white;

    int64_t eval = 0;

#if defined(__AVX2__)
    eval = screlu_forward_avx2(stm_acc, nstm_acc, bucket);
#else
    eval = screlu_forward_scalar(stm_acc, nstm_acc, bucket);
#endif

    eval /= NNUE_QA;
    eval += g_nnue_network.output_bias[bucket];
    eval = (eval * NNUE_SCALE) / (NNUE_QA * NNUE_QB);

    const int material_scale = 2048 +
        90 * countBits(position->bitboards[N] | position->bitboards[n]) +
        90 * countBits(position->bitboards[B] | position->bitboards[b]) +
        180 * countBits(position->bitboards[R] | position->bitboards[r]) +
        360 * countBits(position->bitboards[Q] | position->bitboards[q]);

    eval = (eval * material_scale) / 4096;

    if (eval > mateValue - maxPly) {
        eval = mateValue - maxPly;
    } else if (eval < -mateValue + maxPly) {
        eval = -mateValue + maxPly;
    }

    return (int)eval;
}
