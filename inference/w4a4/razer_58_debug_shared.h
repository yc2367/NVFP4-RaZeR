#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>

// Keep these identical to razer_58.cu
constexpr int VIEW_R = 4;
constexpr int VIEW_C = 4;

static inline float decode_fp4_e2m1(uint8_t nibble) {
    nibble &= 0xF;
    int sign = (nibble >> 3) & 0x1;
    uint8_t mag_code = nibble & 0x7;

    const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    float v = lut[mag_code];
    return sign ? -v : v;
}

static inline float decode_ue4m3(uint8_t u) {
    u &= 0x7F;
    int e = (u >> 3) & 0xF;
    int m = u & 0x7;
    constexpr int bias = 7;

    if (e == 0) {
        if (m == 0) return 0.0f;
        float frac = static_cast<float>(m) / 8.0f;
        return ldexpf(frac, 1 - bias);
    }
    if (e == 0xF) return NAN;

    float frac = 1.0f + static_cast<float>(m) / 8.0f;
    return ldexpf(frac, e - bias);
}

// Print formatting exactly as razer_58.cu
static inline void print_matrix_window(const char* name,
                                       const float* M_,
                                       int rows, int cols,
                                       int start_r, int start_c,
                                       int view_r = VIEW_R,
                                       int view_c = VIEW_C) {
    int R = (view_r < rows) ? view_r : rows;
    int C = (view_c < cols) ? view_c : cols;
    if (start_r + R > rows) start_r = (rows >= R) ? (rows - R) : 0;
    if (start_c + C > cols) start_c = (cols >= C) ? (cols - C) : 0;

    printf("%s (window %dx%d at [%d,%d]) =\n", name, R, C, start_r, start_c);
    for (int r = 0; r < R; ++r) {
        printf("  ");
        for (int c = 0; c < C; ++c) {
            float v = M_[(start_r + r) * cols + (start_c + c)];
            if (std::isnan(v))      printf("%12s ", "NaN");
            else if (std::isinf(v)) printf("%12s ", (v > 0) ? "Inf" : "-Inf");
            else {
                float av = fabsf(v);
                if (av >= 10000000.0f || (av != 0.0f && av < 0.001f))
                    printf("%12.3e ", (double)v);
                else
                    printf("%12.3f ", (double)v);
            }
        }
        printf("\n");
    }
    printf("\n");
}
