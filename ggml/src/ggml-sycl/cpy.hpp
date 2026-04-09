#ifndef GGML_SYCL_CPY_HPP
#define GGML_SYCL_CPY_HPP

#include "common.hpp"
#include <float.h>

typedef void (*cpy_kernel_t)(const char * cx, char * cdst);

__dpct_inline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) {
        return 0;
    }
    if (x >= val[n - 1]) {
        return n - 1;
    }
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < val[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

inline void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q8_0 *  dsti = (block_q8_0 *) cdsti;

    float amax = 0.0f;  // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = xi[j];
        amax          = sycl::fmax(amax, sycl::fabs((float) v));
    }

    const float d  = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = xi[j] * id;

        dsti->qs[j] = sycl::round((float) x0);
    }
}

inline void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q4_0 *  dsti = (block_q4_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK4_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 8.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 8.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

inline void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q4_1 *  dsti = (block_q4_1 *) cdsti;

    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = xi[j];

        vmin = sycl::min(v, vmin);
        vmax = sycl::max(v, vmax);
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = vmin;

    for (int j = 0; j < QK4_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - vmin) * id;
        const float x1 = (xi[QK4_1 / 2 + j] - vmin) * id;

        const uint8_t xi0 = dpct::min(15, (int8_t) (x0 + 0.5f));
        const uint8_t xi1 = dpct::min(15, (int8_t) (x1 + 0.5f));

        dsti->qs[j] = xi0;
        dsti->qs[j] |= xi1 << 4;
    }
}

inline void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q5_0 *  dsti = (block_q5_0 *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; ++j) {
        const float x0 = xi[0 + j] * id;
        const float x1 = xi[QK5_0 / 2 + j] * id;

        const uint8_t xi0 = dpct::min(31, (int8_t) (x0 + 16.5f));
        const uint8_t xi1 = dpct::min(31, (int8_t) (x1 + 16.5f));

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0 / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

inline void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    const float * xi   = (const float *) cxi;
    block_q5_1 *  dsti = (block_q5_1 *) cdsti;

    float min = xi[0];
    float max = xi[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = xi[j];
        min           = v < min ? v : min;
        max           = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f / d : 0.0f;

    dsti->dm.x() = d;
    dsti->dm.y() = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1 / 2; ++j) {
        const float x0 = (xi[0 + j] - min) * id;
        const float x1 = (xi[QK5_1 / 2 + j] - min) * id;

        const uint8_t xi0 = (uint8_t) (x0 + 0.5f);
        const uint8_t xi1 = (uint8_t) (x1 + 0.5f);

        dsti->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1 / 2);
    }
    memcpy(dsti->qh, &qh, sizeof(qh));
}

inline void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    const float *  xi   = (const float *) cxi;
    block_iq4_nl * dsti = (block_iq4_nl *) cdsti;

    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = xi[j];
        if (amax < sycl::fabs((float) v)) {
            amax = sycl::fabs((float) v);
            vmax = v;
        }
    }

    float       d  = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f / d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL / 2; ++j) {
        const float   x0  = xi[0 + j] * id;
        const float   x1  = xi[QK4_NL / 2 + j] * id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        dsti->qs[j]       = xi0 | (xi1 << 4);
        const float v0    = kvalues_iq4nl[xi0];
        const float v1    = kvalues_iq4nl[xi1];
        const float w0    = xi[0 + j] * xi[0 + j];
        const float w1    = xi[QK4_NL / 2 + j] * xi[QK4_NL / 2 + j];
        sumqx += w0 * v0 * xi[j] + w1 * v1 * xi[QK4_NL / 2 + j];
        sumq2 += w0 * v0 * v0 + w1 * v1 * v1;
    }

    dsti->d = sumq2 > 0 ? sumqx / sumq2 : d;
}

// ---------------------------------------------------------------------
// TurboQuant TQ3_0 block helpers (shared by cpy.cpp and set_rows.cpp)
// ---------------------------------------------------------------------
static constexpr float TQ3_INV_SQRT32    = 0.17677669529663688f;
static constexpr float TQ3_SCALE_DIVISOR = 2.1573f;

inline void tq3_wht32_butterfly(float * x) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step * 2) {
            for (int j = i; j < i + step; ++j) {
                const float a = x[j];
                const float b = x[j + step];
                x[j]        = a + b;
                x[j + step] = a - b;
            }
        }
    }
}

inline void cpy_blck_f32_tq3_0(const char * cxi, char * cdsti) {
    const float * xi = (const float *) cxi;
    block_tq3_0 * dst = (block_tq3_0 *) cdsti;

    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };
    const float thresholds[7] = {
        -1.7455f, -1.0385f, -0.4931f, 0.0f, 0.4931f, 1.0385f, 1.7455f
    };

    float rot[32];
    for (int j = 0; j < 32; ++j) {
        rot[j] = xi[j] * (float) signs[j];
    }
    tq3_wht32_butterfly(rot);
    for (int j = 0; j < 32; ++j) {
        rot[j] *= TQ3_INV_SQRT32;
    }

    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float a = sycl::fabs(rot[j]);
        if (a > amax) amax = a;
    }
    const float d  = amax / TQ3_SCALE_DIVISOR;
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

    uint8_t qs[8] = {0,0,0,0,0,0,0,0};
    uint8_t qr[4] = {0,0,0,0};
    for (int j = 0; j < 32; ++j) {
        const float xn = rot[j] * id;
        int idx;
        if      (xn < thresholds[0]) idx = 0;
        else if (xn < thresholds[1]) idx = 1;
        else if (xn < thresholds[2]) idx = 2;
        else if (xn < thresholds[3]) idx = 3;
        else if (xn < thresholds[4]) idx = 4;
        else if (xn < thresholds[5]) idx = 5;
        else if (xn < thresholds[6]) idx = 6;
        else                          idx = 7;

        qs[j / 4] |= (uint8_t)((idx & 3) << (2 * (j % 4)));
        qr[j / 8] |= (uint8_t)(((idx >> 2) & 1) << (j % 8));
    }

    for (int j = 0; j < 8; ++j) dst->qs[j] = qs[j];
    for (int j = 0; j < 4; ++j) dst->qr[j] = qr[j];
    dst->gamma = sycl::vec<float, 1>(d).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

inline void cpy_blck_tq3_0_f32(const char * cxi, char * cdsti) {
    const block_tq3_0 * src = (const block_tq3_0 *) cxi;
    float * dst = (float *) cdsti;

    const float centroids[8] = {
        -2.1573f, -1.3336f, -0.7434f, -0.2428f,
         0.2428f,  0.7434f,  1.3336f,  2.1573f
    };
    const int8_t signs[32] = {
        +1, -1, +1, +1, -1, -1, +1, -1, +1, +1, -1, +1, -1, +1, -1, -1,
        +1, -1, -1, +1, +1, -1, +1, -1, -1, +1, +1, +1, -1, -1, +1, -1
    };

    const float d = sycl::vec<sycl::half, 1>{src->gamma}
                        .convert<float, sycl::rounding_mode::automatic>()[0];

    float rot[32];
    for (int j = 0; j < 32; ++j) {
        const int low2 = (src->qs[j / 4] >> (2 * (j % 4))) & 3;
        const int hi1  = (src->qr[j / 8] >> (j % 8)) & 1;
        const int idx  = low2 | (hi1 << 2);
        rot[j] = d * centroids[idx];
    }

    tq3_wht32_butterfly(rot);
    for (int j = 0; j < 32; ++j) {
        dst[j] = rot[j] * TQ3_INV_SQRT32 * (float) signs[j];
    }
}

void ggml_sycl_cpy(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
void ggml_sycl_dup(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

#endif  // GGML_SYCL_CPY_HPP
