#include <stdio.h>
#include <stdint.h>

void dot(uint32_t *a, uint32_t *b, uint32_t *c, const uint16_t din, const uint16_t dmid, const uint16_t dout) {
    for (uint16_t i = 0; i < din; i++) {
        for (uint16_t o = 0; o < dout; o++) {
            uint32_t acc = 0;
            for (uint16_t m = 0; m < dmid; m++) acc += a[i * dmid + m] * b[m * dout + o];
            c[i * dout + o] = acc;
        }
    }
}

void sum(uint32_t *a, uint32_t *b, uint32_t *c, const uint16_t h, const uint16_t w) {
    for (uint16_t i = 0; i < h; i++) {
        for (uint16_t j = 0; j < w; j++) {
            uint16_t k = i * w + j;
            c[k] = a[k] + b[k];
        }
    }
}

void pmat(uint32_t *a, const uint16_t h, const uint16_t w) {
    for (uint16_t i = 0; i < h; i++) {
        for (uint16_t j = 0; j < w; j++) printf("%4d", a[i * w + j]);
        printf("\n");
    }
}

int main(int argc, char **argv) {
    uint32_t a[] = {
        1, 2,
        3, 4,
        3, 2
    };
    uint32_t b[] = {
        5, 6, 7, 8,
        7, 6, 5, 4
    };
    uint32_t c[] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };
    uint32_t d[] = {
        11, 0, 0, 4,
        0, 3, 0, 0,
        1, 0, 0, 32
    };
    uint32_t e[] = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    pmat(a, 3, 2);
    printf("\n");
    pmat(b, 2, 4);
    printf("\n");

    dot(a, b, c, 3, 2, 4);
    pmat(c, 3, 4);
    printf("\n");

    sum(c, d, e, 3, 4);
    pmat(e, 3, 4);
    printf("\n");

    return 0;
}