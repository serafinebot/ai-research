#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdlib.h>

typedef struct {
    uint8_t ndim;
    uint16_t *shape;
    uint32_t nelem;
    uint32_t *data;
} mat_t;

mat_t *mat_alloc(unsigned int ndim, ...) {
    mat_t *mat = (mat_t *) malloc(sizeof(mat_t));
    if (mat == NULL) return NULL;
    mat->ndim = ndim;
    mat->nelem = 0;

    va_list ap;
    va_start(ap, ndim);
    mat->shape = (uint16_t *) malloc(ndim * sizeof(*mat->shape));
    if (mat->shape == NULL) {
        free(mat);
        va_end(ap);
        return NULL;
    }
    for (uint8_t i = 0; i < ndim; i++) {
        mat->shape[i] = (uint16_t) va_arg(ap, unsigned int);
        mat->nelem = mat->nelem == 0 ? mat->shape[i] : mat->nelem * mat->shape[i];
    }
    va_end(ap);

    mat->data = (uint32_t *) malloc(mat->nelem * sizeof(*mat->data));

    return mat;
}

void mat_free(mat_t *mat) {
    if (mat == NULL) return;
    if (mat->shape != NULL) free(mat->shape);
    if (mat->data != NULL) free(mat->data);
    free(mat);
}

void mat_print(mat_t *mat) {
    if (mat == NULL || mat->shape == NULL || mat->data == NULL) return;

    uint16_t *shape = (uint16_t *) malloc(mat->ndim * sizeof(*mat->shape));
    uint16_t *mod = (uint16_t *) malloc(mat->ndim * sizeof(*mat->shape));
    if (shape == NULL) return;
    uint8_t ndim = mat->ndim;
    shape[ndim-1] = mat->shape[ndim-1];
    for (int8_t i = ndim-2; i >= 0; i--) shape[i] = shape[i+1] * mat->shape[i];

    for (uint32_t i = 0; i < mat->nelem;) {
        // there's no need to calculate the last dimension's modulus as the loop steps by its value (always 0)
        // TODO: technically there's no need to do the modulus (expensive), we could just decrease each value by step size each iteration
        for (uint8_t dim = 0; dim < ndim-1; dim++) mod[dim] = i % shape[dim];

        for (uint8_t dim = 0; dim < ndim-1; dim++) {
            if (mod[dim] == 0) {
                for (uint8_t j = 0; j < ndim-dim-1 && i > 0; j++) printf("\n");
                break;
            }
        }

        for (uint8_t j = 0; j < ndim-1; j++) printf(mod[j] == 0 ? "[" : " ");
        printf("[");
        for (uint16_t j = 0; j < shape[ndim-1]; j++) {
            printf("%3d", mat->data[i++]);
            if (j < shape[ndim-1] - 1) printf(" ");
        }
        for (uint8_t j = 0; j < ndim-1; j++) if (mod[j] == shape[j] - shape[ndim-1]) printf("]");
        printf("]\n");
    }

    free(shape);
}

int main(int argc, char **argv) {
    mat_t *m = mat_alloc(4, 2, 3, 4, 2);
    if (m == NULL) return 1;
    for (uint16_t i = 0; i < m->nelem; i++) m->data[i] = i + 1;
    mat_print(m);
    mat_free(m);
    return 0;
}