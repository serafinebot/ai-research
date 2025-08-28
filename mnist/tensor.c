#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint8_t ndim;
    uint16_t *shape;
    uint32_t nelem;
    uint32_t *data;
} tensor_t;

tensor_t *tensor_alloc(uint8_t ndim, uint16_t *shape) {
    if (shape == NULL) return NULL;

    tensor_t *t = (tensor_t *) malloc(sizeof(tensor_t));
    if (t == NULL) return NULL;

    t->ndim = ndim;
    t->shape = (uint16_t *) malloc(ndim * sizeof(*t->shape));
    if (t->shape == NULL) {
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(*t->shape));

    t->nelem = 0;
    for (uint8_t i = 0; i < ndim; i++) t->nelem = (t->nelem > 0 ? t->nelem : 1) * t->shape[i];

    t->data = (uint32_t *) malloc(t->nelem * sizeof(*t->data));
    if (t->data == NULL) {
        free(t->shape);
        free(t);
        return NULL;
    }
    return t;
}

void tensor_free(tensor_t *t) {
    if (t == NULL) return;
    if (t->shape != NULL) free(t->shape);
    if (t->data != NULL) free(t->data);
    free(t);
}

void tensor_shape_print(FILE *stream, tensor_t *t) {
    if (t == NULL || t->shape == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < t->ndim; i++) {
        fprintf(stream, "%d", t->shape[i]);
        if (i < t->ndim - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}

uint8_t tensor_shape_eq(tensor_t *a, tensor_t *b) {
    if (a->ndim != b->ndim) return 0;
    for (uint8_t i = 0; i < a->ndim; i++) if (a->shape[i] != b->shape[i]) return 0;
    return 1;
}

uint8_t tensor_shape_eq_check(tensor_t *a, tensor_t *b) {
    uint8_t eq = tensor_shape_eq(a, b);
    if (!eq) {
        fprintf(stderr, "shape mismatch");
        tensor_shape_print(stderr, a);
        fprintf(stderr, " != ");
        tensor_shape_print(stderr, b);
        fprintf(stderr, "\n");
    }
    return eq;
}

void tensor_print(tensor_t *t) {
    if (t == NULL || t->shape == NULL || t->data == NULL) return;

    uint16_t *shape = (uint16_t *) malloc(t->ndim * sizeof(*t->shape));
    uint16_t *mod = (uint16_t *) malloc(t->ndim * sizeof(*t->shape));
    if (shape == NULL) return;
    uint8_t ndim = t->ndim;
    shape[ndim-1] = t->shape[ndim-1];
    for (int8_t i = ndim-2; i >= 0; i--) shape[i] = shape[i+1] * t->shape[i];

    for (uint32_t i = 0; i < t->nelem;) {
        // there's no need to calculate the last dimension's modulus as the loop steps by its value (always 0)
        // TODO: there's no need to do the modulus (expensive), we could just decrease each dimension by step size each iteration and reset it when it reaches 0
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
            printf("%3d", t->data[i++]);
            if (j < shape[ndim-1] - 1) printf(" ");
        }
        for (uint8_t j = 0; j < ndim-1; j++) if (mod[j] == shape[j] - shape[ndim-1]) printf("]");
        printf("]\n");
    }

    free(shape);
    free(mod);
}

tensor_t *reshape(tensor_t *t, uint8_t ndim, uint16_t *shape) {
    if (t == NULL) return NULL;
    uint8_t nelem = 0;
    for (uint8_t i = 0; i < ndim; i++) nelem = (nelem > 0 ? nelem : 1) * shape[i];
    if (t->nelem != nelem) return NULL;
    t->ndim = ndim;
    memcpy(t->shape, shape, ndim * sizeof(*t->shape));
    return t;
}

tensor_t *add(tensor_t *a, tensor_t *b) {
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_shape_eq_check(a, b)) return NULL;
    tensor_t *c = tensor_alloc(a->ndim, a->shape);
    if (c == NULL) return NULL;
    for (uint32_t i = 0; i < a->nelem; i++) c->data[i] = a->data[i] + b->data[i];
    return c;
}

tensor_t *mul(tensor_t *a, tensor_t *b) {
    if (a == NULL || b == NULL) return NULL;
    if (!tensor_shape_eq_check(a, b)) return NULL;
    tensor_t *c = tensor_alloc(a->ndim, a->shape);
    if (c == NULL) return NULL;
    for (uint32_t i = 0; i < a->nelem; i++) c->data[i] = a->data[i] * b->data[i];
    return c;
}

int main(int argc, char **argv) {
    tensor_t *t1 = tensor_alloc(3, (uint16_t[]){2,3,4});
    tensor_t *t2 = tensor_alloc(3, (uint16_t[]){2,3,4});
    if (t1 == NULL || t2 == NULL) return 1;
    for (uint16_t i = 0; i < t1->nelem; i++) t1->data[i] = i + 1;
    for (uint16_t i = 0; i < t2->nelem; i++) t2->data[i] = t2->nelem - i;

    printf("t1\n");
    tensor_print(t1);
    printf("\n");

    printf("t2\n");
    tensor_print(t2);
    printf("\n");

    tensor_t *t3 = add(t1, t2);
    printf("t3\n");
    tensor_print(t3);
    printf("\n");

    t3 = mul(t1, t2);
    printf("t3\n");
    tensor_print(t3);
    printf("\n");

    reshape(t2, 2, (uint16_t[]){4, 6});
    printf("t2\n");
    tensor_print(t2);
    printf("\n");

    tensor_free(t1);
    tensor_free(t2);
    tensor_free(t3);

    return 0;
}