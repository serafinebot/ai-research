#include "tensor.h"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

    t->data = (float *) malloc(t->nelem * sizeof(*t->data));
    if (t->data == NULL) {
        free(t->shape);
        free(t);
        return NULL;
    }

    return t;
}

tensor_t *fill(uint8_t ndim, uint16_t *shape, float value) {
    tensor_t *t = tensor_alloc(ndim, shape);
    if (t != NULL && t->data != NULL) memset(t->data, value, t->nelem * sizeof(*t->data));
    return t;
}

void tensor_free(tensor_t *t) {
    if (t == NULL) return;
    if (t->shape != NULL) free(t->shape);
    if (t->data != NULL) free(t->data);
    free(t);
}

bool max(tensor_t *t, float *m) {
    if (m == NULL || t == NULL || t->shape == NULL || t->data == NULL || t->nelem < 1) return false;
    *m = t->data[0];
    for (uint32_t i = 1; i < t->nelem; i++) if (t->data[i] > *m) *m = t->data[i];
    return true;
}

bool min(tensor_t *t, float *m) {
    if (m == NULL || t == NULL || t->shape == NULL || t->data == NULL || t->nelem < 1) return false;
    *m = t->data[0];
    for (uint32_t i = 1; i < t->nelem; i++) if (t->data[i] < *m) *m = t->data[i];
    return true;
}

static uint8_t int_digits(double a) {
    uint8_t n = (uint8_t) fabs(a);
    if (n == 0) return 1;
    return (uint8_t) log10(n) + 1;
}

static bool has_decimals(double x) {
    double intpart;
    double frac = modf(x, &intpart);
    return frac != 0.0;
}

void print(tensor_t *t) {
    if (t == NULL || t->shape == NULL || t->data == NULL) return;
    
    // shape multiples (how many elements fit in each dimension)
    uint16_t *mul = (uint16_t *) malloc(t->ndim * sizeof(*t->shape));
    if (mul == NULL) return;
    uint8_t ndim = t->ndim;
    mul[ndim-1] = t->shape[ndim-1];
    for (int8_t i = ndim-2; i >= 0; i--) mul[i] = mul[i+1] * t->shape[i];

    // shape modulus for current index (shape[i] = idx % shape[i])
    // no need to save the last dimension's modulus as the loop steps by its value (will always be 0)
    uint16_t *mod = (uint16_t *) malloc((t->ndim - 1) * sizeof(*t->shape));
    if (mod == NULL) {
        free(mul);
        return;
    }

    float maxel = 0;
    if (!max(t, &maxel)) return;
    uint8_t ndigits = int_digits(maxel);
    bool decimals = false;
    for (uint32_t i = 0; i < t->nelem && !decimals; i++) decimals = has_decimals(t->data[i]);
    char *fmt = "%*g";
    if (decimals) {
        fmt = "%*.4f";
        ndigits += 5;
    }

    for (uint32_t i = 0; i < t->nelem;) {
        for (uint8_t dim = 0; dim < ndim-1; dim++) {
            mod[dim] = i % mul[dim];
            if (mod[dim] == 0) {
                for (uint8_t j = 0; j < ndim-dim-1 && i > 0; j++) printf("\n");
                break;
            }
        }

        for (uint8_t j = 0; j < ndim-1; j++) printf(mod[j] == 0 ? "[" : " ");
        printf("[");
        for (uint16_t j = 0; j < mul[ndim-1]; j++) {
            printf(fmt, ndigits, t->data[i++]);
            if (j < mul[ndim-1] - 1) printf(" ");
        }
        for (uint8_t j = 0; j < ndim-1; j++) if (mod[j] == mul[j] - mul[ndim-1]) printf("]");
        printf("]\n");
    }

    free(mul);
    free(mod);
}

void shape_print(FILE *stream, uint8_t ndim, uint16_t *shape) {
    if (shape == NULL) return;
    fprintf(stream, "(");
    for (uint8_t i = 0; i < ndim; i++) {
        fprintf(stream, "%d", shape[i]);
        if (i < ndim - 1) fprintf(stream, ", ");
    }
    fprintf(stream, ")");
}

uint8_t broadcast(tensor_t *a, uint16_t **ashape, tensor_t *b, uint16_t **bshape) {
    if (a == NULL || a->shape == NULL || b == NULL || b->shape == NULL) return 0;

    uint8_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    uint8_t offa = ndim - a->ndim, offb = ndim - b->ndim;

    uint16_t *_ashape = (uint16_t *) malloc(ndim * sizeof(*a->shape));
    if (_ashape == NULL) return 0;
    uint16_t *_bshape = (uint16_t *) malloc(ndim * sizeof(*b->shape));
    if (_bshape == NULL) {
        free(_ashape);
        return 0;
    }

    for (uint8_t i = 0; i < ndim; i++) {
        uint16_t sa = i < offa ? 1 : a->shape[i - offa];
        uint16_t sb = i < offb ? 1 : b->shape[i - offb];
        if (sa != sb && sa != 1 && sb != 1) {
            free(_ashape);
            free(_bshape);
            return 0;
        }
        _ashape[i] = sa;
        _bshape[i] = sb;
    }

    if (ashape != NULL) *ashape = _ashape;
    else free(_ashape);
    if (bshape != NULL) *bshape = _bshape;
    else free(_bshape);

    return ndim;
}

tensor_t *squeeze(tensor_t *t, uint8_t dim) {
    if (t == NULL || t->shape == NULL) return NULL;
    if (t->shape[dim] != 1) return NULL;
    if (dim >= t->ndim || t->ndim < 2) return t;
    for (uint8_t i = dim; i < t->ndim-1; i++) t->shape[i] = t->shape[i+1];
    t->ndim--;
    return t;
}

tensor_t *unsqueeze(tensor_t *t, uint8_t dim) {
    if (t == NULL || t->shape == NULL) return NULL;
    if (dim > t->ndim) return t;
    uint8_t ndim = t->ndim + 1;
    uint16_t *shape = (uint16_t *) malloc(ndim * sizeof(*shape));
    if (shape == NULL) return NULL;
    for (uint8_t i = 0; i < dim; i++) shape[i] = t->shape[i];
    shape[dim] = 1;
    for (uint8_t i = dim; i < t->ndim; i++) shape[i+1] = t->shape[i];
    t->ndim = ndim;
    free(t->shape);
    t->shape = shape;
    return t;
}

// TODO: transpose implementation
tensor_t *transpose(tensor_t *t, uint8_t dim1, uint8_t dim2) {
    return t;
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

tensor_t *sumall(tensor_t *t) {
    if (t == NULL || t->shape == NULL || t->data == NULL) return NULL;
    tensor_t *r = tensor_alloc(1, (uint16_t[]){1});
    if (r == NULL) return NULL;
    float acc = 0;
    for (uint32_t i = 0; i < t->nelem; i++) acc += t->data[i];
    *r->data = acc;
    return r;
}

tensor_t *sum(tensor_t *t, int16_t dim, bool keepdim) {
    if (t == NULL || t->shape == NULL || t->data == NULL) return NULL;
    if (dim < -t->ndim || t->ndim <= dim) return NULL;
    uint8_t d = dim >= 0 ? dim : dim + t->ndim;

    printf("dim = %d -> %d\n", dim, d);

    printf("t.shape = ");
    shape_print(stdout, t->ndim, t->shape);
    printf("\n");

    uint16_t *shape = (uint16_t *) malloc(t->ndim * sizeof(*shape));
    if (shape == NULL) return NULL;
    memcpy(shape, t->shape, t->ndim * sizeof(*shape));
    shape[d] = 1;
    tensor_t *r = tensor_alloc(t->ndim, shape);
    free(shape);
    if (r == NULL) return NULL;

    printf("r.shape = ");
    shape_print(stdout, r->ndim, r->shape);
    printf("\n");

    uint32_t step = 1;
    for (uint8_t i = d+1; i < t->ndim; i++) step *= t->shape[i];
    printf("step = %d\n", step);
    uint32_t offset = step * t->shape[d];
    printf("offset = %d\n", offset);

    /*
        t.shape = (3, 3, 4)
        t.data = 
            [[[ 0  1  2  3]
              [ 4  5  6  7]
              [ 8  9 10 11]]

             [[12 13 14 15]
              [16 17 18 19]
              [20 21 22 23]]

             [[24 25 26 27]
              [28 29 30 31]
              [32 33 34 35]]]

        t.sum(0, keepdim=True)
            [[[36 39 42 45]
              [48 51 54 57]
              [60 63 66 69]]]
     */

    if (step == 1) {
        for (uint32_t i = 0; i < r->nelem; i++) {
            printf("r[%3d] = ", i);
            float acc = 0;
            for (uint8_t j = 0; j < t->shape[d]; j++) {
                printf("t[%3d]", i * t->shape[d] + j);
                acc += t->data[i * t->shape[d] + j];
                if (j < t->shape[d] - 1) printf(" + ");
            }
            printf("\n");
            r->data[i] = acc;
        }
    } else {
        for (uint32_t i = 0; i < r->nelem; i++) {
            uint32_t off = i / t->shape[t->ndim-1] * offset;
            printf("r[%3d] = ", i);
            uint16_t idx = off + i % t->shape[t->ndim-1];
            float acc = 0;
            for (uint8_t j = 0; j < t->shape[d]; j++) {
                printf("t[%3d]", idx);
                acc += t->data[idx];
                idx += step;
                if (j < t->shape[d] - 1) printf(" + ");
            }
            printf("\n");
            r->data[i] = acc;
        }
    }

    if (!keepdim) {
        if (squeeze(r, d) == NULL) {
            tensor_free(r);
            return NULL;
        }
    }

    return r;
}

// element wise operation
static tensor_t *ewop(tensor_t *a, tensor_t *b, tensor_op_t op) {
    if (a == NULL || b == NULL) return NULL;

    uint16_t *ashape, *bshape;
    uint8_t ndim = broadcast(a, &ashape, b, &bshape);
    if (ndim == 0) return NULL;

    uint16_t *cshape = (uint16_t *) malloc(ndim * sizeof(*cshape));
    if (cshape == NULL) return NULL;
    uint32_t nelem = ndim > 0 ? 1 : 0;
    for (uint8_t i = 0; i < ndim; i++) {
        cshape[i] = MAX(ashape[i], bshape[i]);
        nelem *= cshape[i];
    }
    tensor_t *c = tensor_alloc(ndim, cshape);
    if (c == NULL) {
        free(cshape);
        return NULL;
    }

    for (uint32_t cidx = 0; cidx < c->nelem; cidx++) {
        // TODO: optimize this, do not use modulus operator
        float aval = a->data[cidx % a->nelem];
        float bval = b->data[cidx % b->nelem];
        switch (op) {
            case OP_ADD: {
                c->data[cidx] = aval + bval;
                break;
            }
            case OP_MUL: {
                c->data[cidx] = aval * bval;
                break;
            }
            default: {
                free(cshape);
                tensor_free(c);
                return NULL;
            }
        }
    }

    free(cshape);

    return c;
}

tensor_t *add(tensor_t *a, tensor_t *b) {
    return ewop(a, b, OP_ADD);
}

tensor_t *mul(tensor_t *a, tensor_t *b) {
    return ewop(a, b, OP_MUL);
}

tensor_t *dot(tensor_t *a, tensor_t *b) {
    if (a == NULL || b == NULL || (a->ndim < 2 && b->ndim < 2)) return NULL;

    printf("a.shape: ");
    shape_print(stdout, a->ndim, a->shape);
    printf("\n");

    unsqueeze(b, b->ndim-2);
    printf("b.shape: ");
    shape_print(stdout, b->ndim, b->shape);
    printf("\n");

    uint16_t *ashape, *bshape;
    b->ndim--; // last dim of a must match second to last dim of b
    uint8_t ndim = broadcast(a, &ashape, b, &bshape);
    b->ndim++;
    printf("ndim: %d\n", ndim);
    if (ndim == 0) return NULL;

    printf("ashape: ");
    shape_print(stdout, ndim, ashape);
    printf("\n");
    printf("bshape: ");
    shape_print(stdout, ndim, bshape);
    printf("\n");

    uint16_t *cshape = (uint16_t *) malloc(ndim * sizeof(*cshape));
    if (cshape == NULL) return NULL;
    uint32_t nelem = 0;
    for (uint8_t i = 0; i < ndim; i++) {
        cshape[i] = MAX(ashape[i], bshape[i]);
        nelem = (nelem > 0 ? nelem : 1) * cshape[i];
    }
    cshape[ndim-1] = b->shape[b->ndim-1];
    tensor_t *c = tensor_alloc(ndim, cshape);
    if (c == NULL) {
        free(ashape);
        free(bshape);
        free(cshape);
        return NULL;
    }
    printf("cshape: ");
    shape_print(stdout, ndim, cshape);
    printf("\n");

    printf("a:\n");
    print(a);
    printf("\n");

    printf("b:\n");
    print(b);
    printf("\n");

    uint32_t aoff = 0, boff = 0, coff = 0;
    uint32_t astep = ashape[a->ndim-1] * ashape[a->ndim-2];
    uint32_t bstep = bshape[b->ndim-1] * bshape[b->ndim-2];
    uint32_t cstep = cshape[c->ndim-1] * cshape[c->ndim-2];
    uint32_t nitr = c->nelem / cstep;

    // coff += cstep;
    nitr = 1;

    /**
     * a.shape = (2, 2, 3)
     * b.shape = (3, 2)
     * c.shape = (2, 2, 2)
     * 
     * c[0,0,0] = a[0,0] * b[:,0]
     * 
     * c[0,0,0] = a[0,0,0]*b[0,0] + a[0,0,1]*b[1,0] + a[0,0,2]*b[2,0]
     * c[0,0,1] = a[0,0,0]*b[0,1] + a[0,0,1]*b[1,1] + a[0,0,2]*b[2,1]
     * c[0,1,0] = a[0,1,0]*b[0,0] + a[0,1,1]*b[1,0] + a[0,1,2]*b[2,0]
     * c[0,1,1] = a[0,1,0]*b[0,1] + a[0,1,1]*b[1,1] + a[0,1,2]*b[2,1]
     * c[1,0,0] = a[1,0,0]*b[0,0] + a[1,0,1]*b[1,0] + a[1,0,2]*b[2,0]
     * c[1,0,1] = a[1,0,0]*b[0,1] + a[1,0,1]*b[1,1] + a[1,0,2]*b[2,1]
     * c[1,1,0] = a[1,1,0]*b[0,0] + a[1,1,1]*b[1,0] + a[1,1,2]*b[2,0]
     * c[1,1,1] = a[1,1,0]*b[0,1] + a[1,1,1]*b[1,1] + a[1,1,2]*b[2,1]
    */

    uint8_t din = ashape[a->ndim-2], dmid = bshape[b->ndim-2], dout = cshape[c->ndim-1];
    for (uint16_t itr = 0; itr < nitr; itr++) {
        for (uint8_t i = 0; i < din; i++) {
            for (uint8_t o = 0; o < dout; o++) {

                float acc = 0;
                printf("c[%d] = ", i * dout + o + coff);
                for (uint8_t m = 0; m < dmid; m++) {
                    printf("%.4f * %.4f", a->data[i * dmid + m + aoff], b->data[m * dout + o + boff]);
                    acc += a->data[i * dmid + m + aoff] * b->data[m * dout + o + boff];
                    if (m < dmid - 1) printf(" + ");
                }
                printf("\n");
                c->data[i * dout + o + coff] = acc;

            }
            // coff += cstep;
        }
        // coff += cstep;
        // boff = (boff + bstep) % b->nelem;
        // aoff = (aoff + astep) % a->nelem;
    }

    printf("c:\n");
    print(c);
    printf("\n");

    free(ashape);
    free(bshape);
    free(cshape);

    return NULL;
}