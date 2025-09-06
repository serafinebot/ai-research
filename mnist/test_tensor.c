#include "tensor.h"
#include <assert.h>

void test_sum() {
    tensor_t *t = tensor_alloc(3, (uint16_t[]){3, 3, 4});
    assert(t != NULL);
    for (uint16_t i = 0; i < t->nelem; i++) t->data[i] = i;

    tensor_t *r = sum(t, 0, true);
    assert(r != NULL);

    printf("t:\n");
    print(t);

    printf("r:\n");
    print(r);

    tensor_free(t);
    tensor_free(r);
}

void test_add() {
    tensor_t *t1 = tensor_alloc(2, (uint16_t[]){3,4});
    assert(t1 != NULL && t1->shape != NULL && t1->data != NULL);
    tensor_t *t2 = tensor_alloc(3, (uint16_t[]){2,3,4});
    assert(t2 != NULL && t2->shape != NULL && t2->data != NULL);

    for (uint16_t i = 0; i < t1->nelem; i++) t1->data[i] = i;
    for (uint16_t i = 0; i < t2->nelem; i++) t2->data[i] = i;

    printf("t1\n");
    print(t1);
    printf("\n");

    printf("t2\n");
    print(t2);
    printf("\n");

    tensor_t *t3 = add(t1, t2);
    assert(t3 != NULL);

    printf("t3\n");
    print(t3);
    printf("\n");

    tensor_free(t1);
    tensor_free(t2);
}

// void test_dot() {
//     // tensor_t *t1 = tensor_alloc(4, (uint16_t[]){3,1,2,4});
//     // tensor_t *t2 = tensor_alloc(3, (uint16_t[]){2,4,2});

//     // tensor_t *t1 = tensor_alloc(2, (uint16_t[]){2, 2});
//     // tensor_t *t2 = tensor_alloc(2, (uint16_t[]){2, 2});

//     tensor_t *t1 = tensor_alloc(3, (uint16_t[]){2, 2, 3});
//     tensor_t *t2 = tensor_alloc(2, (uint16_t[]){3, 2});

//     // tensor_t *t1 = tensor_alloc(2, (uint16_t[]){2, 2});
//     // tensor_t *t2 = tensor_alloc(2, (uint16_t[]){2, 2});
//     for (uint16_t i = 0; i < t1->nelem; i++) t1->data[i] = i + 1;
//     for (uint16_t i = 0; i < t2->nelem; i++) t2->data[i] = t2->nelem - i;

//     dot(t1, t2);
// }


int main(int argc, char **argv) {
    test_sum();
    // test_add();
    // test_dot();

    return 0;
}