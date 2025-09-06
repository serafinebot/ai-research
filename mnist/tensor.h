#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

typedef struct
{
    uint8_t ndim;
    uint16_t *shape;
    uint32_t nelem;
    float *data;
} tensor_t;

typedef enum
{
    OP_ADD,
    OP_MUL
} tensor_op_t;

tensor_t *tensor_alloc(uint8_t ndim, uint16_t *shape);
tensor_t *fill(uint8_t ndim, uint16_t *shape, float value);
void tensor_free(tensor_t *t);
void shape_print(FILE *stream, uint8_t ndim, uint16_t *shape);
void print(tensor_t *t);
uint8_t broadcast(tensor_t *a, uint16_t **ashape, tensor_t *b, uint16_t **bshape);
tensor_t *reshape(tensor_t *t, uint8_t ndim, uint16_t *shape);
tensor_t *squeeze(tensor_t *t, uint8_t dim);
tensor_t *unsqueeze(tensor_t *t, uint8_t dim);
tensor_t *transpose(tensor_t *t, uint8_t dim1, uint8_t dim2);
tensor_t *sumall(tensor_t *t);
tensor_t *sum(tensor_t *t, int16_t dim, bool keepdim);
tensor_t *add(tensor_t *a, tensor_t *b);
tensor_t *mul(tensor_t *a, tensor_t *b);
tensor_t *dot(tensor_t *a, tensor_t *b);
bool max(tensor_t *t, float *m);
bool min(tensor_t *t, float *m);

#endif