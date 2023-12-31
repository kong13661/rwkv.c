#define FLOAT32 0
#define INT8 1
#define INT_MACHINE 2
#define SIMPLE_MATMUL_NDIM 4 // shouldn't be changed
#include <stdlib.h>

typedef struct {
    int ndim;
    int *shape;
    int dtype;
    void *data;
    int *strides;
} ndarray;

int get_nelement(int ndim, int *shape);
void make_strides(ndarray *a);
ndarray *reshape(ndarray *a, int new_ndim, int *new_shape);
int *append_shape(int ndim, int *shape, int append_dim, int append_value);
void free_view(ndarray *a);
void free_ndarray(ndarray *a);
void squeeze_neg_dim(ndarray *a);
void check_mul_dim(ndarray *a, ndarray *b);
void check_element_op_dim(ndarray *a, ndarray *b);
ndarray **mat_broadcast_reshape(ndarray *a, ndarray *b);
ndarray **mat_element_op_reshape(ndarray *a, ndarray *b);
ndarray **matmul_reshape(ndarray *a, ndarray *b);
ndarray *matminus(ndarray *a, ndarray *b);
ndarray *matadd(ndarray *a, ndarray *b);
ndarray *matdot(ndarray *a, ndarray *b);
ndarray *matmul(ndarray *a, ndarray *b);
ndarray *softmax(ndarray *a);
ndarray *relu(ndarray *a);
ndarray *group_norm(ndarray *a, int num_groups, ndarray *ln_w, ndarray *ln_b);
ndarray *sigmoid(ndarray *a);
ndarray *square(ndarray *a);
ndarray *layer_norm(ndarray *a, int *normalized_shape, int normalized_shape_len, ndarray *ln_w, ndarray *ln_b);
ndarray *ndarray_index(ndarray *a, ndarray *idx);
size_t sizeof_dtype(int dtype);