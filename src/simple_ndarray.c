#include "simple_ndarray.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define MATMUL_LOOP(type, a, b, new_array, op)                                                                        \
    new_array->data = calloc(get_nelement(new_array->ndim, new_array->shape), sizeof(type));                          \
    _Pragma("omp parallel for collapse(3) schedule(static) num_threads(8)")                                                      \ 
    for (int n = 0; n < abs(new_array->shape[0]); n++) {                                                              \
        for (int m = 0; m < abs(new_array->shape[1]); m++) {                                                          \
            for (int i = 0; i < abs(new_array->shape[2]); i++) {                                                      \
                int index_prefix = n * new_array->strides[0] + m * new_array->strides[1] + i * new_array->strides[2]; \
                int index_prefix_a = n * a->strides[0] + m * a->strides[1] + i * a->strides[2];                       \
                int index_prefix_b = n * b->strides[0] + m * b->strides[1];                                           \
                for (int k = 0; k < abs(a->shape[3]); k++) {                                                          \
                    int index_b = index_prefix_b + k * b->strides[2];                                                 \
                    for (int j = 0; j < abs(new_array->shape[3]); j++) {                                              \
                        ((float *)new_array->data)[index_prefix + j] +=                                               \
                            ((float *)a->data)[index_prefix_a + k] * ((float *)b->data)[index_b + j];                 \
                    }                                                                                                 \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

// _Pragma("omp parallel for collapse(3) schedule(dynamic, 1)")                                                      \ 


#define MAT_ELEMENT_OP_LOOP(type, a, b, new_array, op)                                                                \
    new_array->data = calloc(get_nelement(new_array->ndim, new_array->shape), sizeof(float));                         \
    for (int n = 0; n < abs(new_array->shape[0]); n++) {                                                              \
        for (int m = 0; m < abs(new_array->shape[1]); m++) {                                                          \
            for (int i = 0; i < abs(new_array->shape[2]); i++) {                                                      \
                int index_prefix = n * new_array->strides[0] + m * new_array->strides[1] + i * new_array->strides[2]; \
                int index_prefix_a = n * a->strides[0] + m * a->strides[1] + i * a->strides[2];                       \
                int index_prefix_b = n * b->strides[0] + m * b->strides[1] + i * b->strides[2];                       \
                for (int j = 0; j < abs(new_array->shape[3]); j++) {                                                  \
                    ((float *)new_array->data)[index_prefix + j * new_array->strides[3]] =                            \
                        ((float *)a->data)[index_prefix_a + j * a->strides[3]] op                                     \ 
                            ((float *)b->data)[index_prefix_b + j * b->strides[3]];                                   \
                }                                                                                                     \
            }                                                                                                         \
        }                                                                                                             \
    }

#define TYPE_OP_LOOP(a, b, new_array, op_loop, op) \
    if (a->dtype == FLOAT32) {                     \
        op_loop(float, a, b, new_array, op);       \
    } else if (a->dtype == INT8) {                 \
        op_loop(int, a, b, new_array, op);         \
    }

#define MAT_ELEMENT_OP_BLOCK(a, b, op)                          \
    ndarray *new_array = (ndarray *)malloc(sizeof(ndarray));    \
    ndarray **_ab = mat_element_op_reshape(a, b);               \
    a = _ab[0];                                                 \
    b = _ab[1];                                                 \
    free(_ab);                                                  \
                                                                \
    if (a->dtype != b->dtype) {                                 \
        printf("matmul error: dtype not match\n");              \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
    new_array->ndim = a->ndim;                                  \
    new_array->dtype = a->dtype;                                \
    new_array->shape = (int *)malloc(sizeof(int) * (a->ndim));  \
    for (int i = 0; i < a->ndim; i++) {                         \
        if (abs(a->shape[i]) == 1) {                            \
            if (abs(b->shape[i]) == 1) {                        \
                if (a->shape[i] == -1 || b->shape[i] == -1)     \
                    new_array->shape[i] = -1;                   \
                else                                            \
                    new_array->shape[i] = 1;                    \
            } else                                              \
                new_array->shape[i] = b->shape[i];              \
        } else                                                  \
            new_array->shape[i] = a->shape[i];                  \
    }                                                           \
    make_strides(new_array);                                    \
                                                                \
    if (a->dtype == FLOAT32) {                                  \
        TYPE_OP_LOOP(a, b, new_array, MAT_ELEMENT_OP_LOOP, op); \
    }                                                           \
    free_view(a);                                               \
    free_view(b);                                               \
    squeeze_neg_dim(new_array);                                 \
    return new_array;

size_t sizeof_dtype(int dtype) {
    switch (dtype) {
    case FLOAT32:
        return sizeof(float);
    case INT8:
        return sizeof(char);
    }
}

int get_nelement(int ndim, int *shape) {
    int element = 1;
    for (int i = 0; i < ndim; i++) {
        element = element * abs(shape[i]);
    }
    return element;
}

void make_strides(ndarray *a) {
    a->strides = (int *)malloc(sizeof(int) * a->ndim);
    a->strides[a->ndim - 1] = 1;
    for (int i = a->ndim - 2; i >= 0; i--) {
        a->strides[i] = abs(a->strides[i + 1] * a->shape[i + 1]);
        if (abs(a->shape[i + 1]) == 1) a->strides[i + 1] = 0;
    }
    if (abs(a->shape[0]) == 1) a->strides[0] = 0;
}

ndarray *softmax(ndarray *a) {
    if (a->dtype != FLOAT32) {
        printf("softmax error: dtype not match\n");
        exit(EXIT_FAILURE);
    }

    int *new_reshape = (int *)malloc(sizeof(int) * (2));
    new_reshape[0] = get_nelement(a->ndim, a->shape) / a->shape[a->ndim - 1];
    new_reshape[1] = a->shape[a->ndim - 1];
    ndarray *flatten_a = reshape(a, 2, new_reshape);
    flatten_a->data = (float *)malloc(sizeof(float) * get_nelement(flatten_a->ndim, flatten_a->shape));
    free(new_reshape);

    for (int i = 0; i < flatten_a->shape[0]; i++) {
        float sum = 0.0f;
        int index = i * flatten_a->strides[0];
        float max_val = ((float *)a->data)[index];
        for (int j = 1; j < flatten_a->shape[1]; j++) {
            if (((float *)a->data)[index + j] > max_val) {
                max_val = ((float *)a->data)[index + j];
            }
        }
        for (int j = 0; j < flatten_a->shape[1]; j++) {
            ((float *)flatten_a->data)[index + j] = expf(((float *)a->data)[index + j] - max_val);
            sum += ((float *)flatten_a->data)[index + j];
        }
        for (int j = 0; j < flatten_a->shape[1]; j++) {
            ((float *)flatten_a->data)[index + j] /= sum;
        }
    }
    ndarray *new_array = reshape(flatten_a, a->ndim, a->shape);
    free_view(flatten_a);
    return new_array;
}

ndarray *relu(ndarray *a) {
    if (a->dtype != FLOAT32) {
        printf("softmax error: dtype not match\n");
        exit(EXIT_FAILURE);
    }

    int *new_reshape = (int *)malloc(sizeof(int) * (1));
    new_reshape[0] = get_nelement(a->ndim, a->shape);
    ndarray *flatten_a = reshape(a, 1, new_reshape);
    flatten_a->data = (float *)malloc(sizeof(float) * get_nelement(flatten_a->ndim, flatten_a->shape));
    free(new_reshape);

    for (int i = 0; i < flatten_a->shape[0]; i++) {
        if (((float *)a->data)[i] < 0) {
            ((float *)flatten_a->data)[i] = 0;
        } else {
            ((float *)flatten_a->data)[i] = ((float *)a->data)[i];
        }
    }

    ndarray *new_array = reshape(flatten_a, a->ndim, a->shape);
    free_view(flatten_a);
    return new_array;
}

ndarray *ndarray_index(ndarray *a, ndarray *idx) {
    if (idx->dtype != INT_MACHINE) {
        printf("index error: idx->dtype != INT\n");
        exit(EXIT_FAILURE);
    }

    ndarray *new_array = (ndarray *)malloc(sizeof(ndarray));
    new_array->ndim = a->ndim + idx->ndim - 1;
    new_array->dtype = a->dtype;
    new_array->shape = (int *)malloc(sizeof(int) * (new_array->ndim));
    for (int i = 0; i < idx->ndim; i++) {
        new_array->shape[i] = idx->shape[i];
    }
    for (int i = 0; i < a->ndim - 1; i++) {
        new_array->shape[i + idx->ndim] = a->shape[i + 1];
    }
    new_array->data = (float *)malloc(sizeof(float) * get_nelement(new_array->ndim, new_array->shape));
    make_strides(new_array);

    int copy_line_num = get_nelement(a->ndim - 1, a->shape + 1);
    int head_num = get_nelement(idx->ndim, idx->shape);

    for (int i = 0; i < head_num; i++) {
        memcpy((float *)new_array->data + i * copy_line_num, (float *)a->data + ((int *)idx->data)[i] * copy_line_num, copy_line_num * sizeof(float));
    }
    return new_array;
}

ndarray *layer_norm(ndarray *a, int *normalized_shape, int normalized_shape_len, ndarray *ln_w, ndarray *ln_b) {
    if (a->dtype != FLOAT32) {
        printf("layer_norm error: dtype not match\n");
        exit(EXIT_FAILURE);
    }

    if (normalized_shape_len > a->ndim) {
        printf("layer_norm error: normalized_shape_len >= a->ndim\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < normalized_shape_len; i++) {
        if (normalized_shape[i] != a->shape[i + a->ndim - normalized_shape_len] ||
            normalized_shape[i] != ln_w->shape[i] ||
            normalized_shape[i] != ln_b->shape[i]) {
            printf("layer_norm error: normalized_shape[i] != a->shape[i + a->ndim - normalized_shape_len] ||\n"
                   "normalized_shape[i] != ln_w->shape[i] ||\n"
                   "normalized_shape[i] != ln_b->shape[i]\n");
            exit(EXIT_FAILURE);
        }
    }

    int new_reshape[2] = {get_nelement(a->ndim, a->shape) / get_nelement(normalized_shape_len, normalized_shape),
                          get_nelement(normalized_shape_len, normalized_shape)};
    int w_reshape = get_nelement(ln_w->ndim, ln_w->shape);
    ndarray *reshaped_a = reshape(a, 2, new_reshape);
    reshaped_a->data = (float *)malloc(sizeof(float) * get_nelement(reshaped_a->ndim, reshaped_a->shape));
    ndarray *reshaped_ln_w = reshape(ln_w, 1, &w_reshape);
    ndarray *reshaped_ln_b = reshape(ln_b, 1, &w_reshape);

    for (int i = 0; i < reshaped_a->shape[0]; i++) {
        int index_i = i * reshaped_a->strides[0];
        float mean = 0.0f, var = 0.0f;
        for (int j = 0; j < reshaped_a->shape[1]; j++) {
            int index_j = index_i + j * reshaped_a->strides[1];
            mean += ((float *)a->data)[index_j];
        }
        mean /= reshaped_a->shape[1];
        for (int j = 0; j < reshaped_a->shape[1]; j++) {
            int index_j = index_i + j * reshaped_a->strides[1];
            var += (((float *)a->data)[index_j] - mean) * (((float *)a->data)[index_j] - mean);
        }
        var /= reshaped_a->shape[1];
        for (int j = 0; j < reshaped_a->shape[1]; j++) {
            int index_j = index_i + j * reshaped_a->strides[1];
            ((float *)reshaped_a->data)[index_j] =
                (((float *)a->data)[index_j] - mean) / sqrtf(var + 1e-5f) * ((float *)reshaped_ln_w->data)[j] +
                ((float *)reshaped_ln_b->data)[j];
        }
    }

    ndarray *new_array = reshape(reshaped_a, a->ndim, a->shape);
    free_view(reshaped_a);
    free_view(reshaped_ln_w);
    free_view(reshaped_ln_b);
    return new_array;
}

ndarray *group_norm(ndarray *a, int num_groups, ndarray *ln_w, ndarray *ln_b) {
    if (a->dtype != FLOAT32) {
        printf("softmax error: dtype not match\n");
        exit(EXIT_FAILURE);
    }
    if (a->ndim < 2) {
        printf("group_norm error: ndim < 2\n");
        exit(EXIT_FAILURE);
    }

    if (ln_w->ndim != 1 || ln_b->ndim != 1) {
        printf("group_norm error: ln_w->ndim != 1 || ln_b->ndim != 1\n");
        exit(EXIT_FAILURE);
    }

    if (ln_w->shape[0] != a->shape[1] || ln_b->shape[0] != a->shape[1]) {
        printf("group_norm error: ln_w->shape[0] != a->shape[1] || ln_b->shape[0] != a->shape[1]\n");
        exit(EXIT_FAILURE);
    }

    if (a->shape[1] % num_groups != 0) {
        printf("group_norm error: shape[1] %% num_groups != 0\n");
        exit(EXIT_FAILURE);
    }

    int *new_reshape = (int *)malloc(sizeof(int) * (4));
    new_reshape[0] = a->shape[0];
    new_reshape[1] = num_groups;
    new_reshape[2] = a->shape[1] / num_groups;
    new_reshape[3] = get_nelement(a->ndim, a->shape) / (a->shape[0] * a->shape[1]);
    ndarray *flatten_a = reshape(a, 4, new_reshape);
    flatten_a->data = (float *)malloc(sizeof(float) * get_nelement(flatten_a->ndim, flatten_a->shape));
    free(new_reshape);

    for (int i = 0; i < flatten_a->shape[0]; i++) {
        int index_i = i * flatten_a->strides[0];
        for (int j = 0; j < flatten_a->shape[1]; j++) {
            int index_j = index_i + j * flatten_a->strides[1];
            float mean = 0.0f, var = 0.0f;
            for (int k = 0; k < flatten_a->shape[2]; k++) {
                int index_k = index_j + k * flatten_a->strides[2];
                for (int l = 0; l < flatten_a->shape[3]; l++) {
                    mean += ((float *)a->data)[index_k + l];
                }
            }
            mean /= flatten_a->shape[2] * flatten_a->shape[3];
            for (int k = 0; k < flatten_a->shape[2]; k++) {
                int index_k = index_j + k * flatten_a->strides[2];
                for (int l = 0; l < flatten_a->shape[3]; l++) {
                    var += (((float *)a->data)[index_k + l] - mean) * (((float *)a->data)[index_k + l] - mean);
                }
            }
            var /= flatten_a->shape[2] * flatten_a->shape[3];
            for (int k = 0; k < flatten_a->shape[2]; k++) {
                int index_k = index_j + k * flatten_a->strides[2];
                int index_ln = j * flatten_a->shape[2] + k;
                for (int l = 0; l < flatten_a->shape[3]; l++) {
                    ((float *)flatten_a->data)[index_k + l] =
                        (((float *)a->data)[index_k + l] - mean) / sqrtf(var + 1e-5f) * ((float *)ln_w->data)[index_ln] +
                        ((float *)ln_b->data)[index_ln];
                }
            }
        }
    }
    ndarray *new_array = reshape(flatten_a, a->ndim, a->shape);
    free_view(flatten_a);
    return new_array;
}

ndarray *sigmoid(ndarray *a) {
    if (a->dtype != FLOAT32) {
        printf("softmax error: dtype not match\n");
        exit(EXIT_FAILURE);
    }

    int *new_reshape = (int *)malloc(sizeof(int) * (1));
    new_reshape[0] = get_nelement(a->ndim, a->shape);
    ndarray *flatten_a = reshape(a, 1, new_reshape);
    flatten_a->data = (float *)malloc(sizeof(float) * get_nelement(flatten_a->ndim, flatten_a->shape));
    free(new_reshape);

    for (int i = 0; i < flatten_a->shape[0]; i++)
        ((float *)flatten_a->data)[i] = 1.0f / (1.0f + expf(-((float *)a->data)[i]));

    ndarray *new_array = reshape(flatten_a, a->ndim, a->shape);
    free_view(flatten_a);
    return new_array;
}

ndarray *square(ndarray *a) {
    if (a->dtype != FLOAT32) {
        printf("softmax error: dtype not match\n");
        exit(EXIT_FAILURE);
    }

    int *new_reshape = (int *)malloc(sizeof(int) * (1));
    new_reshape[0] = get_nelement(a->ndim, a->shape);
    ndarray *flatten_a = reshape(a, 1, new_reshape);
    flatten_a->data = (float *)malloc(sizeof(float) * get_nelement(flatten_a->ndim, flatten_a->shape));
    free(new_reshape);

    for (int i = 0; i < flatten_a->shape[0]; i++)
        ((float *)flatten_a->data)[i] = ((float *)a->data)[i] * ((float *)a->data)[i];

    ndarray *new_array = reshape(flatten_a, a->ndim, a->shape);
    free_view(flatten_a);
    return new_array;
}

ndarray *reshape(ndarray *a, int new_ndim, int *new_shape) {
    // return new ndarray view, share data with a
    ndarray *new_array = (ndarray *)malloc(sizeof(ndarray));
    *new_array = *a;

    int nelement_a = get_nelement(a->ndim, a->shape),
        new_element = get_nelement(new_ndim, new_shape);
    if (nelement_a != new_element) {
        printf("reshape error: element number not match\n");
        exit(EXIT_FAILURE);
    }
    new_array->ndim = new_ndim;
    new_array->shape = (int *)malloc(sizeof(int) * new_ndim);
    for (int i = 0; i < new_ndim; i++) {
        new_array->shape[i] = new_shape[i];
    }

    make_strides(new_array);
    return new_array;
}

int *append_shape(int ndim, int *shape, int append_dim, int append_value) {
    int *new_shape = (int *)malloc(sizeof(int) * (ndim + abs(append_dim)));
    if (append_dim == 0) {
        for (int i = 0; i < ndim; i++) {
            new_shape[i] = shape[i];
        }
        return new_shape;
    }
    if (append_dim < 0) {
        for (int i = 0; i < -append_dim; i++) {
            new_shape[i] = append_value;
        }
        for (int i = 0; i < ndim; i++) {
            new_shape[i - append_dim] = shape[i];
        }
    } else {
        for (int i = 0; i < ndim; i++) {
            new_shape[i] = shape[i];
        }
        for (int i = ndim; i < ndim + append_dim; i++) {
            new_shape[i] = append_value;
        }
    }
    return new_shape;
}

void free_view(ndarray *a) {
    free(a->shape);
    free(a->strides);
    free(a);
}

void free_ndarray(ndarray *a) {
    free(a->data);
    free_view(a);
}

void squeeze_neg_dim(ndarray *a) {
    int *new_shape = (int *)malloc(sizeof(int) * (a->ndim));
    int new_ndim = 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != -1) {
            new_shape[new_ndim] = a->shape[i];
            new_ndim++;
        }
    }
    free(a->shape);
    a->shape = new_shape;
    a->ndim = new_ndim;
}

void check_mul_dim(ndarray *a, ndarray *b) {
    for (int i = 0; i < a->ndim; i++) {
        if (a->ndim - i > 2) {
            if (a->shape[i] != b->shape[i] && abs(a->shape[i]) != 1 && abs(b->shape[i] != 1)) {
                printf("matmul_reshape error: shape not match\n");
                exit(EXIT_FAILURE);
            }
        } else {
            if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2]) {
                printf("matmul_reshape error: shape not match\n");
                exit(EXIT_FAILURE);
            }
            break;
        }
    }
}

void check_element_op_dim(ndarray *a, ndarray *b) {
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i] && abs(a->shape[i]) != 1 && abs(b->shape[i] != 1)) {
            printf("matmul_reshape error: shape not match\n");
            exit(EXIT_FAILURE);
        }
    }
}

ndarray **mat_broadcast_reshape(ndarray *a, ndarray *b) {

    int view_a = 0, view_b = 0;

    if (a->ndim < b->ndim) {
        int *_new_reshape = append_shape(a->ndim, a->shape, -(b->ndim - a->ndim), 1);
        ndarray *new_a = reshape(a, b->ndim, _new_reshape);
        free(_new_reshape);
        if (view_a == 1)
            free_view(a);
        a = new_a;
        view_a = 1;
    } else if (a->ndim > b->ndim) {
        int *_new_reshape = append_shape(b->ndim, b->shape, -(a->ndim - b->ndim), 1);
        ndarray *new_b = reshape(b, a->ndim, _new_reshape);
        free(_new_reshape);
        if (view_b == 1)
            free_view(b);
        b = new_b;
        view_b = 1;
    }

    if (a->ndim > SIMPLE_MATMUL_NDIM) {
        printf("matmul_reshape error: ndim > SIMPLE_MATMUL_NDIM\n");
        exit(EXIT_FAILURE);
    }

    if (a->ndim < SIMPLE_MATMUL_NDIM) {
        int *_new_reshape = append_shape(a->ndim, a->shape, -(SIMPLE_MATMUL_NDIM - a->ndim), -1);
        ndarray *new_a = reshape(a, SIMPLE_MATMUL_NDIM, _new_reshape);
        free(_new_reshape);
        if (view_a == 1)
            free_view(a);
        a = new_a;
        view_a = 1;
        _new_reshape = append_shape(b->ndim, b->shape, -(SIMPLE_MATMUL_NDIM - b->ndim), -1);
        ndarray *new_b = reshape(b, SIMPLE_MATMUL_NDIM, _new_reshape);
        free(_new_reshape);
        if (view_b == 1)
            free_view(b);
        b = new_b;
        view_b = 1;
    }

    if (!view_a) a = reshape(a, a->ndim, a->shape);
    if (!view_b) b = reshape(b, b->ndim, b->shape);
    ndarray **new_array = (ndarray **)malloc(sizeof(ndarray *) * 2);
    new_array[0] = a;
    new_array[1] = b;
    return new_array;
}

ndarray **mat_element_op_reshape(ndarray *a, ndarray *b) {
    ndarray **new_array = mat_broadcast_reshape(a, b);
    check_element_op_dim(new_array[0], new_array[1]);
    return new_array;
}

ndarray **matmul_reshape(ndarray *a, ndarray *b) {
    int view_a = 0, view_b = 0;
    if (a->ndim == 1) {
        int *_new_reshape = append_shape(1, a->shape, -1, -1);
        a = reshape(a, 2, _new_reshape);
        free(_new_reshape);
        view_a = 1;
    }
    if (b->ndim == 1) {
        int *_new_reshape = append_shape(1, b->shape, 1, -1);
        b = reshape(b, 2, _new_reshape);
        free(_new_reshape);
        view_b = 1;
    }
    ndarray **new_array = mat_broadcast_reshape(a, b);
    if (view_a) free_view(a);
    if (view_b) free_view(b);
    check_mul_dim(new_array[0], new_array[1]);
    return new_array;
}

ndarray *matminus(ndarray *a, ndarray *b) {
    MAT_ELEMENT_OP_BLOCK(a, b, -);
}

ndarray *matadd(ndarray *a, ndarray *b) {
    MAT_ELEMENT_OP_BLOCK(a, b, +);
}

ndarray *matdot(ndarray *a, ndarray *b) {
    MAT_ELEMENT_OP_BLOCK(a, b, *);
}

ndarray *matmul(ndarray *a, ndarray *b) {
    ndarray *new_array = (ndarray *)malloc(sizeof(ndarray));
    ndarray **_ab = matmul_reshape(a, b);
    a = _ab[0];
    b = _ab[1];
    free(_ab);

    if (a->dtype != b->dtype) {
        printf("matmul error: dtype not match\n");
        exit(EXIT_FAILURE);
    }
    new_array->ndim = a->ndim;
    new_array->dtype = a->dtype;
    new_array->shape = (int *)malloc(sizeof(int) * (a->ndim));

    for (int i = 0; i < a->ndim - 2; i++) {
        if (abs(a->shape[i]) == 1) {
            if (abs(b->shape[i]) == 1) {
                if (a->shape[i] == -1 || b->shape[i] == -1)
                    new_array->shape[i] = -1;
                else
                    new_array->shape[i] = 1;
            } else
                new_array->shape[i] = b->shape[i];
        } else
            new_array->shape[i] = a->shape[i];
    }

    new_array->shape[a->ndim - 2] = a->shape[b->ndim - 2];
    new_array->shape[a->ndim - 1] = b->shape[a->ndim - 1];
    make_strides(new_array);

    TYPE_OP_LOOP(a, b, new_array, MATMUL_LOOP, *);

    free_view(a);
    free_view(b);
    squeeze_neg_dim(new_array);
    return new_array;
}
