#include "simple_ndarray.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
// #include <omp.h>

#define TOP_P 0.85
#define TEMPERATURE 1.0
#define MAX_TOKEN_LEN 150
#define LENGTH_PER_TRIAL 100
#define END_OF_TEXT 0
#if defined _WIN32
    #include "win.h"
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

typedef struct {
    ndarray *ln0_weight;
    ndarray *ln0_bias;
    ndarray *ln1_weight;
    ndarray *ln1_bias;
    ndarray *ln2_weight;
    ndarray *ln2_bias;
    ndarray *att_time_mix_k;
    ndarray *att_time_mix_v;
    ndarray *att_time_mix_r;
    ndarray *att_time_decay;
    ndarray *att_time_first;
    ndarray *att_receptance_weight;
    ndarray *att_key_weight;
    ndarray *att_value_weight;
    ndarray *att_output_weight;
    ndarray *att_ln_x_weight;
    ndarray *att_ln_x_bias;
    ndarray *ffn_time_mix_k;
    ndarray *ffn_time_mix_r;
    ndarray *ffn_key_weight;
    ndarray *ffn_receptance_weight;
    ndarray *ffn_value_weight;
} Block;

typedef struct {
    int n_embd;
    int n_layer;
    int n_head;
    int head_size;
    Block *blocks;
    ndarray *state;
    ndarray *emb;
    ndarray *ln_out_weight;
    ndarray *ln_out_bias;
    ndarray *head_weight;
} RWKV5;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    TokenIndex *sorted_vocab;
    char **vocab;
    int vocab_size;
    char **table[256][256];
    int table_len[256][256];
    int wlen[256];
    int *good[256];
    int good_len[256];
} VocabTable;

void init_RWKV5(RWKV5 *model) {
    model->n_embd = 0;
    model->n_layer = 0;
    model->n_head = 0;
    model->head_size = 0;
    model->blocks = NULL;
    model->emb = NULL;
    model->head_weight = NULL;
    model->state = NULL;
    model->ln_out_weight = NULL;
    model->ln_out_bias = NULL;
}

void init_Block(Block *block) {
    block->ln0_weight = NULL;
    block->ln0_bias = NULL;
    block->ln1_weight = NULL;
    block->ln1_bias = NULL;
    block->ln2_weight = NULL;
    block->ln2_bias = NULL;
    block->att_time_mix_k = NULL;
    block->att_time_mix_v = NULL;
    block->att_time_mix_r = NULL;
    block->att_time_decay = NULL;
    block->att_time_first = NULL;
    block->att_receptance_weight = NULL;
    block->att_key_weight = NULL;
    block->att_value_weight = NULL;
    block->att_output_weight = NULL;
    block->att_ln_x_weight = NULL;
    block->att_ln_x_bias = NULL;
    block->ffn_time_mix_k = NULL;
    block->ffn_time_mix_r = NULL;
    block->ffn_key_weight = NULL;
    block->ffn_receptance_weight = NULL;
    block->ffn_value_weight = NULL;
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

int read_ndarray(void *data, int pos, ndarray **a_ref) {
    if (*a_ref == NULL) {
        *a_ref = (ndarray *)malloc(sizeof(ndarray));
    }
    ndarray *a = *a_ref;
    a->dtype = *(int *)(data + pos);
    pos += sizeof(int);
    a->ndim = *(int *)(data + pos);
    pos += sizeof(int);
    a->shape = (int *)(data + pos);
    pos += sizeof(int) * a->ndim;
    a->data = data + pos;
    pos += get_nelement(a->ndim, a->shape) * sizeof_dtype(a->dtype);
    return pos;
}

void **append(void **data, void *data_ptr, int *max_len, int *alloc) {
    if (*max_len == *alloc) {
        *max_len *= 2;
        data = realloc(data, sizeof(void *) * (*max_len));
    }
    data[(*alloc)++] = data_ptr;
    return data;
}

int parse_weights(void *data, int pos, RWKV5 *model) {
    int length;
    length = *(int *)(data + pos);
    pos += sizeof(int);
    char *key = calloc(length + 1, sizeof(char));
    strncpy(key, (char *)(data + pos), length);
    pos += length * sizeof(char);
    if (strcmp(key, "emb.weight") == 0) {
        pos = read_ndarray(data, pos, &model->emb);
    } else if (strcmp(key, "ln_out.weight") == 0) {
        pos = read_ndarray(data, pos, &model->ln_out_weight);
    } else if (strcmp(key, "ln_out.bias") == 0) {
        pos = read_ndarray(data, pos, &model->ln_out_bias);
    } else if (strcmp(key, "head.weight") == 0) {
        pos = read_ndarray(data, pos, &model->head_weight);
    } else {
        char *subkey;
        subkey = strtok(key, ".");
        subkey = strtok(NULL, ".");
        int layer = atoi(subkey);
        subkey = strtok(NULL, "\n");
        if (strcmp(subkey, "ln0.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln0_weight);
        } else if (strcmp(subkey, "ln0.bias") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln0_bias);
        } else if (strcmp(subkey, "ln1.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln1_weight);
        } else if (strcmp(subkey, "ln1.bias") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln1_bias);
        } else if (strcmp(subkey, "ln2.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln2_weight);
        } else if (strcmp(subkey, "ln2.bias") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ln2_bias);
        } else if (strcmp(subkey, "att.time_mix_k") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_time_mix_k);
        } else if (strcmp(subkey, "att.time_mix_v") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_time_mix_v);
        } else if (strcmp(subkey, "att.time_mix_r") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_time_mix_r);
        } else if (strcmp(subkey, "att.time_decay") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_time_decay);
        } else if (strcmp(subkey, "att.time_first") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_time_first);
        } else if (strcmp(subkey, "att.receptance.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_receptance_weight);
        } else if (strcmp(subkey, "att.key.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_key_weight);
        } else if (strcmp(subkey, "att.value.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_value_weight);
        } else if (strcmp(subkey, "att.output.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_output_weight);
        } else if (strcmp(subkey, "att.ln_x.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_ln_x_weight);
        } else if (strcmp(subkey, "att.ln_x.bias") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].att_ln_x_bias);
        } else if (strcmp(subkey, "ffn.time_mix_k") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ffn_time_mix_k);
        } else if (strcmp(subkey, "ffn.time_mix_r") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ffn_time_mix_r);
        } else if (strcmp(subkey, "ffn.key.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ffn_key_weight);
        } else if (strcmp(subkey, "ffn.receptance.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ffn_receptance_weight);
        } else if (strcmp(subkey, "ffn.value.weight") == 0) {
            pos = read_ndarray(data, pos, &model->blocks[layer].ffn_value_weight);
        }
    }
    return pos;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id + 1 : -1;
}

char *decode(int index, VocabTable *vocab_table) {
    return vocab_table->vocab[index - 1];
}

void read_metadata(char *checkpoint, RWKV5 *model) {

    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    fread(&model->n_embd, sizeof(int), 1, file);
    fread(&model->n_layer, sizeof(int), 1, file);
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fclose(file);
    int fd = open(checkpoint, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }

    model->blocks = (Block *)malloc(sizeof(Block) * model->n_embd);
    init_Block(model->blocks);

    int pos = 2 * sizeof(int);
    while (pos < file_size)
        pos = parse_weights(data, pos, model);
    model->n_head = model->blocks[0].att_time_decay->shape[0];
    model->head_size = model->blocks[0].ln1_weight->shape[0] / model->n_head;
}

void read_vocab(char *vocab_file, VocabTable *vocab) {
    FILE *file = fopen(vocab_file, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", vocab_file);
        exit(EXIT_FAILURE);
    }
    fread(&vocab->vocab_size, sizeof(int), 1, file);
    int token_len;
    char *token;
    vocab->vocab = (char **)malloc(sizeof(char *) * vocab->vocab_size);
    vocab->sorted_vocab = (TokenIndex *)malloc(sizeof(TokenIndex) * vocab->vocab_size);
    for (int i = 0; i < vocab->vocab_size; i++) {
        fread(&token_len, sizeof(int), 1, file);
        token = (char *)calloc(token_len + 1, sizeof(char));
        fread(token, sizeof(char), token_len, file);
        vocab->vocab[i] = token;
        vocab->sorted_vocab[i].str = token;
        vocab->sorted_vocab[i].id = i;
    }
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 256; j++) {
            fread(&vocab->table_len[i][j], sizeof(int), 1, file);
            vocab->table[i][j] = vocab->table_len[i][j] > 0 ? (char **)malloc(sizeof(char *) * vocab->table_len[i][j]) : NULL;
            for (int k = 0; k < vocab->table_len[i][j]; k++) {
                fread(&token_len, sizeof(int), 1, file);
                token = (char *)calloc(token_len + 1, sizeof(char));
                fread(token, sizeof(char), token_len, file);
                vocab->table[i][j][k] = token;
            }
        }
    }
    for (int i = 0; i < 256; i++) {
        fread(&vocab->wlen[i], sizeof(int), 1, file);
    }
    for (int i = 0; i < 256; i++) {
        fread(&vocab->good_len[i], sizeof(int), 1, file);
        if (vocab->good_len[i] > 0) {
            vocab->good[i] = (int *)malloc(sizeof(int *) * vocab->good_len[i]);
            fread(vocab->good[i], sizeof(int), vocab->good_len[i], file);
        }
    }
    qsort(vocab->sorted_vocab, vocab->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void encodeBytes(char *str, int str_len, VocabTable *vocab_table, int **encode, int *encode_len) {
    *encode = (int *)malloc(sizeof(int) * 256);
    int len_max = 256;
    *encode_len = 0;
    int i = 0;
    char *s;
    int s0, s1;
    int flag;
    char *sss;
    while (i < str_len) {
        s = (char *)calloc(2, sizeof(char));
        s[0] = str[i];
        if (i < str_len - 1) {
            s1 = (unsigned char)str[i + 1];
            s0 = (unsigned char)str[i];
            flag = 0;
            for (int j = 0; j < vocab_table->good_len[s0]; j++) {
                if (vocab_table->good[s0][j] == s1) {
                    flag = 1;
                    break;
                }
            }
            if (flag) {
                sss = (char *)calloc(vocab_table->wlen[s0] + 1, sizeof(char));
                strncpy(sss, str + i, vocab_table->wlen[s0]);
                for (int j = 0; j < vocab_table->table_len[s0][s1]; j++) {
                    if (strlen(sss) >= strlen(vocab_table->table[s0][s1][j]) &&
                        strncmp(sss, vocab_table->table[s0][s1][j],
                                strlen(vocab_table->table[s0][s1][j])) == 0) {
                        free(s);
                        s = vocab_table->table[s0][s1][j];
                        break;
                    }
                }
            }
        }
        int index = str_lookup(s, vocab_table->sorted_vocab, vocab_table->vocab_size);
        if (index == -1) {
            printf("Error: %s not in vocab\n", s);
            exit(EXIT_FAILURE);
        }
        if (*encode_len == len_max) {
            len_max *= 2;
            *encode = realloc(*encode, sizeof(int) * len_max);
        }
        (*encode)[(*encode_len)++] = index;
        i += s[0] == '\0' ? 1 : strlen(s); // specific for '\0'
    }
}

ndarray *channel_mixing(RWKV5 *model, int layer, ndarray *x) {
    ndarray *xk, *xv, *xr, *one, *r, *v, *k, *res, *state;
    state = model->state;
    res = x;
    x = layer_norm(x, &model->n_embd, 1, model->blocks[layer].ln2_weight, model->blocks[layer].ln2_bias);
    int i0 = (2 + model->head_size) * layer;
    ndarray *s_i0 = (ndarray *)malloc(sizeof(ndarray));
    s_i0->ndim = 1;
    s_i0->dtype = FLOAT32;
    s_i0->shape = (int *)malloc(sizeof(int) * s_i0->ndim);
    s_i0->shape[0] = state->shape[1];
    s_i0->data = (float *)state->data + i0 * state->shape[1];
    make_strides(s_i0);

    one = (ndarray *)malloc(sizeof(ndarray));
    one->ndim = 1;
    one->dtype = FLOAT32;
    one->shape = (int *)malloc(sizeof(int) * one->ndim);
    one->shape[0] = 1;
    one->data = (float *)malloc(sizeof(float));
    ((float *)one->data)[0] = 1.0;
    make_strides(one);

    ndarray *tmp1, *tmp2, *tmp;

    tmp1 = matdot(x, model->blocks[layer].ffn_time_mix_k);
    tmp2 = matminus(one, model->blocks[layer].ffn_time_mix_k);
    tmp = tmp2;
    tmp2 = matdot(tmp2, s_i0);
    free_ndarray(tmp);
    xk = matadd(tmp1, tmp2);
    free_ndarray(tmp1);
    free_ndarray(tmp2);

    tmp1 = matdot(x, model->blocks[layer].ffn_time_mix_r);
    tmp2 = matminus(one, model->blocks[layer].ffn_time_mix_r);
    tmp = tmp2;
    tmp2 = matdot(tmp2, s_i0);
    free_ndarray(tmp);
    xr = matadd(tmp1, tmp2);
    free_ndarray(tmp1);
    free_ndarray(tmp2);
    free_ndarray(one);

    for (int i = 0; i < s_i0->shape[0]; i++)
        ((float *)s_i0->data)[i] = ((float *)x->data)[i];
    free_view(s_i0);

    tmp = matmul(model->blocks[layer].ffn_receptance_weight, xr);
    r = sigmoid(tmp);
    free_ndarray(tmp);

    tmp = matmul(model->blocks[layer].ffn_key_weight, xk);
    tmp1 = relu(tmp);
    free_ndarray(tmp);
    k = square(tmp1);
    free_ndarray(tmp1);

    tmp = matmul(model->blocks[layer].ffn_value_weight, k);
    v = matdot(r, tmp);
    free_ndarray(tmp);

    tmp = x;
    x = matadd(v, res);
    free_ndarray(tmp);
    return x;
}

ndarray *time_mixing(RWKV5 *model, int layer, ndarray *x) {
    ndarray *tmp1, *tmp2, *tmp, *res, *state;
    res = x;
    state = model->state;
    x = layer_norm(x, &model->n_embd, 1, model->blocks[layer].ln1_weight, model->blocks[layer].ln1_bias);

    int H = model->n_head, S = model->head_size;
    int i1 = (2 + S) * layer + 1;
    ndarray *s_i1 = (ndarray *)malloc(sizeof(ndarray));
    s_i1->ndim = 1;
    s_i1->dtype = FLOAT32;
    s_i1->shape = (int *)malloc(sizeof(int) * s_i1->ndim);
    s_i1->shape[0] = state->shape[1];
    s_i1->data = (float *)state->data + i1 * state->shape[1];
    make_strides(s_i1);

    ndarray *xk, *xv, *xr, *one, *r, *v, *k;
    one = (ndarray *)malloc(sizeof(ndarray));
    one->ndim = 1;
    one->dtype = FLOAT32;
    one->shape = (int *)malloc(sizeof(int) * one->ndim);
    one->shape[0] = 1;
    one->data = (float *)malloc(sizeof(float));
    ((float *)one->data)[0] = 1.0;
    make_strides(one);

    tmp1 = matdot(x, model->blocks[layer].att_time_mix_k);
    tmp2 = matminus(one, model->blocks[layer].att_time_mix_k);
    tmp = tmp2;
    tmp2 = matdot(tmp2, s_i1);
    free_ndarray(tmp);
    xk = matadd(tmp1, tmp2);
    free_ndarray(tmp1);
    free_ndarray(tmp2);

    int reshape_k[3] = {H, S, 1};
    tmp = matmul(model->blocks[layer].att_key_weight, xk);
    k = reshape(tmp, 3, reshape_k);
    free_view(tmp);
    free_ndarray(xk);

    tmp1 = matdot(x, model->blocks[layer].att_time_mix_v);
    tmp2 = matminus(one, model->blocks[layer].att_time_mix_v);
    tmp = tmp2;
    tmp2 = matdot(tmp2, s_i1);
    free_ndarray(tmp);
    xv = matadd(tmp1, tmp2);
    free_ndarray(tmp1);
    free_ndarray(tmp2);

    int reshape_v[3] = {H, 1, S};
    tmp = matmul(model->blocks[layer].att_value_weight, xv);
    v = reshape(tmp, 3, reshape_v);
    free_view(tmp);
    free_ndarray(xv);

    tmp1 = matdot(x, model->blocks[layer].att_time_mix_r);
    tmp2 = matminus(one, model->blocks[layer].att_time_mix_r);
    tmp = tmp2;
    tmp2 = matdot(tmp2, s_i1);
    free_ndarray(tmp);
    xr = matadd(tmp1, tmp2);
    free_ndarray(tmp1);
    free_ndarray(tmp2);

    int reshape_r[3] = {H, 1, S};
    tmp = matmul(model->blocks[layer].att_receptance_weight, xr);
    r = reshape(tmp, 3, reshape_r);
    free_view(tmp);
    free_ndarray(xr);

    free_ndarray(one);

    for (int i = 0; i < s_i1->shape[0]; i++)
        ((float *)s_i1->data)[i] = ((float *)x->data)[i];
    free_view(s_i1);

    s_i1 = (ndarray *)malloc(sizeof(ndarray));
    s_i1->ndim = 3;
    s_i1->dtype = FLOAT32;
    s_i1->shape = (int *)malloc(sizeof(int) * s_i1->ndim);
    s_i1->shape[0] = H;
    s_i1->shape[1] = S;
    s_i1->shape[2] = S;
    s_i1->data = (float *)state->data + (i1 + 1) * state->shape[1];
    make_strides(s_i1);

    tmp = matdot(k, v);
    free_ndarray(k);
    free_ndarray(v);
    tmp1 = matdot(model->blocks[layer].att_time_first, tmp);
    tmp2 = matadd(tmp1, s_i1);
    free_ndarray(tmp1);
    x = matmul(r, tmp2);
    free_ndarray(tmp2);
    free_ndarray(r);
    tmp1 = matdot(model->blocks[layer].att_time_decay, s_i1);
    tmp2 = matadd(tmp1, tmp);
    free_ndarray(tmp1);
    free_ndarray(tmp);
    free_view(s_i1);
    s_i1 = tmp2;

    float *data = (float *)state->data + (i1 + 1) * state->shape[1];
    for (int i = 0; i < H * S * S; i++) {
        ((float *)data)[i] = ((float *)s_i1->data)[i];
    }
    free_ndarray(s_i1);

    tmp = x;
    int flatten[2] = {1, get_nelement(x->ndim, x->shape)};
    x = reshape(x, 2, flatten);
    free_view(tmp);

    tmp = x;
    x = group_norm(x, H, model->blocks[layer].att_ln_x_weight, model->blocks[layer].att_ln_x_bias);
    free(x->shape);
    x->ndim = 1;
    x->shape = (int *)malloc(sizeof(int) * x->ndim);
    x->shape[0] = H * S;

    free_ndarray(tmp);
    tmp = x;
    tmp1 = matmul(model->blocks[layer].att_output_weight, x);
    free_ndarray(tmp);

    x = matadd(tmp1, res);
    free_ndarray(tmp1);
    return x;
}

int compare_floats(const void *a, const void *b) {
    return *(float *)a < *(float *)b ? 1 : -1;
}

int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_logits(ndarray *out, float temperature, float top_p) {
    ndarray *probs = softmax(out);
    float *sorted_probs = (float *)malloc(sizeof(float) * probs->shape[0]);
    for (int i = 0; i < probs->shape[0]; i++) {
        sorted_probs[i] = ((float *)probs->data)[i];
    }

    qsort(sorted_probs, probs->shape[0], sizeof(float), compare_floats);
    float cumulative_probs = ((float *)sorted_probs)[0];
    int cutoff;
    float cutoff_value;
    for (cutoff = 0; cutoff < probs->shape[0] - 1; cutoff++) {
        if (cumulative_probs > top_p) break;
        cumulative_probs += ((float *)sorted_probs)[cutoff + 1];
    }
    cutoff_value = sorted_probs[cutoff];
    free(sorted_probs);
    for (int i = 0; i < probs->shape[0]; i++) {
        if (((float *)probs->data)[i] < cutoff_value)
            ((float *)probs->data)[i] = 0.0;
    }
    if (temperature != 1.0) {
        for (int i = 0; i < probs->shape[0]; i++) {
            ((float *)probs->data)[i] = powf(((float *)probs->data)[i], 1.0 / temperature);
        }
    }
    float sum = 0.0;
    for (int i = 0; i < probs->shape[0]; i++) {
        sum += ((float *)probs->data)[i];
    }
    unsigned long long rng_seed = (unsigned int)time(NULL);
    float coin = random_f32(&rng_seed);
    for (int i = 0; i < probs->shape[0]; i++) {
        ((float *)probs->data)[i] /= sum;
    }
    return sample_mult(probs->data, probs->shape[0], coin);
}

ndarray *forward(RWKV5 *model, int token) {
    if (model->state == NULL) {
        model->state = (ndarray *)malloc(sizeof(ndarray));
        model->state->dtype = FLOAT32;
        model->state->ndim = 2;
        model->state->shape = (int *)malloc(sizeof(int) * model->state->ndim);
        model->state->shape[0] = model->n_layer * (2 + model->head_size);
        model->state->shape[1] = model->n_embd;
        model->state->data = (float *)calloc(get_nelement(model->state->ndim, model->state->shape), sizeof(float));
    }
    ndarray *tmp = (ndarray *)malloc(sizeof(ndarray));
    ndarray *x = (ndarray *)malloc(sizeof(ndarray));

    tmp->dtype = FLOAT32;
    tmp->ndim = 1;
    tmp->shape = (int *)malloc(sizeof(int) * tmp->ndim);
    tmp->shape[0] = model->emb->shape[1];
    tmp->data = (float *)model->emb->data + token * model->emb->shape[1];
    make_strides(tmp);

    x = layer_norm(tmp, &model->n_embd, 1, model->blocks[0].ln0_weight, model->blocks[0].ln0_bias);
    free_view(tmp);
    tmp = x;

    for (int i = 0; i < model->n_layer; i++) {
        x = time_mixing(model, i, x);
        free_ndarray(tmp);
        tmp = x;
        x = channel_mixing(model, i, x);
        free_ndarray(tmp);
        tmp = x;
    }
    tmp = x;
    x = layer_norm(x, &model->n_embd, 1, model->ln_out_weight, model->ln_out_bias);
    free_ndarray(tmp);
    tmp = x;
    x = matmul(model->head_weight, x);
    free_ndarray(tmp);
    return x;
}

// void decode(){

// }

char *concat_str(char **str_array, int str_array_len) {
    // TODO: \0 ?
    char *out;
    int out_len = 0, out_alloc = 0;
    for (int i = 0; i < str_array_len; i++) {
        out_len += strlen(str_array[i]);
    }
    out = (char *)malloc(sizeof(char) * (out_len + 1));
    for (int i = 0; i < str_array_len; i++) {
        strcpy(out + out_alloc, str_array[i]);
        out_alloc += strlen(str_array[i]);
    }
    return out;
}

main(int argc, char **argv) {
#ifdef _OPENMP
    printf("OpenMP is supported!\n");
#endif
    char *checkpoint_path = "/home/kongfei/models_state_dict/rwkv_world/test.bin"; // checkpoint path
    RWKV5 model;
    VocabTable vocab_table;
    int *encode = NULL, encode_len = 0;
    init_RWKV5(&model);
    read_metadata(checkpoint_path, &model);
    read_vocab("/home/kongfei/models_state_dict/rwkv_world/vocab.bin", &vocab_table);              // vocab path
    encodeBytes("\nElon Musk has", strlen("\nElon Musk has"), &vocab_table, &encode, &encode_len); // input text
    ndarray *out;
    char *concated_str;

    for (int i = 0; i < encode_len; i++) {
        out = forward(&model, encode[i]);
        if (i != encode_len - 1) free_ndarray(out);
    }
    int token, out_last = 0;
    char *out_data[LENGTH_PER_TRIAL];
    for (int i = 0; i < LENGTH_PER_TRIAL; i++) {
        token = sample_logits(out, TEMPERATURE, TOP_P);
        if (token == END_OF_TEXT) break;
        free_ndarray(out);
        out_data[i] = decode(token, &vocab_table);
        concated_str = concat_str(out_data + out_last, i - out_last + 1);
        if (strstr(concated_str, "\ufffd") == NULL) {
            printf("%s", concated_str);
            fflush(stdout);
            out_last = i + 1;
        }
        free(concated_str);
        out = forward(&model, token);
    }
    free_ndarray(out);
    return 0;
}
