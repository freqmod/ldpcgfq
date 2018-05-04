/* Copyright 2018 Frederik M.J. Vestre under GPL / 3 cause BSD. Derivative works may be distributed
   under only one license at their discretion. 
*/
#ifndef LDPC_CODEC_H_
#include <stdint.h>
#include <stdbool.h>

typedef struct 
{
    uint32_t n;
    uint32_t m;
    uint32_t log2q;
    uint32_t data_bytes;
    uint32_t block_bytes;
    uint32_t block_llr_int32_s;
}
blocksize_info_t;
struct alist_matrix;
typedef struct alist_matrix alist_matrix_t;
alist_matrix_t *ldpc_read_alist(char *chk_fn,char *gen_fn);
void ldpc_encode(alist_matrix_t* mtx, uint8_t *src_data, uint8_t *dst_data);
void ldpc_decode(alist_matrix_t* mtx, int32_t *src_data,uint8_t *dst_data);
void ldpc_free_alist(alist_matrix_t *alist);
void ldpc_init();
blocksize_info_t ldpc_get_blocksize(alist_matrix_t* mtx);

#endif
