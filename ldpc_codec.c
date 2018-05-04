/* Copyright 2018 Frederik M.J. Vestre under GPL / 3 cause BSD. Derivative works may be distributed
   under only one license at their discretion. 
   The sllr_fft function is not covered by this copyright. 

   Inspired by GFq_LDPC_NTT.c by Seishi Takamura ( http://ivms.stanford.edu/~varodayan/multilevel/index.html )
   Based on algorithms outlined in the paper "Fourier Transform Decoding of Non-Binary LDPC Codes"
   by Geoffrey J. Byers and Fambirai Takawira School of Electrical,
   Electronic and Computer Engineering, University of KwaZulu-Natal
   http://www.satnac.org.za/proceedings/2004/AccessCoding/No%2078%20-%20Byers.pdf

License for code (except sllr_fft):

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License version 3.0 or 2.1, as
  published by the Free Software Foundation.  This program is
  distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with this program; if not, see <http://www.gnu.org/licenses/>.


Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY  OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
*/

#include <immintrin.h> 
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "ldpc_codec.h"

typedef uint_fast16_t alist_idx;
typedef uint_fast8_t alist_val;

typedef int_fast32_t intgf_t;
typedef uint_fast32_t uintgf_t;

typedef int32_t ntt_t;

#define Q   (1<<Log2Q)      // GF(Q)

#if Q==64
//Octave: log(gf(1:63,6))
const alist_val logq[64] = {0,0,1,6,2,12,7,26,3,32,13,35,8,48,27,18,4,24,
  33,16,14,52,36,54,9,45,49,38,28,41,19,56,5,62,
  25,11,34,31,17,47,15,23,53,51,37,44,55,40,10,
  61,46,30,50,22,39,43,29,60,42,21,20,59,57,58};
//Octave: exp(gf(1:62,6))
const alist_val expq[63] = {1,2,4,8,16,32,3,6,12,24,48,35,5,10,20,40,19,
  38,15,30,60,59,53,41,17,34,7,14,28,56,51,37,
  9,18,36,11,22,44,27,54,47,29,58,55,45,25,50,
  39,13,26,52,43,21,42,23,46,31,62,63,61,57,49,33};
#elif Q==256
//Octave: log(gf(0:254,8))
const alist_val logq[256] = {255,0,1,25,2,50,26,198,3,223,51,238,27,104,199,75,4,100,
  224,14,52,141,239,129,28,193,105,248,200,8,76,113,5,138,101,47,225,
  36,15,33,53,147,142,218,240,18,130,69,29,181,194,125,106,39,249,185,
  201,154,9,120,77,228,114,166,6,191,139,98,102,221,48,253,226,152,37,
  179,16,145,34,136,54,208,148,206,143,150,219,189,241,210,19,92,131,
  56,70,64,30,66,182,163,195,72,126,110,107,58,40,84,250,133,186,61,202,
  94,155,159,10,21,121,43,78,212,229,172,115,243,167,87,7,112,192,247,
  140,128,99,13,103,74,222,237,49,197,254,24,227,165,153,119,38,184,180,
  124,17,68,146,217,35,32,137,46,55,63,209,91,149,188,207,205,144,135,151,
  178,220,252,190,97,242,86,211,171,20,42,93,158,132,60,57,83,71,109,65,
  162,31,45,67,216,183,123,164,118,196,23,73,236,127,12,111,246,108,161,59,
  82,41,157,85,170,251,96,134,177,187,204,62,90,203,89,95,176,156,169,160,
  81,11,245,22,235,122,117,44,215,79,174,213,233,230,231,173,232,116,214,
  244,234,168,80,88,175};

//Octave: exp(gf(0:254,8))
const alist_val expq[255] = {1,2,4,8,16,32,64,128,29,58,116,232,205,135,19,38,76,
  152,45,90,180,117,234,201,143,3,6,12,24,48,96,192,157,39,78,156,
  37,74,148,53,106,212,181,119,238,193,159,35,70,140,5,10,20,40,80,
  160,93,186,105,210,185,111,222,161,95,190,97,194,153,47,94,188,101,
  202,137,15,30,60,120,240,253,231,211,187,107,214,177,127,254,
  225,223,163,91,182,113,226,217,175,67,134,17,34,68,136,13,26,52,104,
  208,189,103,206,129,31,62,124,248,237,199,147,59,118,236,197,151,51,
  102,204,133,23,46,92,184,109,218,169,79,158,33,66,132,21,42,84,168,
  77,154,41,82,164,85,170,73,146,57,114,228,213,183,115,230,209,191,99,
  198,145,63,126,252,229,215,179,123,246,241,255,227,219,171,75,150,49,
  98,196,149,55,110,220,165,87,174,65,130,25,50,100,200,141,7,14,28,56,
  112,224,221,167,83,166,81,162,89,178,121,242,249,239,195,155,43,86,172,
  69,138,9,18,36,72,144,61,122,244,245,247,243,251,235,203,139,11,22,44,
  88,176,125,250,233,207,131,27,54,108,216,173,71,142};
#else
#error "Multiplication matrices are not defined for this field"
#endif

/* Multiplication & division over finite field is easiest done using logarithms, as suggested in 
https://en.wikipedia.org/wiki/Finite_field_arithmetic#Multiplicative_inverse */
static alist_val GF_mul(alist_val a, alist_val b)
{
  if (a == 0 || b == 0) return 0;
  if (a == 1) return b;
  if (b == 1) return a;
  assert(a<256 && b<256);
  return expq[(logq[a] + logq[b]) % (Q-1)];
}

#define GF_add(a, b) ((a)^(b))
#define GF_sub(a, b) ((a)^(b))

typedef struct {
  alist_idx idx;
  alist_idx peer_idx;
  alist_val value;
} gf_val_t;

struct alist_matrix{
  alist_idx N, M, NmM;
  alist_idx q;
  alist_idx cmax,rmax;
  gf_val_t *mlist;
  gf_val_t *nlist;
  alist_idx *col_weight;
  alist_idx *row_weight;
  alist_val *gen_mtx;
};

typedef struct 
{
  alist_matrix_t *mtx;
  ntt_t *s_llr; //source val (q) llr
  ntt_t *q_llr; //val (q) llr
  ntt_t *r_llr; //check (r) llr
} alist_decode_state_t;

/* Lookup functions for multidimentional arrays (to avoid nested pointers and pointer arithmetic inside the rest of the code) */
static gf_val_t *get_nlist_idx(alist_matrix_t *mtx, int i, int j)
{
  return &(mtx->nlist[(i*mtx->cmax)+j]);
}
static gf_val_t *get_mlist_idx(alist_matrix_t *mtx, int i, int j)
{
  return &(mtx->mlist[(i*mtx->rmax)+j]);
}

static alist_val *get_gen_mtx_idx(alist_matrix_t *mtx, int i, int j)
{
  return &(mtx->gen_mtx[(i*(mtx->NmM))+j]);
}

static ntt_t *fQa_idx(ntt_t* fQa, alist_idx ridx, alist_val val)
{
  return &(fQa[(ridx*Q)+val]);
}

static bool *fQaB_idx(bool* fQa, alist_idx ridx, alist_val val)
{
  return &(fQa[(ridx*Q)+val]);
}

static ntt_t *qr_llr_idx(ntt_t* qr, alist_idx rmax, alist_idx cn_idx, alist_idx row_idx, alist_idx val_idx)
{
  return &qr[(((cn_idx*rmax)+row_idx)*Q)+val_idx];
}

//Somewhat based on GFq_LDPC_NTT.c, but mostly rewritten 
//http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
alist_matrix_t *ldpc_read_alist(char *chk_fn,char *gen_fn)
{
  alist_matrix_t *result;
  result = malloc(sizeof(alist_matrix_t));


  int *count;
  uint_fast32_t q;
  alist_idx i,j,int_tmp;
  char buf[BUFSIZ];
  FILE *fp = fopen(chk_fn, "rt");
  if (fp == NULL) {
    fprintf(stderr, "cannot open %s\n", chk_fn);
    exit(-2);
  }
  (void)fgets(buf, sizeof(buf), fp);
  if (sscanf(buf, "%lu%lu%lu", &(result->N), &(result->M), &q) == 2) {
    fprintf(stderr, "Warning! A binary matrix seems to be specified.\n");
    fprintf(stderr, "I'll use it anyways.\n");
    return NULL;
  }

  result->NmM = result->N - result->M;
  result->q = q;

  if (q > Q){
    printf("Q bigger than expected %lu>%u\n", q, Q);
    return NULL;
  } 
  if (q != Q) {
    fprintf(stderr, "GF order(%u) does not match with matrix file(%lu)\n", Q, q);
    return NULL;
  }
  (void)fscanf(fp, "%lu%lu", &(result->cmax), &(result->rmax));
  result->col_weight = malloc(sizeof(alist_idx) * result->N);
  for (i = 0; i < result->N; i++) {
    (void)fscanf(fp, "%lu", &int_tmp);
    result->col_weight[i] = int_tmp;
    if(result->col_weight[i] > result->cmax)
    {
      fprintf(stderr, "Col at index %lu too long %lu>%lu\n", j, result->col_weight[j], result->cmax);
    }
  }

  result->row_weight = malloc(sizeof(alist_idx) * result->M);
  for (j = 0; j < result->M; j++)
  {
    (void)fscanf(fp, "%lu", &int_tmp);
    result->row_weight[j] = int_tmp;
    if(result->row_weight[j] > result->rmax)
    {
      fprintf(stderr, "Row at index %lu too long %lu>%lu\n", j, result->row_weight[j], result->rmax);
    }
  }

  result->nlist = malloc(sizeof(gf_val_t)*result->N*result->cmax);
  result->mlist = malloc(sizeof(gf_val_t)*result->M*result->rmax);
  alist_idx *m_offset_tmp = malloc(sizeof(alist_idx)*result->N);
  memset(m_offset_tmp, 0, sizeof(alist_idx)*result->N);

  //Read N matrix
  for (i = 0; i < result->N; i++) {
    alist_val v;
    for (j = 0; j < result->cmax; j++)
    {
      (void)fscanf(fp, "%hhu%lu", &v, &int_tmp);
      gf_val_t * const nlist_idx   = get_nlist_idx(result, i,j);
      nlist_idx->idx = v-1;
      nlist_idx->value = int_tmp;
    }
  }

  //Read M matrix
  for (i = 0; i < result->M; i++) {
    assert(result->row_weight[i] <= result->rmax);
    for (j = 0; j < result->row_weight[i]; j++) {
      gf_val_t * const mlist_idx = get_mlist_idx(result, i,j);
      alist_val v;
      (void)fscanf(fp, "%hhu%lu", &v, &int_tmp);
      mlist_idx->idx = v-1;
      mlist_idx->value = int_tmp;
      mlist_idx->peer_idx = m_offset_tmp[mlist_idx->idx]; //This is probably not used, but set it for completeness

      gf_val_t * const nlist_peer = get_nlist_idx(result, mlist_idx->idx,m_offset_tmp[mlist_idx->idx]++);
      if(m_offset_tmp[mlist_idx->idx]>result->cmax)
      {
        printf("Error: Too many checks (%lu) point at same value node:%lu\n", m_offset_tmp[mlist_idx->idx], mlist_idx->idx);
      }
      if((nlist_peer->idx != i) || (nlist_peer->value != int_tmp))
      {
        printf("Warning: M and N disagrees (or is diffrently ordered %lu, %lu\n", i, j);
      }
      nlist_peer->idx = i;
      nlist_peer->peer_idx = j;
      nlist_peer->value = int_tmp;
    }

    for (; j < result->rmax; j++) {//skip fillers
      uint_fast32_t a,b;
      (void)fscanf(fp, "%lu%lu", &a, &b);
      if (a!=0 || b!=0) {
        printf("error at row %lu, %lu %lu\n", i, a, b);
        exit(-1);
      }
    }
  }
  //free(count);
  fclose(fp);

  fp = fopen(gen_fn, "rt");
  if (fp == NULL) {
    fprintf(stderr, "cannot open %s\n", chk_fn);
    exit(-2);
  }
  (void)fgets(buf, sizeof(buf), fp);
  {
    uint_fast32_t n,m,q;
    if (sscanf(buf, "%lu%lu%lu", &n, &m, &q) == 2) {
      fprintf(stderr, "Warning! A generator binary matrix seems to be specified.\n");
      return NULL;
    }
    if (m!=result->N || n!=result->NmM || q != result->q) 
    {
      fprintf(stderr, "Generator matrix dimensions does not match check matrix dimensions %lu=%lu, %lu=%lu, %lu=%d\n",n, result->N, m, result->NmM, q,  result->q);
      return NULL;      
    }
  }
  result->gen_mtx = malloc(sizeof(alist_val)*(result->NmM)*result->M);
  for (i=0; i < result->M; i++) {
    alist_val v;
    for (j = 0; j < (result->NmM); j++)
    {
      alist_val * const mlist_idx = get_gen_mtx_idx(result, i,j);
      (void)fscanf(fp, "%hhu ", &v);
      *mlist_idx = v;
    }
  }
  fclose(fp);

  free(m_offset_tmp);
  return result;
}

void ldpc_free_alist(alist_matrix_t *alist)
{
  free(alist->nlist);
  free(alist->mlist);
  free(alist->col_weight);
  free(alist->row_weight);
  free(alist->gen_mtx);
  free(alist);
}

alist_decode_state_t *alloc_state(alist_matrix_t *mtx)
{
  alist_decode_state_t *state = malloc(sizeof(alist_decode_state_t));
  state->mtx = mtx;
  state->s_llr = malloc(sizeof(ntt_t) * state->mtx->N * Q);
  state->r_llr = malloc(sizeof(ntt_t) * state->mtx->M * Q * mtx->rmax);
  state->q_llr = malloc(sizeof(ntt_t) * state->mtx->M * Q * mtx->rmax);
  return state;
}

void free_state(alist_decode_state_t *state, bool free_mtx)
{
  if(free_mtx)
  {
    ldpc_free_alist(state->mtx);
  }
  free(state->s_llr);
  free(state->q_llr);
  free(state->r_llr);
}

/* The enc function may be optimized by using an a-list like matrix and only calculating on non-0 indices */
void enc(alist_matrix_t *mtx, alist_val n[], alist_val m[])
{
  alist_idx i, j;
  for (j = 0; j < mtx->M; j++) {
    alist_val sum = 0;
    for (i = 0; i < mtx->NmM; i++) {
      alist_val * const mlist_idx = get_gen_mtx_idx(mtx, j, i);
      alist_val v = GF_mul(*mlist_idx, n[i]);
      sum = GF_add(v, sum);
    }
    m[j] = sum % Q;
  }
}

/* Here a log base of not-e is used to give more lookup values (and thus a more numerically 
   "correct" decode) since all LLR's are integers and thus has no decimal precision. The log
   base is determined  using trial and error, and made sure to fit (i.e. be 0 for the 
   topmost items) the lookup table below. */
#define LOG_BASE (1.017)
#define LLR_ADD_LOOKUP_MAX (256)
/* Use same datatype as ntt_t to speed up code */
#define ntt_lu_t ntt_t
ntt_lu_t llr_add_positive_lookup[LLR_ADD_LOOKUP_MAX];
ntt_lu_t llr_add_negative_lookup[LLR_ADD_LOOKUP_MAX];
static ntt_t generate_lookup_val(bool pos, ntt_lu_t d)
{
  return (ntt_t)(log(1+((pos?1:-1)*pow(LOG_BASE, -1*abs((double)d))))/log(LOG_BASE));
}
static ntt_lu_t generate_lookup(ntt_lu_t *lookup, bool pos, size_t lookup_max)
{
  for(size_t i=0;i<lookup_max;i++)
  {
    lookup[i] = abs(generate_lookup_val(pos, i));
    //Printf to check that the lookup table is big enough if changing LOG_BASE
    //printf("D:%d,%d\n",i,lookup[i]);
  }
}
static ntt_t lookup(bool pos, ntt_t d)
{
  return abs(d)<LLR_ADD_LOOKUP_MAX ? (pos ? llr_add_positive_lookup[d]:(-1*(ntt_t)llr_add_negative_lookup[d])):0;
}

static ntt_t sllr_add(const ntt_t l1, const ntt_t l2, const bool s1, const bool s2, bool *s_out)
{
#define max(a,b) (((a)>(b))?(a):(b))
#define abs(a) (((a)>0)?(a):(-1*(a)))
  bool seq = s1==s2;
  *s_out = (seq || (l1 >= l2)) ? (s1) : (!s1);
  return max(l1,l2) + lookup(seq, abs(l1-l2));
#undef max
#undef abs
}

#define sllr_sub(l1,l2,s1,s2,s_out) (sllr_add((l1),(l2),(s1),!(s2),(s_out)))

/* Based on GFq_LDPC_NTT.c, adapted for LLR. According to Seishi Takamura
the following function is based on Prof. MacKay's FFT code (with a small change), but i have not been 
able to find the code at his webpage.  */
/* TODO: Rewrite this function based on algorithm in the paper (noted above) as license is unclear. */
void sllr_fft(ntt_t p_m[Q], bool p_s[Q])
{
#if 0
  /* The AVX2 code is not working (and written by me, Frederik) */
  const int32_t offsets[8] = {0, 1, 2,3,4,5,6,7};
  int_fast32_t b, factor = 1, rest;

  for (b = 0; b < Log2Q; b++) 
  {
    for (rest = 0; rest < Q/2; rest++) 
    {
      
      __m256i restV = _mm256_add_epi32(_mm256_set1_epi32(rest), _mm256_loadu_si256(offsets));
      __m256i restH = _mm256_sll_epi32(restV,  _mm_set1_epi32(b));
      __m256i restL = _mm256_and_si256(restV,  _mm256_set1_epi32(factor-1));
      __m256i rest0 = _mm256_add_epi32(_mm256_sll_epi32(restH, _mm256_set1_epi32(b+1)), restL);
      __m256i rest1 = _mm256_add_epi32(rest0, _mm256_set1_epi32(factor));

      ntt_t prest0_m = p_m[rest0];
      bool  prest0_s = p_s[rest0];
      p_m[rest0] = sllr_add(p_m[rest0], p_m[rest1], p_s[rest0], p_s[rest1], &p_s[rest0]);
      p_m[rest1] = sllr_sub(prest0_m,   p_m[rest1],   prest0_s, p_s[rest1], &p_s[rest1]);
    }
    factor += factor;
  }
#else
  int_fast32_t b, factor = 1, rest;
  for (b = 0; b < Log2Q; b++) 
  {
    for (rest = 0; rest < Q/2; rest++) 
    {
      int_fast32_t restH = rest >> b;
      int_fast32_t restL = rest & (factor-1);

      int_fast32_t rest0 = (restH << (b+1)) + restL;
      int_fast32_t rest1 = rest0 + factor;
      ntt_t prest0_m = p_m[rest0];
      bool  prest0_s = p_s[rest0];
      p_m[rest0] = sllr_add(p_m[rest0], p_m[rest1], p_s[rest0], p_s[rest1], &p_s[rest0]);
      p_m[rest1] = sllr_sub(prest0_m,   p_m[rest1],   prest0_s, p_s[rest1], &p_s[rest1]);
    }
    factor += factor;
  }
#endif
}

/* Only used for first decode */
void decode(alist_decode_state_t *state, uint8_t *decode_dest)
{
  for(size_t i=0;i<state->mtx->N;i++)
  {     
    ntt_t max_val = INT32_MIN;
    alist_val max_idx = 0;
    ntt_t rvv[state->mtx->cmax];
    memset(rvv,0,sizeof(rvv));
    for(size_t a=0;a<Q; a++)
    {
      ntt_t llr_val = state->s_llr[(i*Q)+a];
      if(llr_val > max_val)
      {
        max_val = llr_val;
        max_idx = a;
      }
    }
    decode_dest[i] = max_idx;
  } 
}

void validate(alist_decode_state_t *state, uint8_t *decode_dest, uint_fast32_t *vnum)
{
    uint_fast32_t i,k, tmp;
    uint_fast32_t v_total = 0;
    gf_val_t *cidx;
    #pragma omp parallel for reduction(+:v_total)
    for(i=0;i<state->mtx->M;i++)
    {
      tmp = 0;
      for(k=0; k<state->mtx->row_weight[i]; k++)
      {
        cidx = get_mlist_idx(state->mtx, i, k);
        tmp = GF_add(tmp, GF_mul(cidx->value, decode_dest[cidx->idx]));
      }
      if(tmp != 0)
      {
        v_total++;
      } 
    }
    (*vnum)+=v_total;
}

// Sum Product Decoder
// loop_max: maxp iteration
int dec(alist_decode_state_t * restrict state, uint_fast32_t loop_max, uint8_t *decode_dest)
{
  
  alist_matrix_t* mtx = state->mtx;
  uint_fast32_t i, l, vnum;
  bool valid = false;

  // logqa(col) is for llr values
  // logra(col) is for llr check
  //Init logqa:
  for (i = 0; i < mtx->N; i++) {
    for (int_fast32_t k = 0; k < mtx->col_weight[i]; k++) {
      gf_val_t *cidx = get_nlist_idx(state->mtx, i, k);
      memcpy(qr_llr_idx(state->q_llr, mtx->rmax, cidx->idx, cidx->peer_idx,0), &state->s_llr[i*Q], sizeof(ntt_t)*Q);
    }
  }

  //Step 2: Tentative decoding ... 
  decode(state, decode_dest);
  //Step 2: ... & Validate parity
  vnum=0;
  validate(state, decode_dest, &vnum);
  if(vnum == 0)
  {
    printf("Iter %lu, Initial violated checks: %lu/%lu\n", l, vnum, mtx->M);
    valid = true;
    return 0;
  }
  //v_llr = R
  //c_llr = Q
  for(l=0;l<loop_max;l++){
    vnum = 0;

    //Try to update LLR using sfft
    #pragma omp parallel for
    for(i=0;i<mtx->M;i++)
    {
      gf_val_t *cidx;
      ntt_t fQa_m[Q * mtx->rmax];
      bool fQa_s[Q * mtx->rmax];
      
      ntt_t prod_m[Q];
      bool prod_s[Q];

      memset(prod_m, 0, sizeof(prod_m));
      memset(prod_s, 0, sizeof(prod_s));
      for(int_fast32_t k=0; k<mtx->row_weight[i]; k++)
      {
        cidx = get_mlist_idx(state->mtx, i, k);
        memset((fQa_idx(fQa_m, k, 0)), 0, sizeof(ntt_t)*Q);
        memset((fQaB_idx(fQa_s, k, 0)), 0, sizeof(bool)*Q);
        for (int_fast32_t a = 0; a < Q; a++) 
        {
          *(fQa_idx(fQa_m, k, GF_mul(cidx->value, a))) = *qr_llr_idx(state->q_llr, mtx->rmax, i, k,a);
          *(fQaB_idx(fQa_s, k, GF_mul(cidx->value, a))) = 0;
        }
        
        sllr_fft(fQa_idx(fQa_m, k, 0), fQaB_idx(fQa_s, k, 0));

        for (int_fast32_t a = 0; a < Q; a++) 
        {
          if(*(fQaB_idx(fQa_s, k, a)) == 0) //Negative
          {
            prod_s[a] ^= 1; //Flip sign
          }
          prod_m[a] += *( fQa_idx(fQa_m, k, a));
        }
      }

      for (int_fast32_t k = 0; k < mtx->row_weight[i]; k++) 
      {
        ntt_t fRa_m[Q];
        bool  fRa_s[Q];

        for (int_fast32_t a = 0; a < Q; a++) 
        {          
          fRa_s[a] = *(fQaB_idx(fQa_s, k, a)) != prod_s[a];
          ntt_t fram = (prod_m[a] - *( fQa_idx(fQa_m, k, a))) ; 
          fram = fram >> 1; //Scale fram down to get more numerically stable results
          fRa_m[a] = fram;
          //printf("c:l:%03d,i:%03d,k:%03d,a:%02d, Fram:%05d\n", l, i,k, a, fRa_m[a]);
          if(i==4)// && a == 10)
          {
            //printf("c:l:%03d,i:%03d,k:%03d,a:%02d, Fram:%05d, Fras:%d\n", l, i,k, a, fram, fRa_s[a]);
          }
        }

        sllr_fft(fRa_m, fRa_s);

        cidx = get_mlist_idx(state->mtx, i, k);
        for (int_fast32_t a = 0; a < Q; a++) 
        {
          ntt_t fram = fRa_m[GF_mul(cidx->value, a)] - Q ;
#if 0
          if(fRa_s[GF_mul(cidx->value, a)] != 1)
          {
            printf("C:l:%03d,i:%03d,k:%03d,a:%02d, Fram:%05d, Fras:%d\n", l, i,k, a, fram, fRa_s[GF_mul(cidx->value, a)]);
          }
#endif
#if 0
          if(i==4)// && a == 10)
          {
            printf("C:l:%03d,i:%03d,k:%03d,a:%02d, Fram:%d\n", l, i,k, a, fram);
          }
#endif
          *qr_llr_idx(state->r_llr, mtx->rmax, i,k, a) = fram;
        }
      }
    }

    uint_fast32_t csym = 0;
    // update q (q^a_mnf_n^a sum M r_jn^a)
    // From variable -> check
    #pragma omp parallel for reduction(+:csym)
    for(i=0;i<mtx->N;i++)
    {
      gf_val_t *cidx;
      ntt_t max_val = INT32_MIN;
      alist_val max_idx = 0;
      /* These variables are only needed to report number of changed  symbols */
      ntt_t smax_val = INT32_MIN;
      alist_val smax_idx = 0;

      for (int_fast32_t a = 0; a < Q; a++) 
      {
        intgf_t logprod = state->s_llr[(i*Q)+a];
        if(logprod > smax_val)
        {
          smax_val = logprod;
          smax_idx = a;
        }
        for(int_fast32_t k=0; k < mtx->col_weight[i]; k++)
        {
          cidx = get_nlist_idx(state->mtx, i, k);
          logprod += *qr_llr_idx(state->r_llr, mtx->rmax, cidx->idx, cidx->peer_idx, a);
        }
        for (int_fast32_t k = 0; k < mtx->col_weight[i]; k++) 
        {
          cidx = get_nlist_idx(state->mtx, i, k);
          ntt_t qllr = logprod - *qr_llr_idx(state->r_llr, mtx->rmax, cidx->idx, cidx->peer_idx, a); //taka product except k
          *qr_llr_idx(state->q_llr, mtx->rmax, cidx->idx, cidx->peer_idx, a) = qllr;
        }
        if(logprod > max_val)
        {
          max_val = logprod;
          max_idx = a;
        }
      }
      for(int_fast32_t k=0; k < mtx->col_weight[i]; k++)
      {
        for (int_fast32_t a = 0; a < Q; a++) 
        {
          ntt_t *qli = (qr_llr_idx(state->q_llr, mtx->rmax, cidx->idx, cidx->peer_idx, a));
          *qli = *qli - max_val;
        }
      }
      if(max_idx!=smax_idx)
      {
        csym++;
        //printf("C:l:%03d,i:%03d,k:---,a:%02d, sv:%05d, nv:%015d, si:%02d, ni:%02d\n", loop, i,a, smax_val, max_val, smax_idx,max_idx);
      }
      decode_dest[i] = max_idx;
    }
    validate(state, decode_dest, &vnum);
    if(vnum == 0)
    {
      printf("Iter %lu, Violated checks: %02lu/%02lu Changed symbols:%03d/%03d\n", l, vnum, mtx->M, csym, mtx->N);
      valid = true;
      break;
    }
  }

  if(vnum !=0)
  {
    printf("ITER %lu, Violated checks: %02lu/%02lu\n", l, vnum, mtx->M);
    return -1;
  }
  return l;
}

#if LOGQ == 6
static void gf256togf64(uint_fast32_t N, uint8_t *src, uint8_t *dst)
{
  for(int i =0;i<(N/4);i++)
  {
    uint32_t data = (src[(i*3)+2]) | (src[(i*3)+1] << 8) | (src[(i*3)+0] << 16);
    dst[(i*4)+3] = data & ((1<<6)-1);
    dst[(i*4)+2] = (data >>  6) & ((1<<6)-1);
    dst[(i*4)+1] = (data >> 12) & ((1<<6)-1);
    dst[(i*4)+0] = (data >> 18) & ((1<<6)-1);
  }
}

static void gf64togf256(uint_fast32_t N, uint8_t *src, uint8_t *dst)
{
  for(int i =0;i<(N/4);i++)
  {
    uint32_t data = (src[(i*4)+3]) | (src[(i*4)+2] << 6) | (src[(i*4)+1] << 12) | (src[(i*4)+0] << 18);
    dst[(i*3)+2] = data & ((1<<8)-1);
    dst[(i*3)+1] = (data >>  8) & ((1<<8)-1);
    dst[(i*3)+0] = (data >> 16) & ((1<<8)-1);
  }
}
#endif

//End from GFq_LDPC_NTT.c
void ldpc_encode(alist_matrix_t* mtx, uint8_t *src_data, uint8_t *dst_data)
{

    alist_val *x = malloc(sizeof(alist_val) * mtx->N);  // source
    alist_val *s = &x[mtx->NmM];// syndrome (output)
#if Log2Q == 6
    gf256togf64(mtx->NmM, src_data, x);
#elif Log2Q == 8
    memcpy(x, src_data,mtx->NmM);
#else
    #error "LogQ size not supported"
#endif
    enc(mtx, x, s);
#if Log2Q == 6
    gf64togf256(mtx->N, x, dst_data);
#elif Log2Q == 8
    memcpy(dst_data, x,mtx->N);
#else
    #error "LogQ size not supported"
#endif
    free(x);
}
void encfile(alist_matrix_t* mtx, char *src_fn,char *dst_fn)
{
  size_t nbytesup = ((((mtx->N*Log2Q))/8));
  uint8_t *scratch_data = malloc(sizeof(uint8_t) * nbytesup);  // dst
  FILE *fp = fopen(src_fn, "rb");
  if (fp == NULL) {
    fprintf(stderr, "cannot open %s\n", src_fn);
    exit(-2);
  }
  assert(fread(scratch_data, sizeof(uint8_t), ((mtx->NmM*Log2Q)/8), fp)==((mtx->NmM*Log2Q)/8));
  fclose(fp);

  ldpc_encode(mtx,scratch_data, scratch_data);

  fp = fopen(dst_fn, "wb");
  assert(fwrite(scratch_data, sizeof(uint8_t), nbytesup, fp)==nbytesup);
  fclose(fp);
  free(scratch_data);
}

//#define S2D (4)

void ldpc_decode(alist_matrix_t* mtx, int32_t *src_data,uint8_t *dst_data)
{
    alist_decode_state_t *state = alloc_state(mtx);
    size_t nbytesup = ((((mtx->N*Log2Q))/8));
    uint8_t *src_dcd_data = malloc(sizeof(uint8_t) * ((mtx->N)));   // decode destination
    // Convert from binary LLR to GF(q) llr
    memset(dst_data, 0, sizeof(dst_data));
    size_t src_data_offset, dst_data_offset;
    
    for(size_t i =0;i<(mtx->N);i++)
    {
      src_data_offset = i * Log2Q;
      dst_data_offset = i * Q;
      ntt_t pvs = 0;
      ntt_t pv = 0;
      for(int a=0;a<Q;a++)
      {
        ntt_t fct = 0;
        pv = 0;
        //Convert from LLR(2) to LLR(Q)
        for(int k=0;k<Log2Q;k++)
        {
          ntt_t bit_val = (src_data[src_data_offset+(Log2Q-k-1)]);
          pv += ((((a&(1<<k))==0)?-1 : 1)*bit_val);
        }

        state->s_llr[dst_data_offset+a] = 1*(pv);
        pvs+=pv;
      }
      /* Make Log likelyhood to log likelyhood ratio */
      ntt_t a0 = state->s_llr[dst_data_offset];
      for(int a=0;a<Q;a++)
      {
        state->s_llr[dst_data_offset+a] -= a0;
      }
    }
    memset(src_data, 0, sizeof(src_data)); 
    {
      dec(state, 100, src_dcd_data);
    }

#if Log2Q == 6
    gf64togf256(mtx->N, src_dcd_data, dst_data);
#elif Log2Q == 8
    memcpy(dst_data, src_dcd_data, mtx->N);
#else
    #error "LogQ size not supported"
#endif

    free_state(state, false);
}
void decfile(alist_matrix_t* mtx, char *src_fn,char *dst_fn)
{
    size_t nbytesup = ((((mtx->N*Log2Q))/8));
    int32_t *src_data = malloc(sizeof(uint32_t) * mtx->N*Log2Q);  // source
    uint8_t *dst_data = malloc(sizeof(uint8_t) * nbytesup);   // dst
    FILE *fp = fopen(src_fn, "rb");
    if (fp == NULL) {
      fprintf(stderr, "cannot open %s\n", src_fn);
      exit(-2);
    }
    assert(fread(src_data, sizeof(int32_t), ((mtx->N*Log2Q)), fp)==((mtx->N*Log2Q)));
    fclose(fp);
    ldpc_decode(mtx,src_data, dst_data);
    fp = fopen(dst_fn, "wb");
    assert(fwrite(dst_data, sizeof(uint8_t), ((mtx->N*Log2Q)/8), fp)==((mtx->N*Log2Q)/8));
    fclose(fp);
    free(src_data);
    free(dst_data);
}

blocksize_info_t ldpc_get_blocksize(alist_matrix_t* mtx)
{
  blocksize_info_t ret;
  ret.n = mtx->N;
  ret.m = mtx->N;
  ret.log2q = Log2Q;
  ret.data_bytes = ((((mtx->NmM*Log2Q))/8));
  ret.block_bytes = ((((mtx->N*Log2Q))/8));
  ret.block_llr_int32_s = (mtx->N*Log2Q);
  return ret;
}
void ldpc_init()
{
  generate_lookup(llr_add_positive_lookup, true, LLR_ADD_LOOKUP_MAX);
  generate_lookup(llr_add_negative_lookup, false, LLR_ADD_LOOKUP_MAX);
  llr_add_negative_lookup[0] = 255;
}
#ifdef LDPC_STANDALONE
int main(int argc, char **argv)
{
  ldpc_init();
  printf("Lookup @ max: %d, %d\n", llr_add_positive_lookup[LLR_ADD_LOOKUP_MAX-1], llr_add_negative_lookup[LLR_ADD_LOOKUP_MAX-1]);
  //printf("Look0:%lu\n", generate_lookup_val(false, 0));
  alist_matrix_t *alist = ldpc_read_alist(argv[1], argv[2]);   
  printf("Mode: >%s<\n", argv[3]);
  if(strncmp(argv[3], "enc",3)==0)
  {
    encfile(alist, argv[4], argv[5]);
  }
  else
  {
    decfile(alist, argv[4], argv[5]);
  }

  if(alist)
  {
    ldpc_free_alist(alist);
  }
  printf("GF_mul:%d\n",  GF_mul(42, 22));
}
#endif