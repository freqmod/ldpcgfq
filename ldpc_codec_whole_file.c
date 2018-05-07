/* Copyright 2018 Frederik M.J. Vestre under GPL / 3 cause BSD. Derivative works may be distributed
   under only one license at their discretion. 
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "ldpc_codec.h"
int main(int argc, char **argv)
{
  ldpc_init();
  alist_matrix_t *alist = ldpc_read_alist(argv[1], argv[2]);   
  blocksize_info_t alist_info = ldpc_get_blocksize(alist);
  uint8_t data_block[alist_info.block_bytes];
  int32_t llr_block[alist_info.block_llr_int32_s];
  bool encode =strncmp(argv[3], "enc",3)==0;
  bool decode_writefull = strncmp(argv[3], "decw",3)==0;
  size_t debytes_to_write;
  if(decode_writefull)
  {
    debytes_to_write = alist_info.block_bytes;
  }
  else
  {
    debytes_to_write = alist_info.data_bytes; 
  }
  FILE *fip = fopen(argv[4], "rb");
  if (fip == NULL) {
    fprintf(stderr, "cannot open >%s<\n", argv[3]);
    exit(-2);
  }
  FILE *fop = fopen(argv[5], "wb");
  if (fop == NULL) {
    fprintf(stderr, "cannot open >%s<\n", argv[4]);
    exit(-2);
  }

  while(true)
  {
    if(encode)
    {
      
      memset(data_block, 0, sizeof(data_block));
      size_t data_read = fread(data_block, sizeof(uint8_t), alist_info.data_bytes, fip);
      if(data_read == 0)
      {
        break;
      }
      ldpc_encode(alist, data_block, data_block);
      size_t data_written = fwrite(data_block, sizeof(uint8_t), alist_info.block_bytes, fop);
      if(data_read != alist_info.data_bytes)
      {
         break;
      }
    }
    else
    {
      memset(data_block, 0, sizeof(data_block));
      memset(llr_block, 0, sizeof(llr_block));
      size_t data_read = fread(llr_block, sizeof(int32_t), alist_info.block_llr_int32_s, fip);
      if(data_read == 0)
      {
        break;
      }
      ldpc_decode(alist, (int32_t*)llr_block, data_block);
      size_t data_written = fwrite(data_block, sizeof(uint8_t), debytes_to_write, fop);
      if(data_read != alist_info.block_llr_int32_s)
      {
         break;
      }
    }
  }
  fclose(fip);
  fclose(fop);
}