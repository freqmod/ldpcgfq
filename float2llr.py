# -*- coding: UTF-8 -*-
import sys, struct

import numpy as np

read_data = None
with open(sys.argv[1],"rb") as fh:
    read_data = fh.read()
#0 = -0.5
#+1 = 0.5
float_scale_one = 0.5

#pos_int = struct.pack("<i", 4096)
#neg_int = struct.pack("<i", -4096)
floatsize = struct.calcsize("<f")
write_data = np.zeros(len(read_data)/floatsize, dtype=np.uint32)
for x in xrange(0,len(read_data)/floatsize,8):
  for y in xrange(8):
    ii = (x)+y
    oi = (x)+7-y
    soft_val = struct.unpack("<f", read_data[ii*floatsize:(ii+1)*floatsize])[0]
    #The multiplicand here is determined by trial an error
    # (but should in theory be base on snr of the signal and log base used in ldpc_codec.c)
    llr_val = soft_val*1024 
    write_data[oi] = llr_val

print("Bits:", len(write_data))

with open(sys.argv[2], "wb") as f:
    f.write(write_data.astype(np.int32).tobytes())
