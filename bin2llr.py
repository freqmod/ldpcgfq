# -*- coding: UTF-8 -*-
import sys, struct

import numpy as np

read_data = None
with open(sys.argv[1],"rb") as fh:
    read_data = fh.read()

write_data = np.zeros(len(read_data)*8, dtype=np.uint32)
for x in xrange(len(read_data)):
  for y in xrange(8):
    v = ord(read_data[x])
    write_data[(x*8)+y] = 128 if ((v >> (7-y)) & 1) else -128

with open(sys.argv[2], "wb") as f:
    f.write(write_data.astype(np.int32).tobytes())
