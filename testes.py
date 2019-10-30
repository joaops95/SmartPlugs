import random
import struct

floatlist = [random.random() for _ in range(10**4)]
print(floatlist)
buf = struct.pack('%sf' % len(floatlist), *floatlist)
print(buf)
floats = list(struct.unpack('%sf' % len(floatlist), buf))
print(len(floatlist))
print(len(buf))