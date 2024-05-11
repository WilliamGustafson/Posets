from posets import *
import hot
P = Boolean(10)
t = Timer()
hot.cdIndex2(P)
t.stop()
print(t)
