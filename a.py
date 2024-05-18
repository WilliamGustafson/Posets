from posets import *
import hot

def f(n):
	v = tuple(i for i in range(1,2*n,2))
	S = set()
	for i in range(0,len(v)):
		for x in hot.domIdeal(v[i:],1,True): S.add(x)
	return len(S)+1
