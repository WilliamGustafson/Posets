'''
TODO:
>optimize traditional cdIndex algorithm more
>optimize new sum formula more (don't use exponentiation etc.)
	>maybe consider testing exponentiation versus binary ops
>for specific posets (e.g. boolean algebra) can the traditional algorithm beat
the new sum by judiciously choosing which flag f vectors to compute?
>integrate new sum formula into mainline cdIndex method
>is the new sum formula a faster way to compute the ab-index?
	>e.g. for DS posets compute cd-index then convert to ab-index
'''
from posets import *
import math
import time

def cdIndex1(P):
	'''
	Computes the cd-index by computing the ab-index and converting it c's and d's.
	'''
	return Polynomial(sorted(P.abIndex().abToCd().data, key=lambda x:x[1]))

debug = False

def dprint(*args):
	if debug: print(*args)

def domIdeal(v,minvalue=0,strict=False):
	v = tuple(v)
	n = len(v)-1
	u = v
	offset = 1 if strict else 0
	while True:
		yield u
		found_index = False
		for i in range(n,0,-1):
			if u[i]>u[i-1]+offset:
				found_index = True
				break
		if not found_index:
			if u[0] == minvalue:
				return
			i = 0
		u = u[:i] +(u[i]-1,)+ v[i+1:]

def f(this):
	'''
	Returns some of the flag f-vector given a poset.
	'''
	def fVectorCalc(ranks,S,M, i, count):
		newCount = count
		if len(S)==0: return 1
		for j in ranks[S[0]]:
			if M[i][j] == 1:
				newCount += fVectorCalc(ranks, S[1:], M, j, count)
		return newCount
	table = [[tuple(),1]]

	if len(this.ranks)<=2: return table

	n = len(this.ranks)-2
	if n%2==1:
		v = tuple(i for i in range(2,n+1,2))
	else:
		v = tuple(i for i in range(1,n,2))
	dprint('v',v)
	for i in range(0,len(v)):
		u = v[i:]
		for S in domIdeal(u,1,True):
			dprint('S',S)
			table.append([S,fVectorCalc(this.ranks,S,this.incMat,this.ranks[0][0],0)])
	return table

def flagVectors(P):
	n = len(P.ranks)-2
	def fVectorCalc(ranks,S,M, i, count):
		newCount = count
		if len(S)==0: return 1
		for j in ranks[S[0]]:
			if M[i][j] == 1:
				newCount += fVectorCalc(ranks, S[1:], M, j, count)
		return newCount
	flag = [[tuple(),1]]

	#iterate over all subsets of the ranks not containing n
	for i in range(1,1<<(n-1)):
		#construct the corresponding set S
		pad = 1
		elem = 1
		S = []
		while pad <= i:
			if pad&i:
				S.append(elem)

			pad <<= 1
			elem += 1
		flag.append((tuple(S),fVectorCalc(P.ranks,S,P.incMat,P.ranks[0][0],0)))


	table = {x[0] : x[1] for x in flag}
	ret = []

	def pie(S):
		entry = [S,table[S],0]
		#go over T subset of S
		for T in range(1<<len(S)):
			b = bin(T)[2:][::-1]
			T = tuple(S[i] for i in range(len(b)) if b[i]=='1')
			if 1&(len(T)^len(S))==1:
				entry[-1] -= table[T]
			else:
				entry[-1] += table[T]
		return entry

	for S in table:
		entry = pie(S)
		ret.append(entry)
		ret.append([tuple(i for i in range(1,n+1) if i not in S),entry[1],entry[2]])
	return ret

def fibSets(n, prefix, start):
	if n<=1: return [tuple(prefix)]
	if n==2: return [tuple(prefix),tuple(prefix)+(start,)]
	return fibSets(n-1, prefix, start+1) + fibSets(n-2, prefix+[start], start+2)

def fibSet_str(W,n):
	ret = []
	if len(W)==0: return 'c'*n
	for i in W:
		while len(ret)+1 < i: ret.append('c')
		ret.append('d')
		ret.append('d')
	ret.append('c'*(n-len(ret)))
	return ''.join(ret).replace('dd','d')

def cdIndex2(P):
	'''
	Computes the cd-index using the sum in corollary 4.3.
	'''
	global debug
	n = len(P.ranks)-2
	flag_ = f(P)
	flag = {}
	dprint('flag_',flag_)
	for x,y in flag_: flag[tuple(x)] = y
	dprint('flag',flag)
	FS = fibSets(n,[],1)[1:] #non c^n coefficients
	psi = [] #cd-index

	for W in FS:
#		debug=W==(2,4)
		dprint('W',W)
		coeff = 0
		for v in domIdeal(W):
			sumW = sum(W)
			skip = False
			eq_count = 0
			eq_inds=[]
			if v[0]==0:
				if W[0]%2==0: continue
				eq_count = 1
				eq_inds+=[0]
			for i in range(1,len(v)):
				if v[i]==v[i-1]:
					if(v[i]+W[i])%2==0:
						skip = True
						break
					eq_count += 1
					eq_inds+=[i]
			if skip: continue
			dprint('v',v)
			v_eqsum = sum(v[i] for i in eq_inds)
			sumv = sum(v)
			v = tuple(sorted(list(set(i for i in v if i!=0))))
			dprint('adding term', flag[v] * (2)**(eq_count) * (-1)**(sum(v)+sumW))
			dprint('flag[v]',flag[v])
			dprint('eq_count',eq_count)
			dprint('sum(v)',sum(v))
			dprint('sumW',sumW)
#			coeff += flag[v] * (-2)**(eq_count) * (-1)**(sum(v)+sumW+sum(W[i] for i in eq_inds)+v_eqsum)
			coeff += flag[v] * (2)**(eq_count) * (-1)**(sumv+sumW)
		if coeff!=0: psi.append([coeff,fibSet_str(W,n)])
	dprint('psi',psi)
	return Polynomial(psi+[[1,'c'*n]])

def test(P):
	P.cache={}

	t = Timer()
	psi = cdIndex1(P)
	t.stop()

	P.cache={}

	s = Timer()
	phi = cdIndex2(P)
	s.stop()

	print('same result',phi==psi)
	print('cdIndex1 time:',t)
	print('cdIndex2 time:',s)
	x=float(str(t))
	y=float(str(s))
	print('ratio:',x/y)

def bool(n=3,m=10):
	T = [] #times
	S = []
	for k in range(n,m):
		P = Cube(k)
		P.cache = {}

		t = time.perf_counter()
		cdIndex1(P)
		T.append(time.perf_counter()-t)

		P.cache = {}

		s = time.perf_counter()
		cdIndex2(P)
		S.append(time.perf_counter()-s)

	return T,S

