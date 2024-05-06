'''
TODO:
>for specific posets (e.g. Uncrossing posets) can the traditional algorithm beat
the new sum by judiciously choosing which flag f vectors to compute?
>integrate new sum formula into mainline cdIndex method
>is the new sum formula a faster way to compute the ab-index?
	>e.g. for DS posets compute cd-index then convert to ab-index
'''
from posets import *
import poly
import math
import time
import sys

def cdIndex1(P):
	'''
	Computes the cd-index by computing the ab-index and converting it c's and d's.
	'''
	return Polynomial(sorted(P.abIndex().abToCd().data, key=lambda x:x[1]))

def cdIndex3(P):
	'''
	Computes the cd-inndex by compuing the ab-index and converting it to c's ad d's, but like faster this time.
	'''
	flag = flagVectors(P)
	n = len(P.ranks)-2
	def abMonom(S,n):
		ret = []
		for i in range(1,n+1):
			if i in S:
				ret.append('b')
			else:
				ret.append('a')
		return ''.join(ret)
	ab = poly.Polynomial({abMonom(f[0],n):f[2] for f in flag})
	return ab.abToCd()

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
	for i in range(0,len(v)):
		u = v[i:]
		for S in domIdeal(u,1,True):
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
	table = {tuple() : 1}
	ret = []

	m = (n>>1) - (1 if n%2==0 else 0)
	for S in itertools.chain.from_iterable(itertools.combinations(range(1,n+1),k) for k in range(1,m+1)):
		table[S] = fVectorCalc(P.ranks,S,P.incMat,P.ranks[0][0],0)
	del m

	if n%2==0:
		for S in itertools.combinations(range(1,n),n>>1):
			table[S] = fVectorCalc(P.ranks,S,P.incMat,P.ranks[0][0],0)

	def pie(S):
		entry = [S,table[S],0]
		for T in itertools.chain.from_iterable(itertools.combinations(S,k) for k in range(len(S)+1)):
			entry[-1] += table[T] if ((len(S)+len(T))%2==0) else -table[T]
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
	n = len(P.ranks)-2
	flag = {tuple():1}
	###########################################################
	def fVectorCalc(ranks,S,M, i, count):
		newCount = count
		if len(S)==0: return 1
		for j in ranks[S[0]]:
			if M[i][j] == 1:
				newCount += fVectorCalc(ranks, S[1:], M, j, count)
		return newCount

	if len(P.ranks)<=2: return table

	n = len(P.ranks)-2
	if n%2==1:
		v = tuple(i for i in range(2,n+1,2))
	else:
		v = tuple(i for i in range(1,n,2))
	for i in range(0,len(v)):
		u = v[i:]
		for S in domIdeal(u,1,True):
			flag[S] = fVectorCalc(P.ranks,S,P.incMat,P.ranks[0][0],0)
	###########################################################
	FS = fibSets(n,[],1)[1:] #non c^n coefficients
	psi = [] #cd-index

	for W in FS:
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
			v_eqsum = sum(v[i] for i in eq_inds)
			sumv = sum(v)
			v = tuple(sorted(list(set(i for i in v if i!=0))))
			coeff += (flag[v]<<eq_count) if ((sumv+sumW)%2==0) else -(flag[v]<<eq_count)
		if coeff!=0: psi.append([coeff,fibSet_str(W,n)])
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

def main():
	Poset.flagVectors = flagVectors
	n = 6 if len(sys.argv)<2 else int(sys.argv[1])
	P = Boolean(n)

#	P.cache = {}
#	t = time.perf_counter()
#	psi1 = cdIndex1(P)
#	print('flag vector algorithm',time.perf_counter()-t)

	P.cache = {}

	t = time.perf_counter()
	psi2 = cdIndex2(P)
	print('new summation formula',time.perf_counter()-t)
	print('')

	P.cache  = {}
	t = time.perf_counter()
	psi3 = cdIndex3(P)
	print('flag vector algorithm new poly class', time.perf_counter()-t)

#	assert(str(psi1)==str(psi2))
	assert(str(psi2)==str(psi3))

if __name__ == '__main__': main()
