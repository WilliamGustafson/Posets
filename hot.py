from posets import *

def buildIsomorphism(this, that, indices=False):
	d1 = {i: this.rank(i,indices=True) for i in range(len(this))}
	d2 = {i: that.rank(i,indices=True) for i in range(len(that))}
	if collections.Counter(d1.values()) != collections.Counter(d2.values()): return None
	this_ranks = [0 for i in this]
	for r in range(len(this.ranks)):
		for i in this.ranks[r]: this_ranks[i] = r
	that_ranks = [0 for i in that]
	for r in range(len(that.ranks)):
		for i in that.ranks[r]: that_ranks[i] = r
	def get_lengths(l):
		return tuple([len(x) for x in l])
	def set_comp_rks(P,d,ranks):
		for i in range(len(P)):
			row = P.incMat[i]
			upset = frozenset(collections.Counter(ranks[j] for j in range(len(P)) if row[j] == 1).items())
			downset = frozenset(collections.Counter(ranks[j] for j in range(len(P)) if row[j] == -1).items())
			d[i] = (upset, downset) #get_lengths(P.filter([i],indices=True).ranks), get_lengths(P.ideal([i],indices=True).ranks))
		return d
	if 'set_comp_rks()' in this.cache:
		d1 = this.cache['set_comp_rks()']
	else:
		set_comp_rks(this, d1, this_ranks)
		this.cache['set_comp_rks()'] = d1
	if 'set_comp_rks()' in that.cache:
		d2 = that.cache['set_comp_rks()']
	else:
		set_comp_rks(that, d2, that_ranks)
		that.cache['set_comp_rks()'] = d2
	def invert(d):
		ret = {}
		for k in d:
			if d[k] in ret:
				ret[d[k]].append(k)
			else:
				ret[d[k]] = [k]
		return ret
	d1Inv = invert(d1)
	d2Inv = invert(d2)

	def iso(P, Q, map, dP, dQ, dPinv, dQinv, i):
		#all values set return
		if i==len(P): return map
		#candidates are elements that compare to the same number of elements per rank
		cands = dQinv[dP[i]]
		for j in cands:
			#skip if j is already hit
			if j in map.values(): continue
			#i-> is order-preserving
			if all(P.incMat[m[0]][i] == Q.incMat[m[1]][j] for m in map.items()):
				new_map = {i:j}
				new_map.update(map)
				new_map = iso(P,Q,new_map,dP,dQ,dPinv,dQinv,i+1)
				if new_map!=None: return new_map
		return None

	def nextchoice(map, i, jstart):
		#print('nextchoice:')
		#print('\ti',i)
		#print('\tmap',map)
		#print('\tjstart',jstart)
		if i==len(this): return map, -1, True
		cands = d2Inv[d1[i]]
		for j_ in range(jstart,len(cands)):
			j = cands[j_]
			if j in [m[1] for m in map]: continue
			if all(this.incMat[m[0]][i] == that.incMat[m[1]][j] for m in map):
				return map+[[i,j]], j_+1, False
		return None, j_+1, True

	def prevchoice(map, i, jstarts):
		#print('prevchoice:')
		#print('\tmap',map)
		#print('\ti',i)
		#print('\tjstarts',jstarts)
		while i>=0 and jstarts[i] >= len(d2Inv[d1[i]]):
			jstarts[i] = 0
			i -= 1
		if i==-1:
			#print('prevchoice i got to -1, returning None,None,None')
			#print('jstarts',jstarts)
			#print('map',map)
			#print('candidate lengths:',[len(d2Inv[d1[k]]) for k in range(len(this))])
			#print('candidates:',[d2Inv[d1[k]] for k in range(len(this))])
			return None,None,None
		map = [m for m in map if m[0]<i]
#		jstarts[i] += 1
		return map, i, jstarts

	jstarts = [0 for i in range(len(this))]
#	map, map_built = nextchoice([], 0, 0)
	map = []
	map_built = False
	i = 0
	while not map_built and len(map)<len(this):
		#print('*'*14)
		#print('i',i)
		#print('jstarts',jstarts)
		#print('map',map)
		#print('map_built',map_built)
		#print('*'*14)
		new_map, j, map_built = nextchoice(map, i, jstarts[i])
		jstarts[i] = j
		if new_map == None:
			map, i, jstarts = prevchoice(map,i,jstarts)
			map_built = False
			if map==None: map_built = True
		else:
			map = new_map
#			jstarts[i] = j
			i+=1
	map = map

	if map != None:
		if indices:
			map = {m[0] : m[1] for m in map}
		else:
			map = {this[m[0]] : that[m[1]] for m in map}
	return map

def eq(P,Q):
	phi = P.buildIsomorphism(Q)
	if phi==None: return False
	return all(phi[p] == p for p in phi)

def eq2(P,Q):
	if set(P.elements)!=set(Q.elements): return False
	inds = [Q.elements.index(P[i]) for i in range(len(P))]
	return all(P.incMat[i][j] == Q.incMat[inds[i]][inds[j]] for i in range(len(P)) for j in range(i+1,len(P)) )

def testeq(P,Q=None):
	if Q == None: Q = P.copy()

	P.cache={}
	Q.cache={}

	t = Timer()
	ret1 = P==Q
	t.stop()

	P.cache={}
	Q.cache={}

	s = Timer()
	ret2 = eq(P,Q)
	s.stop()

	P.cache={}
	Q.cache={}

	r = Timer()
	ret3 = eq2(P,Q)
	r.stop()

	print('cold:',t,'hot:',s,'hot2:',r)

	return ret1, ret2, ret3

def testiso(P,Q=None):
	if Q==None:
		Q = P.copy().shuffle()
		Q.elements = list(range(len(Q)))

	P.cache = {}
	Q.cache = {}

	t = Timer()
	phi = buildIsomorphism(P,Q)
	t.stop()

	P.cache = {}
	Q.cache = {}

	s = Timer()
	theta = P.buildIsomorphism(Q)
	s.stop()

	r = Timer()
	P == P
	r.stop()

	print('hot:',t,'cold:',s, '__eq__:', r)
	if phi==None: return False
	#print(phi)
	return all(P.less(p1,p2) == Q.less(phi[p1],phi[p2]) for p1 in P for p2 in P),all(P.less(p1,p2) == Q.less(theta[p1],theta[p2]) for p1 in P for p2 in P)# all(P.less(p1,p2,True) == Q.less(theta[p1],theta[p2],True) for p1 in range(len(P)) for p2 in range(len(P)))
