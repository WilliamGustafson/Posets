from .poset import Poset
from .hasseDiagram import HasseDiagram
import itertools

def Empty():
	r'''
	@section@Built in posets@
	Returns an empty poset.
	'''
	return Poset(elements = [], ranks = [], incMat = [])

def Weak(n):
	r'''
	@section@Built in posets@
	Returns the type $A_{n-1}$ weak order (the symmetric group $S_n$).

	\begin{center}
		\includegraphics{figures/weak_3.pdf}

		The poset \verb|Weak(3)|.
	\end{center}
	@exec@
	make_fig(Weak(3),'weak_3',height=4,width=3)
	'''
	def covers(p):
		ret=[]
		for i in range(n-1):
			if p[i]<p[i+1]:
				ret.append(p[:i]+(p[i+1],p[i])+tuple([p[j] for j in range(i+2,n)]))
		return ret
	return Poset(relations={p:covers(p) for p in itertools.permutations(range(1,n+1))})

def Bruhat(n):
	r'''
	@section@Built in posets@
	Returns the type $A_{n-1}$ Bruhat order (the symmetric group $S_n$).

	\begin{center}
		\includegraphics{figures/Bruhat_3.pdf}

		The poset \verb|Bruhat(3)|.
	\end{center}
	@exec@
	make_fig(Bruhat(3), 'Bruhat_3',height=4, width=3)
	'''
	def pairing_to_perm(tau):
		arcs = [[int(x) for x in a.split(',')] for a in tau[1:-1].split(')(')]

		arcs.sort(key = lambda x: x[0])

		return tuple(a[1]-n for a in arcs)

	def nodeLabel(hasseDiagram, i):
		return ('' if n<=9 else ',').join([str(x) for x in hasseDiagram.P.elements[i]])

	t = []
	for i in range(1,n+1):
		t.append(i)
		t.append(2*n+1-i)
	P = Uncrossing(t, upper=True)
	P.hasseDiagram.nodeLabel = nodeLabel
	P.hasseDiagram.offset = 1

	P.elements = [pairing_to_perm(e) for e in P.elements]
	P.name = "Type A_"+str(n)+" Bruhat order"

	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P
def Root(n=3):
	r'''
	@section@Built in posets@
	Returns the type $A_{n+1}$ root poset.

	\begin{center}
		\includegraphics{figures/root_3.pdf}

		The poset \verb|Root(3)|.
	\end{center}

	@exec@
	make_fig(Root(3).reorder(range(2,-1,-1),True),'root_3',height=3,width=3)
	'''
	def covers(i,j):
		if i==1:
			if j==n:
				return []
			return [(1,j+1)]
		elif j==n:
			return [(i-1,n)]
		return [(i-1,j),(i,j+1)]
	return Poset(relations={(i, j) : covers(i,j) for i in range(1, n) for j in range(i+1, (n+1 if i>1 else n))})

def Butterfly(n):
	r'''
	@section@Built in posets@
	Returns the rank $n+1$ bounded poset where ranks $1,\dots,n$ have two elements and all comparisons between ranks.
	\begin{center}
		\includegraphics{figures/Butterfly_3.pdf}

		The poset \verb|Butterfly(3)|.
	\end{center}
	@exec@
	make_fig(Butterfly(3),'Butterfly_3',height=5,width=2)
	'''
	elements = [('a' if i%2==0 else 'b')+str(i//2) for i in range(2*n)]
	ranks = [[i,i+1] for i in range(0,2*n,2)]
	incMat = [[0]*(i//2+1)*2 + [1]*(len(elements)-((i//2+1)*2)) for i in range(len(elements))]
	name = "Rank "+str(n+1)+" butterfly poset"

	P = Poset(incMat, elements, ranks, name = name).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

def Antichain(n):
	r'''
	@section@Built in posets@
	Returns the poset on $1,\dots,n$ with no relations.

	\begin{center}
		\includegraphics{figures/antichain_3.pdf}

		The poset \verb|Antichain(3)|.
	\end{center}

	@exec@
	make_fig(Antichain(3),'antichain_3',height=3,width=3)
	'''
	return Poset(elements=list(range(1,n+1)))

def Chain(n):
	r'''
	@section@Built in posets@
	Returns the poset on $0,\dots,n$ ordered linearly (i.e. by usual ordering of integers).

	\begin{center}
		\includegraphics{figures/chain_3.pdf}

		The poset \verb|Chain(3)|.
	\end{center}

	@exec@
	make_fig(Chain(3),'chain_3',height=5,width=3)
	'''
	elements = list(range(n+1))
	ranks = [[i] for i in elements]
	import operator
	P = Poset(elements=elements, ranks=ranks, less=operator.lt, name = "Length "+str(n)+" chain", trans_close=False)
	#cach some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=False
	P.cache['isGorenstein()']=False
	return P

def Boolean(n, X=None):
	r'''
	@section@Built in posets@
	Returns the poset of subsets of $X$, default is $\{1,\dots,n\}$, ordered by inclusion.

	\begin{center}
		\includegraphics{figures/Boolean_3.pdf}

		The poset \verb|Boolean(3)|.
	\end{center}

	@exec@
	make_fig(Boolean(3),'Boolean_3', height=6, width=4)
	'''
	P = Poset()
	P.elements = list(range(1<<n))
	P.incMat = [[1 if i&j==i else 0 for j in P.elements] for i in P.elements]
	P.incMat = [[P.incMat[i][j] - P.incMat[j][i] for j in range(len(P))] for i in range(len(P))]
	P.ranks = [[] for _ in range(n+1)]
	for p in P.elements:
		P.ranks[len([c for c in bin(p) if c=='1'])].append(p) #p==P.elements.index(p)
	P.elements = [bin(e)[2:][::-1]+('0'*(n-len(bin(e)[2:]))) for e in P.elements]
	P.elements = [tuple([i+1 for i in range(len(e)) if e[i]=='1']) for e in P.elements]
	P.name = "Rank "+str(n)+" Boolean algebra"

	def nodeLabel(hasseDiagram, i):
		S = hasseDiagram.P.elements[i]
		s = str(S).replace(',','') if len(S) <= 1 else str(S)
		return s.replace('(','\\{' if hasseDiagram.in_latex else '{').replace(')','\\}' if hasseDiagram.in_latex else '}').replace(',',', ')
	P.hasseDiagram.nodeLabel = nodeLabel
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True

	if X!=None:
		X = list(X)
		for i in range(len(P)):
			P.elements[i] = tuple([X[j-1] for j in P[i]])
	return P

def Simplex(n):
	r'''
	@section@Built in posets@
	Returns Boolean(n+1) the face lattice of the $n$-dimensional simplex.
	'''
	return Boolean(n+1)

def Polygon(n):
	r'''
	@section@Built in posets@
	Returns the face lattice of the $n$-gon.

	\begin{center}
		\includegraphics{figures/polygon_4.pdf}

		The poset \verb|Polygon(4)|.
	\end{center}

	@exec@
	make_fig(Polygon(4),'polygon_4',height=6,width=5)
	'''
	elements = []
	for i in range(n//2):
		elements.append(i+1)
		elements.append(n-i)
	if n%2 == 1:
		elements.append(n//2+1)
	edges = sorted([(i,i+1) for i in range(1,n)]+[(1,n)], key = lambda e: sorted([elements.index(x) for x in e])[::-1] )
	elements += edges
	def less(i,j):
		return type(j)==tuple and i in j
	def nodeLabel(hasseDiagram, i):
		e = hasseDiagram.P.elements[i]
		if e in hasseDiagram.P.max(): return "$\\widehat{1}$" if hasseDiagram.in_latex else '1'
		if e in hasseDiagram.P.min(): return "$\\widehat{0}$" if hasseDiagram.in_latex else '0'
		if type(e) == int: return str(e)
		return str(e).replace(',',', ')
	P = Poset(elements = elements, less = less, name = str(n)+"-gon face lattice", nodeLabel = nodeLabel, trans_close=False).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

def Cube(n):
	r'''
	@section@Built in posets@
	Returns the face lattice of the $n$-dimensional cube.

	\begin{center}
		\includegraphics{figures/cube_2.pdf}

		The poset \verb|Cube(2)|.
	\end{center}

	@exec@
	make_fig(Cube(2),'cube_2',height=6,width=5)
	'''
	def expand(E):
		return [e+'0' for e in E]+[e+'1' for e in E]+[e+'*' for e in E]

	def less(x,y):
		return x!=y and all([x[i]==y[i] or y[i]=='*' for i in range(len(x))])

	E = ['']
	for i in range(n): E = expand(E)
	P = Poset()
	P.elements = E
	P.incMat = [[1 if less(x,y) else -1 if less(y,x) else 0 for y in P.elements] for x in P.elements]
	P.ranks = [[] for _ in range(n+1)]
	for p in P.elements:
		P.ranks[len([c for c in p if c=='*'])].append(P.elements.index(p))
	P.name = str(n)+"-cube face lattice"

	def sort_key(F): #revlex induced by 0<*<1
		return ''.join(['1' if f == '*' else '2' if f == '1' else '0' for f in F][::-1])

	P = P.sort(sort_key).adjoin_zerohat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

def Torus(n=2, m=2):
	r'''
	@section@Built in posets@
	Returns the face poset of a cubical complex homeomorphic to the $n$-dimensional Torus.

	This poset is isomorphic to the Cartesian product of $n$ copies of $P_m$ with minimum and maximum adjoined
	where $P_m$ is the face lattice of an $m$-gon with its minimum and maximum removed.

	Let~$\ell_m$ be the $m$th letter of the alphabet.
	When $m\le 26$ the set is $\{0,1,\dots,m-1,A,B,\dots,\ell_m\}^n$ and otherwise is $\{0,\dots,m-1,*0,\dots*[m-1]\}^n$.
	The order relation is
	componentwise where $0<A,\ell_m\ 1<A,B\ \dots\ m-1<\ell_{m-1},\ell_m$  for $m\le26$, and $0<*1,*2\ \dots\  m-1<*[m-1],*0$ for $m>26$.

	\begin{center}
		\includegraphics{figures/torus.pdf}

		The poset \verb|Torus(2,2)|.
	\end{center}

	@exec@
	make_fig(Torus(),'torus',height=6,width=6)
	'''
	if m<=26:
		symbols = [str(i) for i in range(m)]+[chr(i+ord('A')) for i in range(m)]
		elements = [''.join(tuple(x)) for x in itertools.product(symbols, repeat=n)]

		def less(e,f):
			if e == f: return False
			for i in range(len(e)):
				if e[i]==f[i]: continue
				if not e[i].isdigit(): return False
				ei = int(e[i])
				if f[i] != chr(ord('A')+ei):
					j = (ei-1)%m
					if f[i]!=chr(ord('A')+j):
						return False
			return True

		def rk(e):
			return len([x for x in e if not x.isdigit()])
	else:
		symbols = [str(i) for i in range(m)]+['*'+str(i) for i in range(m)]
		elements = [tuple(x) for x in itertools.product(symbols, repeat=n)]

		def less(e,f):
			return e!=f and all([e[i] == f[i] or ('*' not in e[i] and f[i] in ('*'+e[i], '*'+str((int(e[i])-1)%m)) ) for i in range(len(e))])

		def rk(e):
			return len([x for x in e if '*' in x])

	ranks = [[] for i in range(n+1)]
	for e in elements: ranks[rk(e)].append(elements.index(e))

	#build order on symbols for sort_key
	verts = ['0']
	for i in range(1,m//2):
		verts.append(str(i))
		verts.append(str(m-i))

	verts.append(str(m//2))
	if m%2 == 1:
		verts.append(str(m//2+1))
	preedges = [s for s in symbols if s not in verts]
	edges = []
	edges.append(preedges[0])
	for i in range(1,m//2):
		edges.append(preedges[i])
		edges.append(preedges[m-i])
	edges.append(preedges[m//2])
	if m%2 == 1:
		edges.append(preedges[m//2+1])

	order = []
	for i in range(len(verts)//2):
		order.append(verts[i])
		order.append(edges[i])
	for i in range(len(verts)//2,len(verts)):
		order.append(edges[i])
		order.append(verts[i])

	def sort_key(F): #revlex induced by 0 < A < B < 1
		return tuple([order.index(f) for f in F][::-1])

	P = Poset(less=less, ranks=ranks, elements=elements, name = str(m)+" subdivided "+str(n)+"-torus").sort(sort_key).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']= n%2 == 1
	P.cache['isGorenstein()']= n == 1
	return P

def GluedCube(orientations = None):
	r'''
	@section@Built in posets@
	Returns the face poset of the cubical complex obtained from a $2\times\dots\times2$ grid of $n=\verb|len(orientations)|$-cubes via a series of gluings as indicated by the parameter \verb|orientations|.

	If \verb|orientations| is \verb|[1,...,1]| the $n$-torus is constructed and if \verb|orientations| is \verb|[-1,...,-1]| the
	projective space of dimension $n$ is constructed.


	If \verb|orientations[i] == 1| the two ends of the large cube are glued so that points with the same
	image under projecting out the $i$th coordinate are identified.

	If \verb|orientations[i] == -1| points on the two ends of the large cube are identified with their antipodes.

	If \verb|orientations[i]| is any other value no gluing is performed for that component.

	\begin{center}
		\includegraphics{figures/gluedcube.pdf}

		The poset \verb|GluedCube([-1,1])|.
	\end{center}

	@exec@
	make_fig(GluedCube([-1,1]),'gluedcube',height=8,width=15)
	'''
	#2-torus by default
	if orientations == None:
		orientations = (1,1)
	n = len(orientations)
	P = Grid(n)
	#do bddy gluings
	gluings = {}
	gluings_inv = {}
	nonreprs=[]
	for (F, eps) in P.complSubposet(P.min()):
#		if (F,eps) in nonreprs: continue #this didn't glue all of the Torus verts
		for i in range(len(F)):
			if F[i] == '0' and F[i] == str(eps[i]):
				if orientations[i] == 1:
					G = F[:i]+'1'+F[i+1:]
					nu = eps[:i]+(1,)+eps[i+1:]
				if orientations[i] == -1:
					G = ''.join(['*' if f == '*' else '1' if f == '0' else '0' for f in F])
					nu = tuple([1 if e == 0 else 0 for e in eps])
				#make sure (G,nu) is a representative in P
				for j in range(len(G)):
					if G[j] == '0' and nu[j] == 1:
						nu = nu[:j]+(0,)+nu[j+1:]
						G = G[:j]+'1'+G[j+1:]

				if (F,eps) in nonreprs:
					k = gluings_inv[(F,eps)]
					gluings[k].append((G,nu))
					gluings_inv[(G,nu)] = k
				else:
					if not (F,eps) in gluings:
						gluings[(F,eps)] = []
					gluings[(F,eps)].append((G,nu))
					gluings_inv[(G,nu)] = (F,eps)
				nonreprs.append((G,nu))
	for k in gluings:
		gluings[k] = list(set(gluings[k]))
	P = P.identify(gluings).adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	return P

def KleinBottle():
	r'''
	@section@Built in posets@
	Returns the face poset of a cubical complex homeomorphic to the Klein Bottle.

	Pseudonym for \verb|GluedCube([-1,1])|.
	'''
	P = GluedCube([-1,1])
	P.name = "Klein Bottle"
	return P

def ProjectiveSpace(n=2):
	r'''
	@section@Built in posets@
	Returns the face poset of a Cubical complex homeomorphic to projective space of dimension $n$.

	Pseudonym for \verb|GluedCube([-1,...,-1])|.

	\begin{center}
		\includegraphics{figures/projectiveSpace.pdf}

		The poset \verb|ProjectiveSpace(2)|.
	\end{center}

	@exec@
	make_fig(ProjectiveSpace(), 'projectiveSpace', height=8, width=15)
	'''
	P = GluedCube([-1]*n)
	P.name = str(n)+"-dimensional projective space"
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']= n%2 == 1
	P.cache['isGorenstein()']= n%2 == 1
	return P

	return P

def Grid(n=2,d=None):
	r'''
	@section@Built in posets@
	Returns the face poset of the cubical complex that forms a $\verb|d[0]|\times\dots\times\verb|d[-1]|$ grid of $n$-cubes.

	\begin{center}
		\includegraphics{figures/grid.pdf}

		The poset \verb|Grid(2,[1,1])|.
	\end{center}

	@exec@
	make_fig(Grid(2,[1,2]),'grid',height=8,width=15)
	'''
	if d == None: d = [2]*n
	cube = Cube(n).complSubposet([0])
	Gamma = Empty()
	identifications = {}
	#build the complex as a union and the list of gluings
	for i in itertools.product(*[range(j) for j in d]):
		elements = [(F, i) for F in cube]
		Gamma = Gamma.union(Poset(elements = elements, incMat = cube.incMat, ranks = cube.ranks))

		for j in range(n):
			if i[j] == d[j]-1: continue
			#glue stuff in i cube with jth part 1
			#to stuff in i+e_j cube with jth part 0
			i2 = [x for x in i]
			i2[j] += 1
			i2 = tuple(i2)
			for F in cube:
				if F[j] == '1':
					if not (F, i) in identifications:
						identifications[(F,i)] = []
					identifications[(F,i)].append((F[:j]+'0'+F[j+1:], i2))
	P = Gamma.identify(identifications).adjoin_zerohat()
	P.name = "x".join([str(x) for x in d])+" "+str(n)+"-cubical grid"

	def sort_key(x):
		if type(x) == int: return ((x,),)
		F,eps = x
		ret = [eps[::-1]]
		ret.append(tuple(['0*1'.index(f) for f in F][::-1]))
		return tuple(ret)
	P = P.sort(key = sort_key)
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']= all([x==1 for x in d])
	P.cache['isGorenstein()']= all([x == 1 for x in d])
	return P


#copied from uncrossing.py
#from https://github.com/WilliamGustafson/cdIndexCalculator
def Uncrossing(t, upper=False):
	r'''
	@section@Built in posets@
	Returns either a lower interval $[\widehat{0},t]$ or the upper interval $[t,\widehat{1}]$ in the uncrossing poset.

	The parameter \verb|t| should be either a pairing encoded as a list \verb|[s_1,t_1,...,s_n,t_n]| where
	\verb|s_i| is paired to \verb|t_i| or \verb|t| can be an integer greater than 1. If t is an integer the entire uncrossing
	poset is returned.

	For more info on the uncrossing poset see: https://arxiv.org/abs/1406.5671

	\begin{center}
		\includegraphics{figures/uc.pdf}

		The poset \verb|Uncrossing(3)==Uncrossing([1,4,2,5,3,6])|.
	\end{center}

	@exec@
	make_fig(Uncrossing(3), 'uc', height=15, width=15, nodescale=0.75)
	'''
	#Throughout pairings are encoded as lists of numbers, each number encodes
	#a pair as two bits set. For example the pairing {{1,3},{2,4}} is encoded
	#as [2**0+2**2,2**1+2**3]=[5,10]
	if type(t) == int:
		n = t
		t = []
		for i in range(1,n+1):
			t.append(i)
			t.append(i+n)
	#converts the pairing given in the input into the internal format described above
	def readPairing(input):
		t = []
		for i in range(0,len(input)//2):
			t.append(1<<(int(input[i<<1])-1)|1<<(int(input[(i<<1)+1])-1))
		return sorted(t)

	def setFormat(x):
		ret = []
		i = 1
		while x!= 0:
			if x&1: ret.append(str(i))
			x >>= 1
			i += 1
		return "("+",".join(ret)+")"

	def pairingFormat(x):
		return "".join([setFormat(y) for y in x])


	#swaps i and j in the pairing p
	def swap(p,i,j):
	#	return sorted([(x^((((x&(1<<i))>>i)^((x&(1<<j))>>j))<<i)^((((x&(1<<i))>>i)^((x&(1<<j))>>j))<<j)) for x in p])
		ret = []
		for arc in p:
			if (arc&(1<<i))>>i != (arc&(1<<j))>>j: #arc contains one of i and j
				ret.append(arc ^ ((1<<i)|(1<<j))) #swap i and j in the arc
			else: #arc contains both i and j or neither so don't swap i and j
				ret.append(arc)
		return sorted(ret)

	#returns the number of crossings of p
	def c(p):
		ret = 0
		for i in range(0,len(p)):
			xi = bin(p[i])[::-1]
			Ni = xi.find('1')
			Ei = xi.rfind('1')
			for j in range(i+1,len	(p)):
				xj = bin(p[j])[::-1]
				Nj = xj.find('1')
				Ej = xj.rfind('1')
				if (Ni - Nj > 0) == (Ei - Ej > 0) == (Nj - Ei > 0): ret += 1

		return ret

	#computes the upper/lower interval generated by the given pairing t
	#or if t is an integer computes the uncrossing poset of given order
	#returns a tuple (P,ranks,M) which is the list of elements, the rank list and the incidence matrix
	def lowerOrderIdeal(t):
		epsilon = 1 if upper else -1
		if not upper and c(t)==0: return [t],[[1],[0]],[[0,-1],[1,0]]

		P=[t]
		ranks = [[0]] #this is built up backwards for convenience and reversed before returning
		M=[[0]]

		num = 1 #index in to P of next element to add
		level = [t] #list of current rank to expand in next step
		leveli = [0] #indices in to P of the elements of level
		newLevel = [] #we build level for the next step during the current step here
		newLeveli = [] #indices in to P for the next step
		newRank = [] #the new rank indices to add
		while len(level) > 0:
			for i in range(0,(len(t)<<1)-1): #iterate over all pairs we can uncross
				for j in range(i+1,len(t)<<1):
					for k in range(0,len(level)): #do the uncross
						temp = swap(level[k],i,j)
						c_temp = c(temp)
						if c_temp != c(level[k])+epsilon: continue
						if temp in P:
							M[P.index(temp)][leveli[k]]=1
							continue
						P.append(temp)
						newRank.append(num)
						if c_temp > 0: #if not minimal continue uncrossing
							newLevel.append(temp)
							newLeveli.append(num)
						num+= 1

						for x in M: x.append(0)
						M.append([0 for x in range(0,len(M[0]))])
						M[-1][leveli[k]]=1

			level = newLevel
			newLevel = []
			leveli = newLeveli
			newLeveli = []
			ranks.append(newRank)
			newRank = []

		ranks = ranks[::-1]
		Poset.transClose(M)
		return P,ranks,M

	if isinstance(t,int):
		n = t
		t = []
		for i in range(1,n+1):
			t.append(i)
			t.append(i+n)
		name = "Rank "+str(n*(n-1)/2+1)+" uncrossing poset"
	else:
		t = readPairing(t)
	n = len(t)
	P,ranks,M = lowerOrderIdeal(t)
	pairings = P
	P = [pairingFormat(p) for p in P]
	if upper: ranks = ranks[1:]
	if 'name' not in locals():
		name = "Interval ["+(str(pairingFormat(t)) if upper else '0')+","+('1' if upper else str(pairingFormat(t)))+"] in the rank "+str(len(ranks))+" uncrossing poset"

	class UncrossingHasseDiagram(HasseDiagram):

		def __init__(this, P, **kwargs):
			super().__init__(P, **kwargs)
			if 'bend' in kwargs: this.bend = kwargs['bend']
			else: this.bend = '0.5'
			if 'nodetikzscale' in kwargs: this.nodetikzscale = kwargs['nodetikzscale']
			else: this.nodetikzscale = '1'
			if 'offset' in kwargs: this.offset = kwargs['offset']
			else: this.offset = 13
			this.pairings = pairings
			this.n = len(pairings[0])

		def latex(this, **kwargs):
			if 'bend' in kwargs: this.bend = kwargs['bend']
			return super().latex(**kwargs)

		def nodeLabel(this,i):
			if this.in_tkinter:
				return str(this.P[i])
			if P[i]==0: return "\\scalebox{2}{$\\widehat{0}$}"
			i = i-1 #zerohat gets added first so shift back
			ret=["\\begin{tikzpicture}[scale="+this.nodetikzscale+"]\n\\begin{scope}\n\t\\medial\n"]
			for arc in [[float(i) for i in range(0,this.n<<1) if (1<<i)&x!=0] for x in this.pairings[i]]:
				ret.append('\t\\draw('+str(int(arc[0]+1))+')..controls+(')
				ret.append(str((arc[0])*(-360.0)/(this.n<<1)-90))
				ret.append(':\\r*'+this.bend+')and+(')
				ret.append(str((arc[1])*(-360.0)/(this.n<<1)-90))
				ret.append(':\\r*'+this.bend+')..('+str(int(arc[1]+1))+');\n')
			return ''.join(ret+["\\end{scope}\\end{tikzpicture}"])

		def nodeName(this,i):
			if this.P[i] == 0: return 'z'
			i = i-1 #zerohat gets added first so shift back
			p=this.pairings[i]
			n=len(p)
			return ''.join([str(j+1) for k in range(0,n) for j in range(0,n<<1) if (1<<j)&p[k]!=0])

		def nodeDraw(this, i):
			size = 10*int(this.nodescale)
			ptsize = this.ptsize if type(this.ptsize)==int else int(this.ptsize[:-2])
			x = float(this.loc_x(this, i))*float(this.scale)+float(this.scale)*float(this.width)/2+float(this.padding)
			y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this, i))*float(this.scale)+float(this.padding))
			if this.P[i] == 0:
				this.canvas.create_text(x,y,text='0')
				return
			this.canvas.create_oval(x-size, y-size, x+size, y+size)

			def pt(i):
				px = x + int(this.nodescale)*math.cos(i*2*math.pi/(2*n)+math.pi/2)*10
				py = y + int(this.nodescale)*math.sin(i*2*math.pi/(2*n)+math.pi/2)*10
				return px, py
			for j in range(2*n):
				px,py = pt(j)
				this.canvas.create_oval(px-ptsize, py-ptsize, px+ptsize, py+ptsize, fill='black')

			tau = [x.split(',') for x in this.P[i].replace('(','').split(')')[:-1]]

			for j in range(n):
				s = int(tau[j][0])
				t = int(tau[j][1])
				this.canvas.create_line(*pt(s),*pt(t))
			return



	extra_packages = "\\def\\r{1}\n\\def\\n{"+str(n<<1)+"}\n\\newcommand{\\medial}{\n\\draw circle (\\r);\n\\foreach\\i in{1,...,\\n}\n\t{\n\t\\pgfmathsetmacro{\\j}{-90-360/\\n*(\\i-1)}\n\t\\fill (\\j:-\\r) circle (2pt) node [anchor=\\j] {$\\i$};\n\t\\coordinate (\\i) at (\\j:-\\r);\n\t}\n}"
	P = Poset(M, P, ranks, name = name, hasse_class = UncrossingHasseDiagram, extra_packages = extra_packages)
	if not upper:
		P = P.adjoin_zerohat()
		P.hasseDiagram = UncrossingHasseDiagram(P, extra_packages=extra_packages)
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

def Bnq(n=2, q=2):
	r'''
	@section@Built in posets@
	Returns the poset of subspaces of the vector space $\F_q^n$ where $\F_q$ is the field with q elements.

	Currently only implemented for q a prime. Raises a not implemented error if q is not prime.

	\begin{center}
		\includegraphics{figures/Bnq.pdf}

		The poset \verb|Bnq(3,2)|.
	\end{center}

	@exec@
	make_fig(Bnq(3,2),'Bnq',height=6,width=8)
	'''
	def isprime(x):
		d = 2
		while d*d <= x:
			if x%d == 0:
				return false
			d += 1
		return True
	if not isprime(q):
		raise NotImplementedError("Bnq with nonprime q is not implemented")
	#does dot product
	def dot(v,w):
		vmodqi=v%q
		wmodqi=w%q
		ret=(vmodqi*wmodqi)
		qj=q
		qi=1
		vmodqj=vmodqi
		wmodqj=wmodqi
		for i in range(1,n):
			qj*=q
			qi*=q
			vmodqi=vmodqj
			wmodqi=wmodqj
			vmodqj=v%qj
			wmodqj=w%qj
			ret+=((vmodqj-vmodqi)*(wmodqj-wmodqi))/(qi*qi)
		return ret%q

	#turns a number into a list
	def vec(v):
		ret=[v%q]
		qi=1
		qj=q
		for i in range(1,n):
			qi*=q
			qj*=q
			ret.append(int((v%qj-v%qi)/qi))
		return ret

	#compute all hyperplanes
	#hyperplanes are represented as numbers in range(0,2**(q**n)-1)
	#the 1-bits set indicate the elements
	hyperplanes=[]
	qn = q**n
	for v in range(1,qn):
		H=0
		for w in range(0,qn):
			if dot(v,w)==0: H|=(1<<w)
		hyperplanes.append(H)

	#Do intersection of hyperplanes to fill out spaces
	spaces=set([(1<<(qn))-1]+hyperplanes) #first term is whole space
	newspaces=hyperplanes
	while len(newspaces)>0:
		newnewspaces=set([])
		for S in newspaces:
			for H in hyperplanes:
				if S&H!=S: newnewspaces.add(S&H)
		spaces=spaces.union(newnewspaces)
		newspaces=newnewspaces
#	lengths=[[]for i in range(0,n+1)]
#	for S in spaces: lengths[int(math.log(len([j for j in range(qn) if (1<<j)&S!=0]),q))].append(S)

	spaces=sorted(list(spaces))
#	for i in range(0,len(lengths)): lengths[i]=sorted(list(lengths[i]))
	return Poset(elements = spaces, less = lambda i,j: i!=j and i&j == i)

def DistributiveLattice(P, indices=False):
	r'''
	@section@Built in posets@
	Returns the lattice of ideals of a given poset.

	\begin{center}
		\includegraphics{figures/DL.pdf}

		The poset \verb|DistributiveLattice(Root(3))|.
	\end{center}

	@exec@
	make_fig(DistributiveLattice(Root(3)),'DL',height=6,width=4)
	'''
	#make principal ideals
	M = P.incMat
	irr=[]
	for i in range(0,len(M)):
		x=1<<i
		for j in range(0,len(M[i])):
			if M[i][j]==-1: x|=1<<j
		irr.append(x)
	#add all unions to make distr lattice
	ideals=[0]+[i for i in irr]
	new=[i for i in irr]
	while len(new)>0:
		last=new
		new=[]
		for l in last:
			for i in irr:
				x=l|i
				if x not in ideals:
					ideals.append(x)
					new.append(x)
	ranks=[[] for i in range(0,len(M)+1)]
	if indices:
		elements = ideals
		def less(I, J):
			return I!=J and I&J==I
	else:
		elements = []
		for I in ideals:
			ranks[len([c for c in bin(I) if c=='1'])].append(I)
			elements.append(tuple(P[i] for i in range(len(P)) if (1<<i)&I!=0))
		def less(I,J):
			return I!=J and all(i in J for i in I)
	def irr_nodeName(this, i):
		return 'irr_'+HasseDiagram.nodeName(this,i)
	def make_node_options(S):
		def node_options(this, i):
			if i in S:
				return 'color=black'
			return 'color=gray'
		return node_options
	def make_line_options(S):
		def line_options(this,i,j):
			if i in S and j in S:
				return 'color=black'
			return 'color=gray'
		return line_options
	class DistributiveHasseDiagram(HasseDiagram):
		def __init__(this,JP,P,indices=False,**kwargs):
			super().__init__(JP,**kwargs)
			this.Irr = P
			#everything in HasseDiagram uses indices so we need to store ideals as lists of indices
			if not indices:
				this.P.elements = [[this.Irr.elements.index(e) for e in J] for J in this.P.elements]
		def latex(this, **kwargs):
			irrArgs = {k[4:]:v for k,v in kwargs.items() if k[:4]=='irr_'}
			irrDefaults = this.Irr.hasseDiagram.__dict__.copy()
			this.Irr.hasseDiagram.__dict__.update(irrArgs)
			this.Irr.hasseDiagram.nodeName = irr_nodeName

			ret = super().latex(**kwargs)

			this.Irr.hasseDiagram.__dict__.update(irrDefaults)
			return ret

		def nodeLabel(this, i):
			idealLatex = this.Irr.latex(node_options = make_node_options(this.P[i]), line_options = make_line_options(this.P[i]))
			idealLatex = ''.join(idealLatex.split('\n')[2:-1])
#			idealLatex = idealLatex[len('\\begin{tikzpicture}') : -1*len('\\end{tikzpicture}')]
			return '\\begin{tikzpicture}\\begin{scope}\n'+idealLatex+'\n\\end{scope}\\end{tikzpicture}'

	JP = Poset(elements = elements, less = less)
	JP.hasseDiagram = DistributiveHasseDiagram(JP,P,indices)
	return JP

#def SignedBirkhoff(P):
#	D = DistributiveLattice(P, indices=True)
#	def maximal(I):
#		elems = [i for i in range(1<<len(P)) if (1<<i)&I!=0]
#		return [i for i in elems if all(P.incMat[i][j]!=1 for j in elems]
#	achains = [maximal(I) for I in D]
#


def LatticeOfFlats(data):
	r'''
	@section@Built in posets@
	Returns the lattice of flats given either a list of edges of a graph or the rank function of a (poly)matroid.

	When the input represents a graph it should be in the format \verb|[[i_1,j_1],...,[i_n,j_n]]|
	where the pair \verb|[i_k,j_k]| represents an edge between \verb|i_k| and \verb|j_k| in the graph.

	When the input represents a (poly)matroid the input should be a list of the ranks of
	sets ordered reverse lexicographically (i.e. binary order). For example, if f is the
	rank function of a (poly)matroid with ground set size 3 the input should be
		\[
		\verb|[f({}),f({1}),f({2}),f({1,2}),f({3}),f({1,3}),f({2,3}),f({1,2,3})]|.
		\]

	This function may return a poset that isn't a lattice if
	the input function isn't submodular or a preorder that isn't a poset if the input
	is not order-preserving.

	\begin{center}
		\includegraphics{figures/lof_triangle.pdf}

		The poset \verb|LatticeOfFlats([[1,2],[2,3],[3,1]])|.
	\end{center}

	\begin{center}
		\includegraphics{figures/lof_poly.pdf}

		The poset \verb|LatticeOfFlats([0,1,2,2,1,3,3,3])|.
	\end{center}

	@exec@
	make_fig(LatticeOfFlats([[1,2],[2,3],[1,3]]),'lof_triangle',height=5,width=6)
	make_fig(LatticeOfFlats([0,1,2,2,1,3,3,3]),'lof_poly',height=6,width=4)
	'''
	def int_to_tuple(i): #converts an into to a tuple of the set bits (1-indexed)
		b = bin(i)[2:][::-1]
		return tuple(j+1 for j in range(len(b)) if b[j]=='1')
	##############
	#Make all flats
	##############
	flats=set()
	#####
	#data is a graph
	#flats are vertex partitions
	#####
	data = list(data)
	if hasattr(data[0], '__iter__'):
		#grab all vertices
		V = list(set(itertools.chain(*data)))
		#normalize data to be 0-indexed numbers
		data = [[V.index(e[0]),V.index(e[1])] for e in data]
		n=len(V)
		#here we iterate over all subsets of edges,
		#compute the corresponding partition and add it to flats
		for S in range(0,1<<len(data)):
			F = [1<<i for i in range(n)] #start with all vertices separate
			for i in [i for i in range(len(data)) if (1<<i)&S!=0]: #iterates over elements of S
				b1=F[data[i][0]]
				b2=F[data[i][1]]

				for j in range(n):
					if (1<<j)&b1!=0: F[j]|=b2 #if j is in the block b1 add the block b2 to the block containing j
					if (1<<j)&b2!=0: F[j]|=b1 #likewise with b1 and b2 exchanged

			flats.add(tuple(F))

		def elem_conv(e): #used to relabel after construction
			#remove duplicate blocks, sort make a tuple and for each block sort and make a tuple
			return tuple(
				sorted(
					list(set(
						tuple(
							sorted(list(
								V[i-1] for i in int_to_tuple(x)
							))
						)
						for x in e
					))
				)
				)

		def less(x,y): #reverse refinement
			return x!=y and all([x[i]&y[i]==x[i] for i in range(len(x))])
	######
	#data is a polymatroid
	######
	else:
		n = len(bin(len(data))) - 3
		for S in range(0,len(data)):
			is_flat = True
			for i in range(n):
				if S!=(S|(1<<i)) and data[S|(1<<i)] == data[S]:
					is_flat = False
					break
			if not is_flat: continue
			flats.add(S)

		def less(x,y): #containment
			return x!=y and x&y==x

		elem_conv = int_to_tuple
	##############
	#lattice is flats ordered under inclusion
	##############
	ret = Poset(elements=flats, less=less)
	ret.elements = [elem_conv(e) for e in ret.elements]
	return ret

def PartitionLattice(n=3):
	r'''
	@section@Built in posets@
	Returns the lattice of partitions of a $1,\dots,n$ ordered by refinement.

	\begin{center}
		\includegraphics{figures/Pi.pdf}

		The partition lattice $\Pi_4$.
	\end{center}

	@exec@
	make_fig(PartitionLattice(4),'Pi',width=12,height=8,nodescale=0.75)
	'''
	return LatticeOfFlats(itertools.combinations(range(1,n+1),2))

def NoncrossingPartitionLattice(n=3):
	r'''
	@section@Built in posets@
	Returns the lattice of noncrossing partitions of $1,\dots,n$ ordered by refinement.

	\begin{center}
		\includegraphics{figures/NC.pdf}

		The noncrossing partition lattice $\text{NC}_4$.
	\end{center}

	@exec@
	make_fig(NoncrossingPartitionLattice(4),'NC',height=8,width=12,nodescale=0.75)
	'''
	def noncrossing(p):
		for i in range(len(p)):
			pi = p[i]
			for j in range(i+1,len(p)):
				pj =p[j]
				if pj[0]<pi[0]:
					if any(x >= pi[0] and x<=pi[-1] for x in pj):
						return False
				elif pj[0]>pi[-1]:
					return True
				else:
					if any(x>pi[-1] for x in pj):
						return False
		return True
	Pi = PartitionLattice(n)
	return Pi.subposet([p for p in Pi if noncrossing(p)])

def UniformMatroid(n=3,r=3,q=1):
	r'''
	@section@Built in posets@
	Returns the lattice of flats of the uniform ($q$-)matroid of rank $r$ on $n$ elements.

	\begin{center}
		\includegraphics{figures/unif.pdf}

		The uniform matroid of rank $3$ on $4$ elements.
	\end{center}

	\begin{center}
		\includegraphics{figures/qunif.pdf}

		The uniform $2$-matroid of rank $3$ of dimension $4$.
	\end{center}

	@exec@
	make_fig(UniformMatroid(4,3,1),'unif',height=5,width=5)
	make_fig(UniformMatroid(4,3,2),'qunif',height=8,width=12,labels=False)
	'''
	if q==1:
		return Boolean(n).rankSelection(list(range(r))+[n])
	else:
		return Bnq(n,q).rankSelection(list(range(r))+[n])

def MinorPoset(L,genL=None, weak=False):
	r'''@section@Built in posets@
	Returns the minor poset given a lattice \verb|L| and a list of generators \verb|genL|.

	The join irreducibles are automatically added to \verb|genL|. If \verb|genL| is not provided the generating set will be only the
	join irreducibles.

	For more info on minor posets see: https://arxiv.org/abs/2205.01200
	'''
	if genL == None: genL = []
	genL = [L.elements.index(g) for g in genL]
	#make L the incMat so we can copy code, L_P is the poset
	L_set = L.elements
	L_P = L.copy()
	L = L_P.incMat
	L_P.elements = list(range(len(L_P.elements)))
	##################################
	#compute a table of all joins in L
	##################################
	joins = [[0 for i in range(0,len(L))]for j in range(0,len(L))]
	for i in range(0,len(L)):
		for j in range(i,len(L)):
			k = L_P.join(i,j,True)
			if k == None:
				raise Exception('input L to MinorPoset must be a lattice')
			joins[i][j]=k
			joins[j][i]=k
	L_Pmin = L_P.min()
	if len(L_Pmin)>1: raise Exception('input L to MinorPoset must be a lattice')
	zerohat = L_Pmin[0]
	#find irreducibles and make sure they're in genL
	for l in L_P.elements:
		if len(L_P.interval(zerohat, l).complSubposet([l]).max()) == 1:
			if l not in genL:
				genL.append(l)
	######################################################
	#compute all the minors
	#minors are encoded as a list whose first element
	#is the minimal element and the second element is the
	#list of generators
	######################################################
	minors = [[0,genL]]
	minors_M = [[0]]
	minors_ranks = [[] for i in range(0,len(genL)+1)]
	minors_ranks[len(genL)].append(0) #will add a zerohat later at index 0
	new = [[0,genL]]
	while len(new)>0:
		old = new
		new = []
		for l in old:
			r = minors.index(l)
			for i in range(0,len(l[1])):
				minor=[l[0],l[1][:i]+l[1][i+1:]] #delete i
				if minor in minors:
					s = minors.index(minor) #save index to adjust incidence matrix
				else:
					#add the minor and add a row and column to the incidence matrix
					s = len(minors_M)
					minors_ranks[len(minor[1])].append(s)
					minors.append(minor)
					for x in minors_M: x.append(0)
					minors_M.append([0 for x in range(-1,s)])
				minors_M[r][s] = -1
				minors_M[s][r] = 1
				if minor not in new: new.append(minor)

				#contract i
				temp = set([joins[l[1][i]][j] for j in l[1]])
				temp.remove(l[1][i])
				minor=[l[1][i],sorted(list(temp))]
				if weak: #extra deletions
					contr_gens = [L_P.join(g, l[0], indices=True) for g in genL]
					extra_del = [L_P.join(g, l[1][i], indices=True) for g in contr_gens if g not in l[1]]
					minor[1] = [g for g in minor[1] if g not in extra_del]
					if minor[0] in extra_del: continue
				if minor in minors:
					s = minors.index(minor)
				else:
					s = len(minors_M)
					minors_ranks[len(minor[1])].append(s)
					minors.append(minor)
					for x in minors_M: x.append(0)
					minors_M.append([0 for x in range(-1,s)])
				minors_M[r][s] = -1
				minors_M[s][r] = 1
				if minor not in new: new.append(minor)
	Poset.transClose(minors_M)

	##############
	#set up hasse diagram
	##############
	import types
	def minor_contains(KH,i):
		if i==L_set.index(KH[0]): return True
		if not L_P.lesseq(L_set.index(KH[0]),i,True): return False
		#join all generators of KH below i
		if len([h for h in KH[1] if L_P.lesseq(L_set.index(h),i)])==0: return False
		if len([h for h in KH[1] if L_P.lesseq(L_set.index(h),i)])==1: return i in [L_set.index(h) for h in KH[1]]
		k = list(itertools.accumulate([L_P.min(True)[0]]+list(h for h in KH[1] if L_P.lesseq(L_set.index(h),i,True)), lambda x,y: L_P.join(x,L_set.index(y),True)))[-1]
		return k == i

	def latt_nodeName(this, i):
		return 'latt_'+HasseDiagram.nodeName(this,i)

	def make_node_options(KH):
		def node_options(this, i):
			if minor_contains(KH,i): return 'color=black'
			return 'color=gray'
		return node_options

	def make_line_options(KH):
		def line_options(this, i, j):
			if minor_contains(KH,i) and minor_contains(KH,j): return 'color=black'
			return 'color=gray'
		return line_options

	class Genlatt(Poset):
		def __init__(this, G, *args, **kwargs):
			super().__init__(*args, **kwargs)
			this.G = G
			this.edges = {}
			#find all extra edges of diagram
			for l in this.complSubposet(this.max()):
				this.edges[l] = []
				for g in G:
					lg = this.join(l,g)
					if lg == l or len(this.interval(l,lg))==2: continue
					this.edges[l].append(lg)
			#overwrite L's covers function so hasse diagram does all edges
			def covers(this):
				return this.edges

	class MinorPosetHasseDiagram(HasseDiagram):

		def __init__(this, P, L, G, **kwargs):
			super().__init__(P, **kwargs)
			this.L = Genlatt(G,L)

		def latex(this, **kwargs):
			latt_args = {k[5:] : v for k,v in kwargs.items() if k[:5] == 'latt_'}
			latt_defaults = this.L.hasseDiagram.__dict__.copy()
			this.L.hasseDiagram.__dict__.update(latt_args)
			this.L.hasseDiagram.nodeName = latt_nodeName

			ret = super().latex(**kwargs)

			this.L.hasseDiagram.__dict__.update(latt_defaults)

			return ret

		def nodeLabel(this, i):
			if i in this.P.min(): return '$\\emptyset$'
			minorLatex = this.L.latex(node_options = make_node_options(this.P[i]), line_options = make_line_options(this.P[i]))
			minorLatex = ''.join(minorLatex.split('\n')[2:-1])
			return '\\begin{tikzpicture}\\begin{scope}\n'+minorLatex+'\n\\end{scope}\\end{tikzpicture}'


	P = Poset(minors_M, minors, minors_ranks)
	P.elements = [tuple([L_set[M[0]],tuple(L_set[g] for g in M[1])]) for M in P]
	print('P.elements',P.elements)
	print('L_set',L_set)
	P = P.adjoin_zerohat()
	P.hasseDiagram = MinorPosetHasseDiagram(P,L_P,genL)
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P
