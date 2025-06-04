'''@no_doc@no_children@'''
from .poset import Poset,Genlatt
from .hasseDiagram import *
from .utils import TriangularArray
import itertools

def Empty():
	r'''
	Returns an empty poset.

	@section@Built in posets@
	'''
	return Poset(elements = [], ranks = [], zeta = [])

def Bruhat(n,weak=False):
	r'''
	Returns the type $A_{n-1}$ Bruhat order (the symmetric group $S_n$)
	or the type $A_{n-1}$ left weak order.

	\begin{center}
		\begin{minipage}{0.4\textwidth}
			\begin{center}
			\includegraphics{figures/Bruhat_3.pdf}

			The poset \verb|Bruhat(3)|
			\end{center}
		\end{minipage}
		\begin{minipage}{0.4\textwidth}
			\begin{center}
			\includegraphics{figures/Weak_3.pdf}

			The poset \verb|Bruhat(3,True)|
			\end{center}
		\end{minipage}
	\end{center}

	@exec@
	make_fig(Bruhat(3), 'Bruhat_3',height=4, width=3)
	make_fig(Bruhat(3,True), 'Weak_3',height=4, width=3)
	@section@Built in posets@
	'''
	def pairing_to_perm(tau):
		arcs = [[int(x) for x in a] for a in tau]

		arcs.sort(key = lambda x: x[0])

		return tuple(a[1]-n for a in arcs)

	def nodeLabel(hasseDiagram, i):
		return ('' if n<=9 else ',').join([str(x) for x in hasseDiagram.P.elements[i]])
	def nodeName(hasseDiagram, i):
		return hasseDiagram.nodeLabel(hasseDiagram,i).replace(',','-')

	P = Uncrossing(n, E_only=True,weak=weak,zerohat=False).dual()
	P.hasseDiagram.nodeLabel = nodeLabel
	P.hasseDiagram.nodeName = nodeName
	P.hasseDiagram.offset = 1

	P.elements = [pairing_to_perm(e) for e in P.elements]
	P.name = "Type A_"+str(n)+" Bruhat order"

	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=not weak
	P.cache['isGorenstein()']=not weak
	P.cache['isLattice()'] = (n<3) or weak
	return P
def Root(n=3):
	r'''
	Returns the type $A_{n+1}$ root poset.

	\begin{center}
		\includegraphics{figures/root_3.pdf}

		The poset \verb|Root(3)|.
	\end{center}

	@exec@
	make_fig(Root(3).reorder(range(2,-1,-1),True),'root_3',height=3,width=3)
	@section@Built in posets@
	'''
	def covers(i,j):
		if i==1:
			if j==n:
				return []
			return [(1,j+1)]
		elif j==n:
			return [(i-1,n)]
		return [(i-1,j),(i,j+1)]
	def nodeName(this,i):
		return '-'.join(str(x) for x in this.P[i])
	return Poset(relations={(i, j) : covers(i,j) for i in range(1, n) for j in range(i+1, (n+1 if i>1 else n))},nodeName=nodeName)

def Butterfly(n):
	r'''
	Returns the rank $n+1$ bounded poset where ranks $1,\dots,n$ have two elements and all comparisons between ranks.
	\begin{center}
		\includegraphics{figures/Butterfly_3.pdf}

		The poset \verb|Butterfly(3)|.
	\end{center}
	@exec@
	make_fig(Butterfly(3),'Butterfly_3',height=5,width=2)
	@section@Built in posets@
	'''
	elements = [('a' if i%2==0 else 'b')+str(i//2) for i in range(2*n)]
	ranks = [[i,i+1] for i in range(0,2*n,2)]
	zeta = [([1] if i%2 else [1,0]) + [1]*(len(elements)-((i//2+1)*2)) for i in range(len(elements))]
	name = "Rank "+str(n+1)+" butterfly poset"

	P = Poset(zeta, elements, ranks, name = name,nodeName = lambda this,i: str(this.P[i])).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	P.cache['isLattice()'] = n<2
	return P

def Antichain(n):
	r'''
	Returns the poset on $1,\dots,n$ with no relations.

	\begin{center}
		\includegraphics{figures/antichain_3.pdf}

		The poset \verb|Antichain(3)|.
	\end{center}

	@exec@
	make_fig(Antichain(3),'antichain_3',height=3,width=3)
	@section@Built in posets@
	'''
	return Poset(elements=list(range(1,n+1)))

def Chain(n):
	r'''
	Returns the poset on $0,\dots,n$ ordered linearly (i.e. by usual ordering of integers).

	\begin{center}
		\includegraphics{figures/chain_3.pdf}

		The poset \verb|Chain(3)|.
	\end{center}

	@exec@
	make_fig(Chain(3),'chain_3',height=5,width=3)
	@section@Built in posets@
	'''
	elements = list(range(n+1))
	ranks = [[i] for i in elements]
	P = Poset(zeta = TriangularArray(1 for _ in TriangleRange(len(elements))), elements=elements, ranks=ranks, name = "Length "+str(n)+" chain", trans_close=False)
	#cach some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=False
	P.cache['isGorenstein()']=False
	P.cache['isLattice()'] = True
	return P

def Boolean(n):
	r'''
	Returns the poset of subsets of a set, ordered by inclusion.

	The parameter $n$ may be an integer, in which case the poset of
	subsets of $\{1,\dots,n\}$ is returned, or an iterable in which
	case the poset of subsets of $n$ is returned.

	\begin{center}
		\includegraphics{figures/Boolean_3.pdf}

		The poset \verb|Boolean(3)|.
	\end{center}

	@exec@
	make_fig(Boolean(3),'Boolean_3', height=6, width=4)
	@section@Built in posets@
	'''
	if hasattr(n,'__iter__'):
		X = n
		n = len(X)
	else:
		X = None
	P = Poset()
	P.elements = list(range(1<<n))
	P.zeta = TriangularArray([1 if i&j==i else 0 for i in range(1<<n) for j in range(i,1<<n)])
	P.ranks = [[] for _ in range(n+1)]
	for p in P.elements:
		P.ranks[len([c for c in bin(p) if c=='1'])].append(p) #p==P.elements.index(p)
	P.elements = [bin(e)[2:][::-1]+('0'*(n-len(bin(e)[2:]))) for e in P.elements]
	P.elements = [tuple([i+1 for i in range(len(e)) if e[i]=='1']) for e in P.elements]
	P.name = "Rank "+str(n)+" Boolean algebra"

	def nodeLabel(hasseDiagram, i):
		p = hasseDiagram.P[i]
		if len(p)==1: return '\\{'+str(p[0])+'\\}'
		return '\\{'+str(p)[1:-1]+'\\}'
#		S = hasseDiagram.P.elements[i]
#		s = str(S).replace(',','') if len(S) <= 1 else str(S)
#		return s.replace('(','\\{' if hasseDiagram.in_latex else '{').replace(')','\\}' if hasseDiagram.in_latex else '}').replace(',',', ')
	def nodeName(hasseDiagram, i):
		if i==0: return 'e'
		return ('' if n<10 else '-').join(str(x) for x in hasseDiagram.P[i])
	P.hasseDiagram.nodeLabel = nodeLabel
	P.hasseDiagram.nodeName = nodeName
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	P.cache['isLattice()'] = True

	if X!=None:
		X = list(X)
		for i in range(len(P)):
			P.elements[i] = tuple([X[j-1] for j in P[i]])
	return P

def Simplex(n):
	r'''
	Returns \verb|Boolean|\verb|(n+1)| the face lattice of the $n$-dimensional simplex.
	@section@Built in posets@
	'''
	return Boolean(n+1)

def Polygon(n):
	r'''
	Returns the face lattice of the $n$-gon.

	\begin{center}
		\includegraphics{figures/polygon_4.pdf}

		The poset \verb|Polygon(4)|.
	\end{center}

	@exec@
	make_fig(Polygon(4),'polygon_4',height=6,width=5)
	@section@Built in posets@
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
	def nodeName(hasseDiagram,i):
		if type(hasseDiagram.P[i])==tuple: return '-'.join(str(x) for x in hasseDiagram.P[i])
		return str(hasseDiagram.P[i])
	P = Poset(elements = elements, less = less, name = str(n)+"-gon face lattice", nodeLabel = nodeLabel, nodeName = nodeName, trans_close=False).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	P.cache['isLattice()']=True
	return P

def Cube(n):
	r'''
	Returns the face lattice of the $n$-dimensional cube.

	\begin{center}
		\includegraphics{figures/cube_2.pdf}

		The poset \verb|Cube(2)|.
	\end{center}

	@exec@
	make_fig(Cube(2),'cube_2',height=6,width=5)
	@section@Built in posets@
	'''
	def expand(E):
		return [e+'0' for e in E]+[e+'1' for e in E]+[e+'*' for e in E]

	def less(x,y):
		return x!=y and all([x[i]==y[i] or y[i]=='*' for i in range(len(x))])

	def sort_key(F): #revlex induced by 0<*<1
		return ''.join(['1' if f == '*' else '2' if f == '1' else '0' for f in F][::-1])

	elements = ['']
	for i in range(n): elements = expand(elements)
	elements.sort(key=sort_key)
	ranks = [[] for _ in range(n+1)]
	for p in elements:
		ranks[len([c for c in p if c=='*'])].append(elements.index(p))
	name = str(n)+"-cube face lattice"

	if n>=1:
		def nodeName(this,i):
			return this.P[i]
	else:
		def nodeName(this,i):
			return ['0','*','1','-'][i]
	P = Poset(elements = elements, ranks = ranks, less = less, name = name, nodeName = nodeName).adjoin_zerohat()

	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	P.cache['isLattice()']=True
	return P

def Torus(n=2, m=2):
	r'''
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
	@section@Built in posets@
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
		def nodeName(this,i):
			return this.P[i]
	else:
		symbols = [str(i) for i in range(m)]+['*'+str(i) for i in range(m)]
		elements = [tuple(x) for x in itertools.product(symbols, repeat=n)]

		def less(e,f):
			return e!=f and all([e[i] == f[i] or ('*' not in e[i] and f[i] in ('*'+e[i], '*'+str((int(e[i])-1)%m)) ) for i in range(len(e))])

		def rk(e):
			return len([x for x in e if '*' in x])

		def nodeName(this,i):
			return '-'.join(this.P[i])

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

	P = Poset(less=less, ranks=ranks, elements=elements, name = str(m)+" subdivided "+str(n)+"-torus",nodeName=nodeName).sort(sort_key).adjoin_zerohat().adjoin_onehat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']= n%2 == 1
	P.cache['isGorenstein()']= n == 1
	P.cache['isLattice()'] = n == 0
	return P

def GluedCube(orientations = None):
	r'''
	Returns the face poset of the cubical complex obtained from a $2\times\dots\times2$ grid of cubes of dimension \verb|len(orientations)| via a series of gluings as indicated by the parameter \verb|orientations|.

	If \verb|orientations| is \verb|[1,...,1]| a torus is constructed and if \verb|orientations| is \verb|[-1,...,-1]| the
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
	@section@Built in posets@
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
	Returns the face poset of a cubical complex homeomorphic to the Klein Bottle.

	Pseudonym for \verb|GluedCube([-1,1])|.
	@section@Built in posets@
	'''
	P = GluedCube([-1,1])
	P.name = "Klein Bottle"
	return P

def ProjectiveSpace(n=2):
	r'''
	Returns the face poset of a Cubical complex homeomorphic to projective space of dimension $n$.

	Pseudonym for \verb|GluedCube([-1,...,-1])|.

	\begin{center}
		\includegraphics{figures/projectiveSpace.pdf}

		The poset \verb|ProjectiveSpace(2)|.
	\end{center}

	@exec@
	make_fig(ProjectiveSpace(), 'projectiveSpace', height=8, width=15)
	@section@Built in posets@
	'''
	P = GluedCube([-1]*n)
	P.name = str(n)+"-dimensional projective space"
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']= n%2 == 1
	P.cache['isGorenstein()']= n%2 == 1
	P.cache['isLattice()'] = n==0
	return P

	return P

def Grid(n=2,d=None):
	r'''
	Returns the face poset of the cubical complex consisting of a $\verb|d[0]|\times\dots\times\verb|d[-1]|$ grid of $n$-cubes.

	\begin{center}
		\includegraphics{figures/grid.pdf}

		The poset \verb|Grid(2,[1,1])|.
	\end{center}

	@exec@
	make_fig(Grid(2,[1,2]),'grid',height=8,width=15)
	@section@Built in posets@
	'''
	if d == None: d = [2]*n
	cube = Cube(n).complSubposet([0])
	Gamma = Empty()
	identifications = {}
	#build the complex as a union and the list of gluings
	for i in itertools.product(*[range(j) for j in d]):
		elements = [(F, i) for F in cube]
		Gamma = Gamma.union(Poset(elements = elements, zeta = cube.zeta, ranks = cube.ranks))

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
	P.cache['isLattice()'] = all([x==1 for x in d])
	return P

def Uncrossing(t, upper=False, weak=False, E_only=False, zerohat=True):
	r'''
	Returns either a lower interval $[\widehat{0},t]$ or the upper interval $[t,\widehat{1}]$ in the uncrossing poset.

	The parameter \verb|t| should be either a pairing encoded as a list \verb|[s_1,t_1,...,s_n,t_n]| where
	\verb|s_i| is paired to \verb|t_i| or an integer greater than 1. If t is an integer the entire uncrossing
	poset of rank $\binom{t}{2}+1$ is returned.

	Covers in the uncrossing poset are of the form $\sigma<\tau$
	where $\sigma$ is obtained from $\tau$ by swapping points
	$i$ and $j$ to remove a crossing.
	If \verb|weak| is \verb|True| then the weak subposet
	is returned that has cover relations
	$\sigma<\tau$ when $\sigma$ is
	obtained from $\tau$ by removing a single crossing
	via swapping two adjacent points. If \verb|E_only| is
	\verb|True| only swaps $(i,j)$ such that the pairing
	$\tau$ satisfies $\tau(i)<i$ and $\tau(j)<j$ are used.
	These two flags are provided
	because this function acts as a backend to \verb|Bruhat|.
	Calling \verb|Uncrossing(n,E_only=True)| constructs
	the Bruhat order on $\mathfrak{S_n}$ and adding
	\verb|weak=True| constructs the weak order on $\mathfrak{S}_n$.

	If \verb|zerohat| is \verb|False| then no minimum is adjoined.

	Raises a \verb|ValueError| when \verb|t| is an integer less than 2.

	For more info on the uncrossing poset see \cite{lam-15}.

	\begin{center}
		\includegraphics{figures/uc.pdf}

		The poset \verb|Uncrossing(3)==Uncrossing([1,4,2,5,3,6])|.
	\end{center}

	@exec@
	make_fig(Uncrossing(3), 'uc', height=15, width=15, nodescale=0.75)
	@section@Built in posets@
	'''
	#Throughout pairings are encoded as lists of numbers, each number encodes
	#a pair as two bits set. For example the pairing {{1,3},{2,4}} is encoded
	#as [2**0+2**2,2**1+2**3]=[5,10]
	if type(t) == int:
		if t<2: raise ValueError(f't must be at least 2, got t={t}')
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
			if x&1: ret.append(i)
			x >>= 1
			i += 1
		return tuple(ret)

	def pairingFormat(x):
		return tuple(sorted(tuple(setFormat(y) for y in x)))
#		return "".join(sorted([setFormat(y) for y in x]))


	#swaps i and j in the pairing p
	def swap(p,i,j):
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
	#returns a tuple (P,ranks,M) which is the list of elements, the rank list and the zeta matrix
	def lowerOrderIdeal(t):
		epsilon = 1 if upper else -1
		if not upper and c(t)==0: return [t],[[1],[0]],[1,1,1]#[[0,-1],[1,0]]

		P=[t]
		ranks = [[0]] #this is built up backwards for convenience and reversed before returning
		M=[[1]]
		relations = {0:[]}

		num = 1 #index in to P of next element to add
		level = [t] #list of current rank to expand in next step
		leveli = [0] #indices in to P of the elements of level
		newLevel = [] #we build level for the next step during the current step here
		newLeveli = [] #indices in to P for the next step
		newRank = [] #the new rank indices to add
		def I_set(tau):
			if not E_only: return range(0,(len(t)<<1)-1)
			def iterator():
				for i in range(0,(len(t)<<1)-1):
					pad = (1<<i)-1
					if any(
						(arc&(1<<i)!=0)
						and
						arc&pad==0
						for arc in tau
						): continue
					yield i
			return iterator()
		def J_set(tau,i):
			if not E_only: return range(i+1,(len(t)<<1))
			possible=[i+1] if weak else range(0,(len(t)<<1))
			def iterator():
				for j in possible:
					pad = (1<<j)-1
					if any(
						(arc&(1<<j)!=0)
						and
						arc&pad==0
						for arc in tau
						): continue
					yield j
			return iterator()
		while len(level) > 0:
#			for i in range(0,(len(t)<<1)-1): #iterate over all pairs we can uncross
#				for j in range(i+1,len(t)<<1):
			for k in range(0,len(level)): #do the uncross
				for i in I_set(level[k]):
					for j in J_set(level[k],i):
						temp = swap(level[k],i,j)
						c_temp = c(temp)
						if c_temp != c(level[k])+epsilon: continue
						if temp in P:
							relations[P.index(temp)].append(leveli[k])
							M[P.index(temp)][leveli[k]]=1
							M[leveli[k]][P.index(temp)]=-1
							continue
						P.append(temp)
						newRank.append(num)
						if c_temp > 0: #if not minimal continue uncrossing
							newLevel.append(temp)
							newLeveli.append(num)
						num+= 1

						for x in M: x.append(0)
						M.append([0 for x in range(0,len(M[0]))])
						relations[len(P)-1]=[]
						assert(len(M)==len(P))
						relations[len(P)-1].append(leveli[k])
						M[-1][leveli[k]]=1
						M[leveli[k]][-1]=-1

			level = newLevel
			newLevel = []
			leveli = newLeveli
			newLeveli = []
			ranks.append(newRank)
			newRank = []

		ranks = ranks[::-1]
		#breakpoint()
#		M = TriangularArray([m[i:] for i,m in enumerate(M)],flat=False)
#		Poset.transClose(M)
		#TODO construct from relations instead of doing things manually
		#TODO get rid of M entirely, and rename P elements
		#this is all just leftover from the before time
		M,new_order = Poset.zeta_from_relations(relations,P)
		P=[P[i] for i in new_order]
		ranks = [[new_order.index(i) for i in rk] for rk in ranks]
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
			else: this.offset = 0.5
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
			if i == 0: return 'z'
			i = i-1 #zerohat gets added first so shift back
			p=this.pairings[i]
			n=len(p)
			return '/'.join(['_'.join(str(j+1) for j in range(0,n<<1) if (1<<j)&p[k]!=0) for k in range(0,n)])
#			return ''.join([str(j+1) for k in range(0,n) for j in range(0,n<<1) if (1<<j)&p[k]!=0])

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

			tau = this.P[i]

			for j in range(n):
				s = tau[j][0]
				t = tau[j][1]
				this.canvas.create_line(*pt(s),*pt(t))
			return



	preamble = "\\def\\r{1}\n\\def\\n{"+str(n<<1)+"}\n\\newcommand{\\medial}{\n\\draw circle (\\r);\n\\foreach\\i in{1,...,\\n}\n\t{\n\t\\pgfmathsetmacro{\\j}{-90-360/\\n*(\\i-1)}\n\t\\fill (\\j:-\\r) circle (2pt) node [anchor=\\j] {$\\i$};\n\t\\coordinate (\\i) at (\\j:-\\r);\n\t}\n}"
	P = Poset(M, P, ranks, name = name, hasse_class = UncrossingHasseDiagram, preamble = preamble)
	if not upper:
		P = P.adjoin_zerohat() if zerohat else P
		P.hasseDiagram = UncrossingHasseDiagram(P, preamble=preamble)
#	lex on (source pts, sink pts) sorts for the 312 decomposition but is a little uglier
#	P = P.sort(key=lambda x:tuple() if x==0 else (tuple(y[0] for y in eval('('+x.replace(')','),')+')')),tuple(y[1] for y in eval('('+x.replace(')','),')+')'))))
	P = P.sort(key=lambda x:tuple() if x==0 else x)
	pairings.sort(key=lambda p:P.elements.index(pairingFormat(p)))
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

def Bnq(n=2, q=2):
	r'''
	Returns the poset of subspaces of the vector space $\F_q^n$ where $\F_q$ is the field with q elements.

	Currently only implemented for \verb|q| a prime. Raises an instance of \verb|NotImplementedError| if \verb|q| is not prime.

	\begin{center}
		\includegraphics{figures/Bnq.pdf}

		The poset \verb|Bnq(3,2)|.
	\end{center}

	@exec@
	make_fig(Bnq(3,2),'Bnq',height=10,width=16)
	@section@Built in posets@
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


	def basis(S):
		basis = []
		span = 1 #zero space
		span_filter=spaces
		while span!=S:
			#find first vector in S but not span
			T = S^span
			v = bin(T)[2:-1][::-1].index('1')+1
			basis.append(v)
			#intersect all spaces containing span and v
			span = (1<<qn)-1 #whole space
			span_filter = [W for W in span_filter if W&(1<<v)!=0]
			for W in span_filter:
				span &= W
		return tuple(basis)

	def list_to_mat(B):
#		B = basis(S)
		return tuple(tuple(vec(b)) for b in B)

	def nodeLabel(hd, i):
		if i==0: return '$\\left(\\begin{matrix}'+' & '.join('0'*n)+'\\end{matrix}\\right)$'
		return '$\\left(\\begin{matrix}' + r'\\'.join(' & '.join(str(x) for x in row) for row in hd.P[i])+'\\end{matrix}\\right)$'

	if q<10:
		def nodeName(hd, i):
			return '0' if i==0 else '/'.join(''.join(str(x) for x in y) for y in hd.P[i])
	else:
		def nodeName(hd, i):
			return '0' if i==0 else '/'.join('-'.join(str(x) for x in y) for y in hd.P[i])
	P = Poset(elements = spaces, less = lambda i,j: i!=j and i&j == i, nodeLabel=nodeLabel,preamble='\\usepackage{amsmath}',nodeName=nodeName)
	#sort ranks: revlex on bases induced by ordering vectors by interpreting them as base q representations of numbers
	P.elements = [basis(S)[::-1] for S in P]
	P = P.sort()
	P.elements = [list_to_mat(B)[::-1] for B in P]
	P.cache['isRanked()'] = True
	P.cache['isEulerian()'] = False
	P.cache['isGorenstein()'] = False
	P.cache['isLattice()'] = True
	return P

def DistributiveLattice(P, indices=False):
	r'''
	Returns the lattice of ideals of a given poset.

	\begin{center}
		\includegraphics{figures/DL.pdf}

		The poset \verb|DistributiveLattice(Root(3))|.
	\end{center}

	When generating a Hasse diagram with \verb|latex()| use
	the prefix \verb|irr_| to control options for the node diagrams.

	@exec@
	make_fig(DistributiveLattice(Root(3)),'DL',height=10,width=6,irr_height=0.75,irr_width=1,irr_scale='1',irr_labels=False)
	@section@Built in posets@
	'''
	#make principal ideals
	Z = P.zeta
	irr=[]
	for i in range(0,len(P)):
		x=1<<i
		coli=list(Z.col(i))
		for j in range(len(coli)):
			if coli[j]!=0: x|=1<<j
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
	ranks=[[] for i in range(0,len(P)+1)]
	if indices:
		elements = ideals
		def less(I, J):
			return I!=J and I&J==I
	else:
		elements = []
		for I in ideals:
			ranks[len([c for c in bin(I) if c=='1'])].append(len(elements))
			elements.append(tuple(P[i] for i in range(len(P)) if (1<<i)&I!=0))
		def less(I,J):
			return I!=J and all(i in J for i in I)
	def nodeName(this,i):
		return '/'+'/'.join(str(j) for j in range(len(this.Q)) if (1<<j)&ideals[i]!=0)+'/'
	JP = Poset(
		elements = elements,
		less = less,
		hasse_class = SubposetsHasseDiagram,
		nodeName=nodeName,
		prefix='irr',
		Q=P,
		irr_scale='0.1'
		)
	JP.cache['isRanked()'] = True
	JP.cache['isEulerian()'] = len(JP) == 1<<len(P)
	JP.cache['isGorenstein()'] = JP.cache['isEulerian()']
	JP.cache['isLattice()'] = True
	return JP
def Intervals(P):
	r'''
	Returns the lattice of intervals of a given poset (including the empty interval).

	\begin{center}
		\includegraphics{figures/interval.pdf}

		The poset \verb|Intervals(Boolean(2))|.
	\end{center}

	When generating a Hasse diagram with \verb|latex()| use
	the prefix \verb|int_| to control options for the node diagrams.

	@exec@
	make_fig(Intervals(Boolean(2)),'interval',height=10,width=6,int_height=1,int_width=1,int_scale='1',int_labels=False)
	@section@Built in posets@
	'''
	Z=P.zeta
	elements = [tuple()]+[(P[i],P[j]) for i in range(len(P)) for j in range(i,len(P)) if Z[i,j]!=0]
	ranks = [[0]]+[[] for _ in range(len(P.ranks))]
	for i in range(len(elements))[1:]:
		ranks[1+P.rank(elements[i][1])-P.rank(elements[i][0])].append(i)
	def less(I,J):
		return len(I)==0 or P.lesseq(J[0],I[0]) and P.lesseq(I[1],J[1])

	def is_in(i,I):
		return len(I)>0 and P.lesseq(I[0],i) and P.lesseq(i,I[1])

	ret = Poset(
		elements=elements,
		ranks=ranks,
		less=less,
		hasse_class=SubposetsHasseDiagram,
		Q=P,
		is_in=is_in,
		prefix='int',
		int_scale='0.1'
		)
	ret.cache['isLattice()'] = True
	return ret
#def SignedBirkhoff(P):
#	D = DistributiveLattice(P, indices=True)
#	def maximal(I):
#		elems = [i for i in range(1<<len(P)) if (1<<i)&I!=0]
#		return [i for i in elems if all(P.incMat[i][j]!=1 for j in elems]
#	achains = [maximal(I) for I in D]
#


def LatticeOfFlats(data,as_genlatt=False):
	r'''
	Returns the lattice of flats given either a list of edges of a graph or the rank function of a (poly)matroid.

	When the input represents a graph it should be in the format \verb|[[i_1,j_1],...,[i_n,j_n]]|
	where the pair \verb|[i_k,j_k]| represents an edge between \verb|i_k| and \verb|j_k| in the graph.

	When the input represents a (poly)matroid the input should be a list of the ranks of
	sets ordered reverse lexicographically (i.e. binary order). For example, if f is the
	rank function of a (poly)matroid with ground set size 3 the input should be
		\[
		\verb|[f({}),f({1}),f({2}),f({1,2}),f({3}),f({1,3}),f({2,3}),f({1,2,3})]|.
		\]

	When \verb|as_genlatt| is \verb|True| the return value is an instance
	of \verb|Genlatt| with generating set the closures of singletons.

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
	@section@Built in posets@
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

		def nodeLabel(hd,i):
			return '/'.join(''.join(str(x) for x in y) for y in hd.P[i])
		nodeName = nodeLabel
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

		def nodeLabel(hd,i):
			return '\\{'+str(hd.P[i])[1:-1]+'\\}'
		def nodeName(hd,i):
			return nodeLabel(hd,i).replace('\\}','/').replace('\\}','/').replace(',','-')

		elem_conv = int_to_tuple
	##############
	#lattice is flats ordered under inclusion
	##############
	ret = Poset(elements=flats, less=less, nodeLabel=nodeLabel)
	ret.elements = [elem_conv(e) for e in ret.elements]
	ret.cache['isLattice()'] = True
	return ret

def PartitionLattice(n=3):
	r'''
	Returns the lattice of partitions of a $1,\dots,n$ ordered by refinement.

	\begin{center}
		\includegraphics{figures/Pi.pdf}

		The partition lattice $\Pi_4$.
	\end{center}

	@exec@
	make_fig(PartitionLattice(4),'Pi',width=12,height=8,nodescale=0.75)
	@section@Built in posets@
	'''
	P = LatticeOfFlats(itertools.combinations(range(1,n+1),2))
	P.cache['isRanked()'] = True
	P.cache['isEulerian()'] = n<2
	return P

def NoncrossingPartitionLattice(n=3):
	r'''
	Returns the lattice of noncrossing partitions of $1,\dots,n$ ordered by refinement.

	\begin{center}
		\includegraphics{figures/NC.pdf}

		The noncrossing partition lattice $\text{NC}_4$.
	\end{center}

	@exec@
	make_fig(NoncrossingPartitionLattice(4),'NC',height=12,width=12,nodescale=0.75)
	@section@Built in posets@
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

	def nodeName(this,i):
		return '/'.join(''.join(str(b) for b in B) for B in this.P[i])

	def nodeLabel(this,i):
		if this.in_tkinter:
			return str(this.P[i])
		ret=["\\begin{tikzpicture}\n\\begin{scope}\n\t\\medial\n"]
		for block in this.P[i]:
			if len(block)==1: continue
			ret.append('%'+str(this.P[i]))
			ret.append('%'+str(i)+'\n')
			ret.append('\n%'+str(len(this.P[i]))+'\n')
			ret.append('\t\\filldraw'+'--'.join('('+str(j)+')' for j in block)+';')
		return ''.join(ret+["\\end{scope}\\end{tikzpicture}"])

	def nodeDraw(this, i):
		size = 10*int(this.nodescale)
		ptsize = this.ptsize if type(this.ptsize)==int else int(this.ptsize[:-2])
		x = float(this.loc_x(this, i))*float(this.scale)+float(this.scale)*float(this.width)/2+float(this.padding)
		y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this, i))*float(this.scale)+float(this.padding))
		this.canvas.create_oval(x-size, y-size, x+size, y+size)

		def pt(i):
			px = x + int(this.nodescale)*math.cos(i*2*math.pi/(2*n)+math.pi/2)*10
			py = y + int(this.nodescale)*math.sin(i*2*math.pi/(2*n)+math.pi/2)*10
			return px, py
		for j in range(2*n):
			px,py = pt(j)
			this.canvas.create_oval(px-ptsize, py-ptsize, px+ptsize, py+ptsize, fill='black')

		for block in this.P[i]:
			this.canvas.create_polygon(list(itertools.chain(*[pt(j) for j in block])), outline='black',fill='gray',width=1)
		return
	Pi = PartitionLattice(n)
	P = Pi.subposet([p for p in Pi if noncrossing(p)])
	P.hasseDiagram.nodeDraw = nodeDraw
	P.hasseDiagram.nodeLabel = nodeLabel
	P.hasseDiagram.nodeName = nodeName
	P.hasseDiagram.preamble = "\\def\\r{1}\n\\def\\n{"+str(n)+"}\n\\newcommand{\\medial}{\n\\draw circle (\\r);\n\\foreach\\i in{1,...,\\n}\n\t{\n\t\\pgfmathsetmacro{\\j}{-90-360/\\n*(\\i-1)}\n\t\\fill (\\j:-\\r) circle (2pt) node [anchor=\\j] {$\\i$};\n\t\\coordinate (\\i) at (\\j:-\\r);\n\t}\n}"
	P.cache['isLattice()'] = True
	P.cache['isRanked()'] = True
	P.cache['isEulerian()'] = n==1

	return P

def UniformMatroid(n=3,r=3,q=1):
	r'''
	Returns the lattice of flats of the uniform ($q$-)matroid of rank $r$ on $n$ elements.

	Currently only implemented for \verb|q=1| or a prime. Raises an instance of \verb|NotImplementedError| if \verb|q| is neither 1 nor prime.

	\begin{center}
		\includegraphics{figures/unif.pdf}

		The poset \verb|UniformMatroid(4,3)|.
	\end{center}

	\begin{center}
		\includegraphics{figures/qunif.pdf}

		The poset \verb|UniformMatroid(4,3,2)|.
	\end{center}

	@exec@
	make_fig(UniformMatroid(4,3,1),'unif',height=5,width=5)
	make_fig(UniformMatroid(4,3,2),'qunif',height=8,width=12,labels=False,ptsize='1.25pt')
	@section@Built in posets@
	'''
	if q==1:
		P = Boolean(n).rankSelection(list(range(0,r))+[n])
	else:
		P = Bnq(n,q).rankSelection(list(range(0,r))+[n])
	P.cache['isEulerian()'] = q==1 and r==n
	P.cache['isGorentein()'] = (q==1 and r==n) or r<=1
	P.cache['isLattice()'] = True
	return P

def MinorPoset(L,genL=None, weak=False):
	r'''
	Returns the minor poset given a lattice \verb|L| and a list of generators \verb|genL|, or a list of edges specifying a graph.

	The join irreducibles are automatically added to \verb|genL|. If \verb|genL| is not provided the generating set will be only the
	join irreducibles.

	If \verb|L| is an instance of the \verb|Poset| class then it
	is assumed to be a lattice, an instance of \verb|Genlatt| is
	created from \verb|L| and \verb|genL| and the minor poset of
	the encoded {\genlatt} is returned. In this case the returned
	poset when plotted with \verb|Poset.latex| has elements
	represented as {\genlatts}.

	If \verb|L| is not an instance of the \verb|Poset| class
	it should be an iterable of length 2 iterables that
	specify edges of a graph. For example, \verb|L=[[1,2],[2,3],[3,1]]|
	specifies the 3-cycle graph. The minor poset of the graph
	is returned. In this case when plotting the returned poset
	with \verb|Poset.latex| the elements are represented as graphs.
	Furthermore, there are a few additional options you can use
	to control the presentation of the graphs in the Hasse
	diagram:
	\begin{itemize}
		\item[]{\verb|G_scale| -- Scale of the graph, default is 1.}
		\item[]{\verb|G_pt_size| -- size in points to use for the
		vertices, default is 2.}
		\item[]{\verb|G_node_options| -- Options to place on nodes in the graph, default is \verb|''|.}
		\item[]{\verb|G_node_sep| -- String used to separate names of vertices in the vertex names for minors, default is \verb|'/'|.}
		\item[]{\verb|G_label_dist| -- Distance of vertex to
			its label, default is \verb|1/4|.}
		\item[]{\verb|G_label_scale| -- Scale factor for the
			vertex labels, default is 1.}
	\end{itemize}

	If \verb|weak| is \verb|True| then the weak minor poset is
	returned. Briefly, this poset does not have relations
	$(K,H)\le(M,I)$ when some generator $g$ was deleted to form
	$(M,I)$ and $g\le\zerohat_K$.

	For more info on minor posets see \cite{gustafson-23}.


	\begin{center}
		\includegraphics[width=0.4\textwidth]{figures/M_triangle.pdf}

		The poset \verb|MinorPoset([[1,2],[2,3],[3,1]])|.
	\end{center}

	\begin{center}

		\includegraphics[width=0.4\textwidth]{figures/M_lof_triangle.pdf}
		\hspace{0.5in}
		\includegraphics[width=0.4\textwidth]{figures/M_lof_triangle_weak.pdf}

		On the left the poset \verb|MinorPoset(LatticeOfFlats([[1,2],[2,3],[3,1]]))| and on the right
		the poset \verb|MinorPoset(LatticeOfFlats([[1,2],[2,3],[3,1]]),weak=True)|.
	\end{center}

	\begin{center}
		\includegraphics[width=0.4\textwidth]{figures/M_lof_poly.pdf}
		\hspace{0.5in}
		\includegraphics[width=0.4\textwidth]{figures/M_lof_poly_weak.pdf}

		On the left the poset \verb|MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3]))| and on the right the poset \verb|MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3]), weak=True)|.
	\end{center}

	\begin{center}
		\includegraphics[width=0.4\textwidth]{figures/M_B_2.pdf}
		\hspace{0.5in}
		\includegraphics[width=0.4\textwidth]{figures/M_B_2_weak.pdf}

		On the right the poset \verb|MinorPoset(LatticeOfFlats(Boolean(2),Boolean(2)[1:4]))| and on the left
		the poset \verb|MinorPoset(LatticeOfFlats(Boolean(2),Boolean(2)[1:4]),weak=True)|.
	\end{center}

	@exec@
	make_fig(MinorPoset([[1,2],[2,3],[1,3]]),'M_triangle',height=10,width=10,G_line_options='line width=1pt', G_scale=3/8, G_label_scale=7/8, G_label_dist=.25,G_pt_size=7/4)
	make_fig(MinorPoset(LatticeOfFlats([[1,2],[2,3],[1,3]])),'M_lof_triangle',height=10,width=8,L_height=0.75,L_width=1,L_labels=False)
	make_fig(MinorPoset(LatticeOfFlats([[1,2],[2,3],[1,3]]),weak=True),'M_lof_triangle_weak',height=10,width=8,L_height=0.75,L_width=1,L_labels=False)
	make_fig(MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3])),'M_lof_poly',height=10,width=12,L_height=1,L_width=1,L_labels=False)
	make_fig(MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3]),weak=True),'M_lof_poly_weak',height=10,width=12,L_height=1,L_width=1,L_labels=False)
	make_fig(MinorPoset(Boolean(2),Boolean(2)[1:4],weak=True), 'M_B_2_weak',height=10,width=8,L_height=1,L_width=1,L_labels=False)
	make_fig(MinorPoset(Boolean(2),Boolean(2)[1:4],weak=False), 'M_B_2',height=10,width=8,L_height=1,L_width=1,L_labels=False)
	@section@Built in posets@
	'''
	if isinstance(L,Poset): return Genlatt(L, G=genL).minorPoset(weak)

	#L is presumed to be a list of edges of a graph
	LG = Genlatt(LatticeOfFlats(L))
	M = LG.minorPoset()

	class GraphMinorPosetDiagram(HasseDiagram):

		def __init__(this,P,**kwargs):
			super().__init__(P,**kwargs)
			for arg,default in [('G_scale',1),('G_pt_size',2),('G_line_options',''),('G_node_options',''),('G_node_sep','/'),('G_label_dist',1/4),('G_label_scale','1')]:
				setattr(this,arg,(kwargs[arg] if arg in kwargs else default))
				this.defaults[arg]=getattr(this,arg)
			this.validate()
		def validate(this):
			super().validate()
			if hasattr(this,'G_line_options') and type(this.G_line_options)==str:
				G_line_options=this.G_line_options
				this.G_line_options=lambda hd,i,j:G_line_options
				this.G_line_options.__name__="'"+G_line_options+"'"
			if hasattr(this,'G_node_options') and type(this.G_node_options)==str:
				G_node_options=this.G_node_options
				this.G_node_options=lambda hd,i:G_node_options
				this.G_node_options.__name__="'"+G_node_options+"'"

		def nodeLabel(this,i):
			if i==0: return '$\\widehat{0}$'
			ret=['\\begin{tikzpicture}\n\\begin{scope}']

			KH = this.P[i]
			degrees = lambda i: 90-i*360/len(KH.min()[0])
			#make coordinates
			for index,block in enumerate(KH.min()[0]):
				ret.append('\\coordinate(G_{})at({}:{});'.format(block[0], degrees(index), this.G_scale))
			#draw edges
			for g in KH.G:
				verts = [block for block in KH.min()[0] if block not in g]
				assert(len(verts)==2)
				ret.append('\\draw[{}](G_{})--(G_{});'.format(this.G_line_options(this,verts[0][0],verts[1][0]),verts[0][0],verts[1][0]))
			#draw points
			for block in KH.min()[0]:
				ret.append('\\fill[{}](G_{})circle({}pt);'.format(this.G_node_options(this,block[0]),block[0],this.G_pt_size*this.G_scale))

			#make labels
			for index,block in enumerate(KH.min()[0]):
				ret.append('\\node[{}][shift={{+({}:{})}}]at(G_{}){{\\scalebox{{{}}}{{{}}}}};'.format(this.G_node_options(this,i),degrees(index), this.G_label_dist, block[0], this.G_label_scale,this.G_node_sep.join(str(x) for x in block)))

			ret.append('\\end{scope}\n\\end{tikzpicture}')
			return '\n'.join(ret)

	M.hasseDiagram = GraphMinorPosetDiagram(M,G_node_sep='')
	if not weak:
		M.cache['isRanked()'] = True
		M.cache['isGorenstein()'] = True
		M.cache['isEulerian()'] = True
	if weak:
		M.cache['isLattice()'] = True
		M.cache['isEulerian()'] = len(L) == 1<<len(L.ranks[1])
		M.cache['isGorenstein()'] = len(L) == 1<<len(L.ranks[1])
	return M
