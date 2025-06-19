'''@no_doc@no_children@'''
import itertools
import copy
import decorator
import collections
import math
import random
from .polynomial import *
from .hasseDiagram import *
from .utils import *
import numpy as np
###########################################
#Cube_1 and Chain_1 needed for pyr and prism
#can't import examples.py because of circular
#dependencies, just reimplement
###########################################
def Chain_1():
	'''@no_doc@'''
	return Poset(relations={0:[1]})

def Cube_1():
	'''@no_doc@'''
	return Poset(relations={0:['0','1'],'0':['*'],'1':['*']})
###########################################
#Poset Class
###########################################
class Poset:
	r'''
	A class representing a finite partially ordered set (possibly quasigraded).

	Posets are encoded by a list \verb|elements|, a zeta
	function \verb|zeta| describing the relations and a
	list \verb|ranks| that specifies the length of each element.
	Instances of \verb|Poset| also
	have an attribute \verb|hasseDiagram| which is an instance
	of the \verb|HasseDiagram| class used for plotting the poset.

	To construct a poset you must pass one of the zeta matrix
	matrix \verb|zeta|, a function \verb|less| or a list/dictionary
	\verb|relations| to describe
	the relations. Additionally you may wish to specify the
	elements as a list called \verb|elements| or by using
	the \verb|relations| argument. The full list of constructor
	arguments are listed below.

	Usually, the values of the zeta function are all 0 or 1,
	with 1 indicating a relation. The given zeta function may
	take other values, with 0 always indicating no relation
	and any other value indicating a relation with the
	specified weight. Additionally, if you provide a value
	for \verb|ranks| you can specify any non-negative integer for
	the ranks of elements.

	\begin{itemize}
		\item[]{
		\verb|zeta| -- A triangular array indexed by $i,j$ such
		that $i<j$ whose entries are 0 if $i$ and $j$ are
		incomparable and otherwise an arbitrary weight. This
		argument may be an instance of \verb|TriangularArray|,
		a nested iterable or a flat iterable; in the latter case
		\verb|flat_zeta| should be \verb|True|. Elements are
		read row-wise when \verb|zeta| is an iterable.

		Note, unless \verb|trans_close| is \verb|False|
		it is only necessary to specify cover relations in the
		zeta matrix as the transitive closure will be computed.
		}
		\item[]{
		\verb|elements| -- A list specifying the elements of the poset.

			The default value is \verb|[0,...,len(zeta)-1]|
			Note, the list will be reordered into a linear extension as this is necessary to store the zeta function as a triangular array.
		}
		\item[]{
		\verb|ranks| -- A list of lists. The $i$th list is a list of indices of element of
			length $i$. This argument is inessential, if not provided it will be computed by the constructor.
			If constructing a large poset with an easily computed
			rank function you may wish to compute and pass the
			rank function to the constructor. This
			option is also useful if you are constructing
			a quasigraded poset, the provided rank list
			does not necessarily have to match up with
			the usual length function.
		}
		\item[]{
		\verb|relations| -- Either a list of pairs $(x,y)$ such that $x<y$ or a dictionary
			whose values are lists of elements greater than the associated key.
			This is used to construct the zeta function if it is not provided.
			Note, it is only necessary to specify cover relations.
		}
		\item[]{
		\verb|less| -- A function that given two elements $p,q$ returns \verb|True| when
			$p < q$. This is used to construct the zeta function if neither \verb|zeta|
			nor \verb|relations| are provided. It is only necessary to specify cover relations.
		}
		\item[]{
		\verb|indices| -- A boolean indicating indices instead of elements are used in
			\verb|relations| and \verb|less|. The default is \verb|False|.
		}
		\item[]{
		\verb|name| -- An optional identifier. If not provided no name attribute is set.
		}
		\item[]{
		\verb|hasse_class| -- An instance of \verb|hasse_class| is constructed with arguments being
			this poset plus all keyword arguments passed to \verb|Poset|, i.e.:
				\begin{verbatim}
				this.hasseDiagram = hasse_class(this, **kwargs)
				\end{verbatim}
			If you subclass \verb|HasseDiagram| to change default drawing behavior pass
			your subclass when constructing a poset.

			The default value is \verb|HasseDiagram|.
		}
		\item[]{
		\verb|trans_close| -- If \verb|True| the transitive closure of \verb|zeta| is
			computed, this should be \verb|False| only if the provided matrix satisfies
			\verb|zeta[i,j]==0| for $i$ and $j$ incomparable and \verb|zeta[i,j]!=0| otherwise.
			Similarly, if providing \verb|relations| or \verb|less| the argument \verb|trans_close| should be \verb|False|
			only if all relations were specified by the
			given data.
		}
	\end{itemize}
	Any extra keyword arguments are passed to \verb|HasseDiagram| (or \verb|hasse_class|
	if specified).


	Function calls to several of the more costly computations are cached. Generally,
	functions in this class do not change the poset but instead return a new poset.
	\verb|Poset| objects may be considered immutable (this is not enforced in any way).
	If you alter a poset you should clear the cache via: \verb|this.cache = {}|.

	@is_section@section_key@0@
	@sections_order@Operations@Subposet Selection@Internal Computations@Queries@Invariants@Maps@Miscellaneous@PosetIsoClass@@
	'''
	__slots__ = ('cache','hasseDiagram','zeta','elements','ranks','name')
	def __init__(this, zeta=None, elements=None, ranks=None, less=None, name='',\
		 hasse_class=None, trans_close=True, relations=None, indices=False, flat_zeta=False, that=None,**kwargs):
		r'''
		See \verb|Poset|.

		@section@Miscellaneous@
		'''
		#defining data in order or priority: Poset instance, zeta function, relations, less function
		if that!=None:
			zeta = that #copy can be keyword or first arg
		if isinstance(zeta, Poset):
			
			for s in Poset.__slots__:
				setattr(this, s, getattr(zeta, s))
			this.cache = {k:v for k,v in zeta.cache.items()}
			this.hasseDiagram = copy.copy(zeta.hasseDiagram)
			this.hasseDiagram.P = this
			return
		elif zeta is not None:
			if isinstance(zeta,TriangularArray):
				this.zeta = zeta	
			else:
				this.zeta = TriangularArray(zeta,flat=flat_zeta)
		elif relations is not None:
			if isinstance(relations,dict):
				this.elements = list(set(itertools.chain(*relations.values(),relations.keys()))) if elements is None else elements
#				this.zeta,this.elements = Poset.zeta_from_relations(relations,this.elements)
			else:
				elems = set() if elements is None else MockSet()
				dict_relations = {}
				for x,y in relations:
					if x not in dict_relations: dict_relations[x] = []
					dict_relations[x].append(y)
					elems.add(x)
					elems.add(y)
				relations = dict_relations
				this.elements = list(elems) if elements is None else list(elements)
			if not indices:
				relations = {this.elements.index(e) : [this.elements.index(f) for f in relations[e]] for e in relations}
			this.zeta,new_order = Poset.zeta_from_relations(relations, this.elements)
			this.elements = [this.elements[i] for i in new_order]
			if ranks is not None:
				ranks = [[new_order.index(i) for i in rk] for rk in ranks]
		elif less is not None:
			assert elements is not None,'`elements` must be provided if specifying a poset via `less`'
			relations = {}
			if ranks is not None:
				if not indices:
					given_less = less
					less = lambda i,j : given_less(elements[i],elements[j])
				for rk in range(len(ranks)-1):
					for i in ranks[rk]:
						relations_i = []
						for j in ranks[rk+1]:
							if less(i,j): relations_i.append(j)
						if len(relations_i)>0: relations[i] = relations_i
			else:
				if indices:
					Less = lambda e,i,f,j:less(i,j)
				else:
					Less = lambda e,i,f,j:less(e,f)
				for i,e in enumerate(elements):
					relations_i = []
					for j,f in enumerate(elements):
						if i==j: continue
						if Less(e,i,f,j): relations_i.append(j)
					if len(relations_i)>0: relations[i] = relations_i
			elements = list(elements)
			this.zeta,new_order = Poset.zeta_from_relations(relations,elements)
			this.elements = [elements[i] for i in new_order]
			if ranks is not None:
				ranks = [[new_order.index(i) for i in rk] for rk in ranks]

		else: #no data provided poset is (possibly empty) antichain
			if elements == None:
				this.zeta = TriangularArray(data=[])
				this.elements = []
			else:
				this.zeta = TriangularArray((1 if i==0 else 0 for i in TriangleRange(len(elements))))
		if not hasattr(this, 'elements'):
			this.elements = list(range(this.zeta.size)) if elements is None else elements
		this.ranks = Poset.make_ranks(this.zeta) if ranks is None else ranks
		this.cache = {}
		this.name = name
		if hasse_class == None: hasse_class = HasseDiagram
		this.hasseDiagram = hasse_class(this, **kwargs)
		if trans_close: 
			Poset.transClose(this.zeta)

	def zeta_from_relations(relations,elements):
		r'''
		Given a dictionary of relations and a list of elements returns the zeta matrix and the elements reordered in a linear extension.

		The dictionary \verb|relations| should have keys that are indices into the list \verb|elements| and values that are lists of indices 
		into \verb|elements|. If an index \verb|i| is contained in the list \verb|relations[j]| then  the $j$th elements is
		less than the $i$th element in the poset.

		@section@Miscellaneous@
		'''
		E = list(range(len(elements)))
		linear_elements = []
		while len(E)>0:
			minimal = [e for e in E if e not in itertools.chain(*(relations[e] for e in relations if e in E))]
			if len(minimal)==0:
				raise ValueError(f"The given relations do not specify a partial order, they violate the anti-symmetry axiom.")
			for e in E:
				if e in minimal:
					E.remove(e)
					linear_elements.append(e)
					break
		#find linear extension
		zeta = []
		for i,e in enumerate(linear_elements):
			if e not in relations:
				zeta+=[1]+[0]*(len(linear_elements)-i-1)
			else:
				zeta+=[1]+[1 if f in relations[e] else 0 for f in linear_elements[i+1:]]
		zeta = TriangularArray(zeta)
		return zeta, linear_elements



	def transClose(T):
		r'''
		Given an instance of \verb|TriangularArray| encoding a (possibly weighted) relation, via $x\sim y$ when the $x,y$ entry is nonzero computes the transitive closure.

		Note, an induced relation $i<j$ is weighted
		by \verb|T[i,k]*T[k,j]| where $j$ and $k$ are
		the minimal indices greater than $i$ satisfying
		the conditions \verb|T[i,k]!=0| and \verb|T[k,j]!=0|.

		@section@Miscellaneous@
		'''
		if len(T.data)==0: return
		for i in range(T.size-1,-1,-1):
			T[i,i] = 1
			for j in range(i+1,T.size):
				if T[i, j]==0: continue
				for k in range(j+1,T.size):
					if T[j, k]!=0 and T[i,k]==0:
						T[i, k] = T[i,j]*T[j,k]
		T.data[-1] = 1
				
		return

	def __str__(this):
		r'''
		Returns a nicely formatted string listing the zeta matrix, the ranks list and the elements of the poset.

		@section@Miscellaneous@
		'''
		ret = ['zeta = ']+[str(this.zeta)]
		if hasattr(this, 'name'):
			ret = [this.name]+ret
		ret.append('ranks = '+str(this.ranks))
		ret.append('elements = '+str(this.elements))
		return '\n'.join(ret)

	def __repr__(this):
		r'''
		Gives a string that can be evaluated to recreate the poset.

		To use the returned string with \verb|eval| the claass \verb|Poset| must be in the namespace and \verb|repr(this.elements)|
		must return a suitable string for evaluation.

		@section@Miscellaneous@
		'''
		return 'Poset(zeta='+repr(this.zeta)+', elements='+repr(this.elements)+', ranks='+repr(this.ranks)+',name='+repr(this.name)+')'

	def __eq__(this,that):
		r'''
		Returns \verb|True| when \verb|that| is a \verb|Poset| representing the same poset as \verb|this| and \verb|False| otherwise.

		@section@Miscellaneous@
		'''
		if not isinstance(that,Poset): return False
		if any(e not in that.elements for e in this) or any(f not in this.elements for f in that): return False
		inds = [that.elements.index(e) for e in this.elements]
		return all(this.zeta[i, j] == that.zeta[tuple(sorted((inds[i], inds[j])))] for i in range(this.zeta.size) for j in range(i,this.zeta.size) )

	def __iter__(this):
		r'''
		Wrapper for \verb|this.elements__iter__|.

		@section@Miscellaneous@
		'''
		return this.elements.__iter__()

	def __getitem__(this, i):
		r'''
		Wrapper for \verb|this.elements.__getitem__|.

		@section@Miscellaneous@
		'''
		return this.elements.__getitem__(i)
	def __contains__(this, p):
		r'''
		Wrapper for \verb|this.elements.__contains__|.

		@section@Miscellaneous@
		'''
		return this.elements.__contains__(p)

	def __len__(this):
		r'''
		Wrapper for \verb|this.elements.__len__|.

		@section@Miscellaneous@
		'''
		return this.elements.__len__()

	##############
	#Operations
	##############
	def adjoin_zerohat(this, label=None):
		r'''
		Returns a new poset with a new minimum adjoined.

		By default the label is the first non-negative integer that is not an element of the poset.

		@section@Operations@
		'''
		zeta = TriangularArray([1 for _ in range(len(this.elements)+1)]+this.zeta.data)
		if label==None:
			label=0
			while label in this.elements:
				label += 1
		assert(not label in this.elements)
		ranks = [[0]]+[[r+1 for r in row] for row in this.ranks]
		elements=[label]+this.elements
		P = Poset(zeta = zeta, elements = elements, ranks = ranks, trans_close = False)
		if 'isRanked()' in this.cache:
			P.cache['isRanked()'] = this.cache['isRanked()']
		if 'isGorenstein()' in this.cache and this.cache['isGorenstein()']:
			P.cache['isGorenstein()'] = False
		if 'isEulerian()' in this.cache and this.cache['isEulerian()']:
			P.cache['isEulerian()'] = False
		return P

	def adjoin_onehat(this, label=None):
		r'''
		Returns a new poset with a new maximum adjoined.

		The label default is the same as \verb|Poset.adjoin_zerohat|

		@section@Operations@
		'''
		zeta = TriangularArray(itertools.chain(*(this.zeta.row(i)+[1] for i in range(this.zeta.size)),[1]))
		if label==None:
			label=0
			while label in this.elements:
				label += 1
		assert(not label in this.elements)
		ranks = [[r for r in row] for row in this.ranks] + [[len(this.elements)]]
		elements=this.elements+[label]
		P = Poset(zeta = zeta, elements = elements, ranks = ranks, trans_close = False)
		if 'isRanked()' in this.cache:
			P.cache['isRanked()'] = this.cache['isRanked()']
		if 'isGorenstein()' in this.cache and this.cache['isGorenstein()']:
			P.cache['isGorenstein()'] = False
		if 'isEulerian()' in this.cache and this.cache['isEulerian()']:
			P.cache['isEulerian()'] = False
		return P

	def identify(this, X, indices=False):
		r'''
		Returns a new poset after making identifications indicated by the argument \verb|X|.

		The new relation is defined by $p\le q$ when there exists any representatives $p'$ and $q'$ such that $p'\le q'$ in the original poset.

		The argument \verb|X| should either be a dictionary where keys are the representatives and the value is a list of elements
		to identify with the key, or a list of lists where the first element of each list is the representative.
		Trivial equivalence classes need not be specified.

		Raises \verb|ValueError| if the identifications do not yield a poset (due to violation of the anti-symmetry axiom).

		@section@Operations@
		'''
		if type(X)==dict:
			X = [[k]+list(v) for (k,v) in X.items()]
		if indices:
			X = [[this.elements[X[i][j]] for j in range(len(X[i]))] for i in range(len(X))]
		X = [[x[0]]+[y for y in x[1:] if y!=x[0]] for x in X]
		removed = list(itertools.chain(*[X[i][1:] for i in range(len(X))]))
		reprs = [X[i][0] for i in range(len(X))]
		elements = [e for e in this.elements if e not in removed]

		def less(p, q):
			if p not in reprs:
				if q not in reprs:
					return this.less(p,q)
				j = reprs.index(q)
				return any([this.less(p, X[j][k]) for k in range(len(X[j]))])
			if p in reprs:
				i = reprs.index(p)
				if q not in reprs:
					return any([this.less(X[i][k], q) for k in range(len(X[i]))])
				j = reprs.index(q)
				return any([this.less(X[i][k], X[j][m]) for k in range(len(X[i])) for m in range(len(X[j]))])

		return Poset(elements = elements, less = less)

	def dual(this):
		r'''
		Returns the dual poset, this has the same elements and has relation $p\le q$ when $q\le p$ in the original poset.

		@section@Operations@
		'''
		P = Poset()
		P.elements = this.elements[::-1]
		P.zeta = this.zeta.revtranspose()
		n = len(this.elements)-1
		P.ranks = [[n-i for i in rk] for rk in this.ranks[::-1]]
		P.hasseDiagram = copy.copy(this.hasseDiagram)
		P.hasseDiagram.P = P
		if 'isRanked()' in this.cache:
			P.cache['isRanked()'] = this.cache['isRanked()']
		if not this.isRanked():
			P.ranks = Poset.make_ranks(P.zeta)
		if 'isEulerian()' in this.cache:
			P.cache['isEulerian()'] = this.cache['isEulerian()']
		if 'isGorenstein()' in this.cache:
			P.cache['isGorenstein()'] = this.cache['isGorenstein()']
		return P

	def element_union(E, F):
		r'''
		Computes the disjoint union of lists \verb|E| and \verb|F|.

		If \verb|E| and \verb|F| have empty intersection return value is \verb|E+F|
		otherwise return value is ($E\times\{0\})\cup(F\times\{1\}$). This is used by operation methods
		such as \verb|Poset.union| and \verb|Poset.starProduct|.

		@section@Operations@
		'''
		if any([e in F for e in E]) or any([f in E for f in F]):
			return [(e,0) for e in E] + [(f,1) for f in F]
		else:
			return E+F

	def union(this, that):
		r'''
		Computes the disjoint union of two posets.

		The labels in the returned poset are determined
		by \verb|element_union|.

		@section@Operations@
		'''
		elements = Poset.element_union(this.elements, that.elements)
		zeta = [this.zeta.row(i)+[0]*len(that.elements) for i in range(this.zeta.size)] + [that.zeta.row(i) for i in range(that.zeta.size)]

		that_ranks = [[r+len(this.elements) for r in rk] for rk in that.ranks]
		if len(this.ranks)>len(that_ranks):
			small_ranks = that_ranks
			big_ranks = this.ranks
		else:
			small_ranks = this.ranks
			big_ranks = that_ranks
		small_ranks = small_ranks + [[] for i in range(len(big_ranks) - len(small_ranks))]

		ranks = [small_ranks[i]+big_ranks[i] for i in range(len(small_ranks))]

		return Poset(zeta, elements, ranks)

	def bddUnion(this, that):
		r'''
		Computes the disjoint union of two posets with maximum and minimums identified, that is, the poset \[\big( (P\wout\{\max P,\min P\})\sqcup(Q\wout\{\max Q,\min Q\})\big)\cup\{\zerohat,\onehat\}.\]

		The labels in the returned poset are the same as in \verb|element_union|.

		@section@Operations@
		'''
		return this.properPart().union(that.properPart()).adjoin_zerohat().adjoin_onehat()

	def bddProduct(this, that):
		r'''
		Computes the Cartesian product of two posets with maximum and minimum adjoined, that is, the poset \[\big((P\wout\{\max P,\min P\})\times(Q\wout\{\max Q,\min Q\})\big)\cup\{\zerohat,\onehat\}.\]

		@section@Operations@
		'''
		return this.properPart().cartesianProduct(that.properPart()).adjoin_zerohat().adjoin_onehat()

	def starProduct(this, that):
		r'''
		Computes the star product of two posets.

		This is the union of \verb|this| with the maximum removed and \verb|that| with the minimum
		removed and all relations $p<q$ for $p$ in \verb|this| and $q$ in \verb|that|.

		@section@Operations@
		'''
		that_part = that.complSubposet(that.min())
		this_part = this.complSubposet(this.max())
		elements = Poset.element_union(this_part.elements, that_part.elements)
		zeta = itertools.chain(*(this.zeta.row(i)+[1]*len(that_part.elements) for i in range(this.zeta.size)), that_part.zeta.data)
		ranks = this_part.ranks + [[r+len(this_part.elements) for r in rk ] for rk in that_part.ranks]

		return Poset(zeta, elements, ranks, flat_zeta=True)

	def cartesianProduct(this, that):
		r'''
		Computes the Cartesian product.

		@section@Operations@
		'''
		elements = list(itertools.product(this.elements,that.elements))# in this.elements for q in that.elements]
		ranks = [[] for i in range(len(this.ranks)+len(that.ranks)-1)]
		for i in range(len(this.elements)):
			for j in range(len(that.elements)):
				ranks[this.rank(i,True)+that.rank(j,True)].append(i*len(that.elements)+j)


		thiszeta = this.zeta
		thatzeta = that.zeta
		pair_inds = list(itertools.product(range(len(this.elements)),range(len(that.elements))))	
		zeta = [(0 if pair_inds[i][0]>pair_inds[j][0] else thiszeta[pair_inds[i][0],pair_inds[j][0]]) * (0 if pair_inds[i][1] > pair_inds[j][1] else thatzeta[pair_inds[i][1],pair_inds[j][1]]) for i in range(len(pair_inds)) for j in range(i,len(pair_inds))]
		return Poset(elements=elements, ranks=ranks, zeta=zeta,flat_zeta=True)

	def diamondProduct(this, that):
		r'''
		Computes the diamond product which is the Cartesian product of the two posets with their minimums removed and then adjoined with a new minimum.

		@section@Operations@
		'''
		return this.complSubposet(this.min()).cartesianProduct(that.complSubposet(that.min())).adjoin_zerohat()

	def pyr(this):
		r'''
		Computes the pyramid of a poset, that is, the Cartesian product with a length 1 chain.

		@section@Operations@
		'''
		return this.cartesianProduct(Chain_1())

	def prism(this):
		r'''
		Computes the prism of a poset, that is, the diamond product with \verb|Cube(1)|.

		@section@Operations@
		'''
		return this.diamondProduct(Cube_1())
	##############
	#End Operations
	##############
	##############
	#Queries
	##############
	@cached_method
	def isRanked(this):
		r'''
		Checks whether the poset is ranked.

		@section@Queries@
		'''
		return all(all(this.rank(v)==this.rank(k)+1 for v in V)for (k,V) in this.covers().items())

	@cached_method
	def isEulerian(this):
		r'''
		Checks whether the given poset is Eulerian (every interval with at least 2 elements has an equal number of odd and even rank elements).

		@section@Queries@
		'''
		if 'isRanked()' in this.cache and this.cache['isRanked()']==False: return False
		bottom = this.min()
		top = this.max()
		if len(bottom)!=1 or len(top)!=1:
			return False

		for p in this.elements:
			for q in this.elements:
				if this.less(p,q):
					if this.mobius(p,q)!=(-1)**(this.rank(q)-this.rank(p)):
						return False
		return True

	@cached_method
	def isGorenstein(this):
		r'''
		Checks if the poset is Gorenstein*.

		A poset is Gorenstein* if the proper part of all intervals with more than
		two elements have sphere homology. In other words, this function checks that
		\begin{center}
			\verb|this.interval(p,q).properPart().bettiNumbers()|
		\end{center}
		is either \verb|[2]| or \verb|[1,0,...,0,1]| for all $p\le q$ such that $\abs{\text{rk}(q)-\text{rk}(p)}\ge2$.

		@section@Queries@
		'''
		if len(this.ranks)==1: return len(this.elements)==1
		if len(this.ranks)==2: return len(this.elements)==2
		def check_homol(P):
			if len(P.elements)==2: return True
			b = P.properPart().bettiNumbers()
			return b==[2] or (b[0] == 1 and b[-1] == 1 and all([x == 0 for x in b[1:-1]]))
		return all([check_homol(this.interval(i,j,True)) for i in range(len(this.elements)) for j in range(i+2,len(this.elements)) if this.zeta[i,j]!=0])

	@cached_method
	def isLattice(this):
		r'''
		Checks if the poset is a lattice.

		Note, returns \verb|True| if \verb|this| is an empty poset.

		@section@Queries@
		'''
		if len(this.min()) > 1: return False
		if len(this.max()) > 1: return False
		for i in range(len(this)):
			for j in range(len(this)):
				if this.join(i,j,True)==None: return False
		return True

	@cached_method
	def covers(this, indices=False):
		r'''
		Returns the list of covers of the poset.

		An element $q$ covers $p$ when $p<q$ and $p<r<q$ implies $r=p$ or $r=q$.

		@section@Queries@
		'''
		#isRanked calls covers so we can't call isRanked,
		#but if its cached calling it doesn't really call it
		if 'isRanked()' in this.cache and this.isRanked():
			ret = {}
			for i in set(range(len(this))).difference(this.max(indices=True)):
				p = i if indices else this.elements[i]
				ret[p] = []
				for j in this.ranks[this.rank(p,indices)+1]:
					q = j if indices else this.elements[j]
					if this.less(i,j,indices=True):
						ret[p].append(q)
			return ret
		else:
			ret = {}
			for i in range(len(this.elements)):
				zetai = this.zeta.row(i)
				filt = []
				icovers=[]
				for j in range(1,len(this.elements)-i):
					if zetai[j]!=0:
						if all(this.zeta[k+i,j+i]==0 for k in filt):
							icovers.append(j+i)
						filt.append(j)
				if len(icovers)>0: ret[i] = icovers
			if not indices:
				return {this.elements[i] : [this.elements[j] for j in ret[i]] for i in ret}
			return ret
							
						
						
					
#			if indices:
#				P = this.relabel()
#				return {i : P.filter((i,),strict=True).min() for i in P if i not in P.max()}
#			else:
#				return {p : this.filter((p,),strict=True).min() for p in this if p not in this.max(indices=indices)}
	##############
	#End Queries
	##############
	##############
	#Subposet selection
	##############
	@cached_method
	def min(this, indices=False):
		r'''
		Returns a list of the minimal elements of the poset.

		@section@Subposet Selection@
		'''
		z = this.zeta.data
		ret_indices = [j for j in range(this.zeta.size) if all(x==0 for x in list(this.zeta.col(j))[:-1])]
		return ret_indices if indices else [this.elements[i] for i in ret_indices]
	@cached_method
	def max(this, indices=False):
		r'''
		Returns a list of the maximal elements of the poset.

		@section@Subposet Selection@
		'''
		ret_indices = [i for i in range(this.zeta.size) if all(m==0 for m in this.zeta.row(i)[1:])]
		return ret_indices if indices else [this.elements[i] for i in ret_indices]

	def complSubposet(this, S, indices=False):
		r'''
		Returns the subposet of elements not contained in \verb|S|.

		@section@Subposet Selection@
		'''
		if not indices:
			S = [this.elements.index(s) for s in S]
		return this.subposet([i for i in range(this.zeta.size) if i not in S],True)


	def subposet(this, S, indices=False, keep_hasseDiagram=True):
		r'''
		Returns the subposet of elements in \verb|S|.

		@section@Subposet Selection@
		'''
		if not indices:
			S = [this.elements.index(s) for s in S]
		#TODO don't force sort S but ensure it is a linear extension?
		S = sorted(S)
		elements = [this.elements[s] for s in S]
		zeta = this.zeta.subarray(S)
		P = Poset(zeta, elements,trans_close=False)
		if keep_hasseDiagram:
			P.hasseDiagram = copy.copy(this.hasseDiagram)
			P.hasseDiagram.P = P
		return P

	def interval(this, i, j, indices=False):
		r'''
		Returns the closed interval $[i,j]$.

		@section@Subposet Selection@
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
		zeta = this.zeta
		element_indices = [k for k in range(i,j+1) if zeta[i,k]!=0 and zeta[k,j]!=0]

		return this.subposet(element_indices, indices=True)

	def filter(this, x, indices=False, strict=False):
		r'''
		Returns the subposet of elements greater than or equal to any element of \verb|x|.

		If \verb|strict| is \verb|True| then the element \verb|x| is not included in the returned poset and
		if it is \verb|False| the element \verb|x| is included.

		If \verb|indices| is \verb|True| then the element \verb|x| is interpreted as indices into the poset,
		either way the return value is a subposet of the original poset.

		Note, \verb|x| should not be an element of the poset but an iterable of elements; to construct a principal
		filter for an element \verb|p| of a poset \verb|P| use \verb|P.filter((p,))|.

		@section@Subposet Selection@
		'''
		if not indices: x = [this.elements.index(p) for p in x]
		m = min(x)
		elements = [i for i in range(m,len(this.elements)) if any(this.zeta[y, i]!=0 for y in x)]
		if strict:
			for y in x: elements.remove(y)
		ret = this.subposet(elements,True)
		return ret

	def ideal(this, x, indices=False, strict=False):
		r'''
		Returns the subposet of elements less than or equal to any element of \verb|x|.

		See \verb|Poset.filter| for an explanation of the arguments.

		@section@Subposet Selection@
		'''
		if not indices: x = [this.elements.index(p) for p in x]
		m = max(x)
		elements = [i for i in range(m+1) if any(this.zeta[i,y]!=0 for y in x if i<=y)]
		if strict:
			for y in x: elements.remove(y)
		ret = this.subposet(elements,True)
		return ret

	@cached_method
	def properPart(this):
		r'''
		Returns the subposet of all elements that are neither maximal nor minimal.

		@section@Subposet Selection@
		'''
		return this.complSubposet(this.max()+this.min())

	def rankSelection(this, S):
		r'''
		Returns the subposet of elements whose rank is contained in \verb|S|.

		Note, this does not automatically include the minimum or maximum.

		@section@Subposet Selection@
		'''
		ret = this.subposet([x for x in itertools.chain(*[this.ranks[s] for s in S])], indices=True)
		if 'isRanked()' in this.cache and this.isRanked():
			ret.cache['isRanked()'] = True
		return ret
	##############
	#End Subposet selection
	##############
	##############
	#Internal computations
	##############
	def less(this, i, j, indices=False):
		r'''
		Returns whether $i$ is strictly less than $j$.

		@section@Internal Computations@
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)

		return i < j and this.zeta[i, j]!=0

	def lesseq(this, i, j, indices=False):
		r'''
		Returns whether $i$ is less than or equal to $j$.

		@section@Internal Computations@
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
		return i<=j and this.zeta[i, j]!=0

	def isAntichain(this, A, indices=False):
		r'''
		Returns whether the given set is an antichain ($i\not<j$ for all $i$ and $j$).

		@section@Internal Computations@
		'''
		if not indices: A = [this.elements.index(a) for a in A]
		return all(x==0 or y==0 for x,y in zip(this.zeta.subarray(A).data,TriangleRange(len(A))))
		return all(x==0 for x in this.zeta.data)

	@cached_method
	def join(this, i, j, indices=False):
		r'''
		Computes the join of $i$ and $j$, if it does not exist returns \verb|None|.

		@section@Internal Computations@
		'''
		def _join(i,j,M):
			i,j=sorted((i,j))
			if this.zeta[i,j]!=0: return j
			m = [x for x in range(j+1,len(this.elements)) if this.zeta[i,x]!=0 and this.zeta[j,x]!=0]

			if len(m)==0: return None
			x=min(m)
			if all(this.zeta[x,y]!=0 for y in m): return x
			return None

		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
			ret = _join(i, j, this.zeta)
			if ret == None: return None
			return this.elements[ret]
		return _join(i, j, this.zeta)

	@cached_method
	def mobius(this, i=None, j=None, indices=False):
		r'''
		Computes the value of the M\"obius function from $i$ to $j$.

		If neither $i$ nor $j$ are provided returns all values
		of the M\"obius function as an instance of
		\verb|TriangularArray|.

		If $i$ (respectively $j$) is not provided then it
		is assumed to be the minimum (maximum) and raises 
		an exception of type \verb|ValueError| if the poset
		does not have a unique minimum (maximum).

		@section@Internal Computations@
		'''
		if i is None and j is None:
			return this.zeta.inverse()
		if i is None:
			mins = this.min(indices)
			if len(mins)>1: raise ValueError("No unique minimum and argument j was provided, argument i must be provided")
			i=mins[0]
		if j is None:
			maxes = this.max(indices)
			if len(maxes)>1: raise ValueError("No unique maximum and argument i was provided, argument j must be provided")
			j=maxes[0]
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
		if 'mobius(None, None, False)' in this.cache:
			return this.cache['mobius(None, None, False)'][i,j]
		if i == j: return 1
		if not this.less(i,j,True) and not this.less(j,i,True): return 0

		@cached_method
		def calc_mobius(this, i,j):
			if i == j: return 1
			ret = 1
			for k in range(i+1,j):
				if this.zeta[i,k]!=0 and this.zeta[k,j]!=0:
					ret += calc_mobius(this, i,k)
			return -ret

		return calc_mobius(this, i,j)

	@cached_method
	def rank(this, i, indices=False):
		r'''
		Returns the length of $i$ (the length of the longest chain ending at $i$).

		Returns \verb|None| if \verb|i| is not an element
		(or \verb|i| is not a valid index if \verb|indices| is \verb|True|).

		@section@Internal Computations@
		'''
		if not indices:
			i = this.elements.index(i)

		for j in range(len(this.ranks)):
			if i in this.ranks[j]: return j
		return None
	##############
	#End Internal computations
	##############
	##############
	#Invariants
	##############

	def fVector(this):
		r'''
		Returns flag $\overline{f}$-vector as a dictionary with keys $S\subseteq[n]$.

		This method is intended for use with a poset, possibly quasi-graded, that has a unique minimum and maximum. On a classical poset this counts chains that contain the first
		minimal element, \verb|this[this.ranks[0][0]]| and ignores the
		final rank. 
		
		Chains are weighted by the product of the zeta
		values along the chain.

		@section@Invariants@
		'''
		n = len(this.ranks)-2
		def fVectorCalc(ranks,S,M, i, count, weight):
			newCount = count
			if S == tuple():
				return weight * M[i, ranks[-1][0]]
			for j in ranks[S[0]]:
				if i<j and M[i, j]!=0:
					newCount += fVectorCalc(ranks, S[1:], M, j, count, weight*M[i, j])
			return newCount
		f = {tuple():this.zeta[this.ranks[0][0],this.ranks[-1][0]]}

		if n<=0: return f

		for S in subsets(range(1,n+1)):
			f[tuple(S)]=fVectorCalc(this.ranks,S,this.zeta,this.ranks[0][0],0,1)
		return f

	@cached_method
	def hVector(this):
		r'''
		Returns the flag $\overline{h}$-vector of the poset.

		@section@Invariants@
		'''
		f = this.fVector()
		h = {}
		for S in subsets(range(1,len(this.ranks)-1)):
				h[S] = (-1)**len(S) * sum((-1)**len(T)*f[T] for T in subsets(S))
		return h
	@cached_method
	def flagVectors(this):
		r'''
		Returns the table of flag $\overline{f}$- and $\overline{h}$-vectors as a dictionary with keys $S\subseteq[n]$ encoded as tuples and with elements \verb|(f_S, h_S)|.

		@section@Invariants@
		'''
		f = this.fVector()
		h = this.hVector()
		return {S:[f[S],h[S]] for S in f}

	def sparseKVector(this):
		r'''
		Returns the sparse $k$-vector $k_S = \sum_{T\subseteq S}(-1)^{\abs{S\wout T}}\overline{h}_T$.

		The sparse $k$-vector only has entries $k_S$ for sparse sets $S$,
		that is, sets $S\subseteq[\text{rk}(P)-1]$ such that if $i\in S$ then $i+1\not\in S$.
		The sparse $k$-vector is returned as a dictionary whose keys are tuples.

		@section@Invariants@
		'''
		n = len(this.ranks)-2
		def sparseSubsets(X):
			'''
			Generator for all the sparse subsets of a given list.
			'''
			if len(X)<=1:
				yield tuple()
				return
			if len(X)==2:
				yield tuple(X[:1])
				yield tuple()
				return
			for i in range(len(X[:-1])-1,-1,-1):
				for S in sparseSubsets(X[i+2:]):
					yield itertools.chain(itertools.repeat(X[i],1), S)
			yield tuple()

		def fVectorCalc(ranks,S,M,i,count,f,truncated_S,weight=1):
			newCount = count
			if len(S)==0: return weight*M[ranks[0][0],i]
			for j in ranks[S[-1]]:
				if j<i and M[j, i]!=0:
					newCount += fVectorCalc(ranks, S[:-1], M, j, count, f,truncated_S+(S[-1],),weight*M[j, i])
			return newCount
		fVector = {}
		for S in sparseSubsets(tuple(range(1,n+1))):
			S = tuple(S)
			if S in fVector: break
			fVector[S] = fVectorCalc(this.ranks,S,this.zeta,this.ranks[-1][0],0,fVector,tuple())

		kVector = {}
		for S in fVector:
			kv = 0
			for T in subsets(S):
#			for T in itertools.chain(*(itertools.combinations(S,k) for k in range(len(S)+1))):
				T = tuple(T)
				l = len(S)-len(T)
				kv += (1 if l%2==0 else -1) * (fVector[T]*(1<<l))
			kVector[S] = kv
		return kVector


	def flagVectorsLatex(this,standalone=False):
		r'''
		Returns a string of latex code representing the table of flag vectors of the poset.

		Note, the package \href{https://ctan.org/pkg/longtable}{longtable} is required to compile the output.

		@section@Invariants@
		'''
		table = this.flagVectors()
		ret = []
		if standalone:
			ret.append('\\documentclass[article]\n\\usepackage{longtable}\n\\begin{document}\n')
		ret.append("\\begin{longtable}{c|c|c}\n\t$S$&$f_S$&$h_S$\\\\\n\t\\hline\n\t\\endhead\n")
		for t in table:
			ret.append("\t\\{")
			ret.append(str(t)[1:-1])
			ret+=["\\} & ",str(table[t][0])," & ",str(table[t][1]),"\\\\\n\t\\hline\n"]
		ret.append("\\end{longtable}")
		if standalone: ret.append('\n\\end{document}')
		return ''.join(ret)


	@cached_method
	def abIndex(this):
		r'''
		Returns the \text{ab}-index of the poset as an instance of \verb|Polynomial|.

		If the poset has a unique minimum and maximum but isn't ranked
		this computes the \text{ab}-index considering the poset to be
		quasigraded (in the sense of \cite{ehrenborg-goresky-readdy-15}) with $\overline{\zeta}=\zeta$ and $\rho$ the rank function $x\mapsto$\verb|P.rank(x)|. 

		For more information on the \text{ab}-index see \cite{bayer-21}

		@section@Invariants@
		'''
		ab = {}
		fh = this.flagVectors()
		for x in fh:
			u = ['a']*(len(this.ranks)-2)
			for s in x: u[s-1] = 'b'
			ab[''.join(u)] = fh[x][1]

		return Polynomial(ab)

	@cached_method
	def cdIndex(this):
		r'''
		Returns the \textbf{cd}-index as an instance of \verb|Polynomial|.

		If the given poset
		does not have a \textbf{cd}-index then a \textbf{cd}-polynomial is still
		returned, but this is not meaningful. If you wish to check whether a poset has
		a \textbf{cd}-index then check the Boolean below:
		\begin{center}
			\verb|this.abIndex().cdToAb()!=this.abIndex()|
		\end{center}

		If the given poset is semi-Eulerian then the \textbf{cd}-index
		as defined in \cite{juhnke-kubitzke-24} is computed.

		This function correctly computes the \cv\dv-index
		of a quasigraded Eulerian poset (see \cite{ehrenborg-goresky-readdy-15}).

		For computation we use the sparse $k$-vector formula see \cite[Proposition 7.1]{billera-ehrenborg-00}. For more info on the \textbf{cd}-index see \cite{bayer-21}.

		@section@Invariants@
		'''
		def cdMonom(W,n):
			if len(W)==0: return 'c'*n
			ret = []
			num_d = 0
			i = 1
			while i <= n:
				if i in W:
					ret.append('d')
					i += 1
				else:
					ret.append('c')
				i += 1
			return ''.join(ret)
		def theOrder(X,Y):
			return len(X)==0 or (X[0]<=Y[0] and all(X[i]<=Y[i] and X[i]>=Y[i-1]+2 for i in range(1,len(X))))
		phi = Polynomial()
		k = this.sparseKVector()
		n = len(this.ranks)-2
		for S in k:
			coeff = 0
			for T in k:
				if len(T)!=len(S): continue
				if not theOrder(T,S): continue
				coeff += (-1)**(sum(T)+sum(S)) * k[T]
			phi[cdMonom(S,n)] = coeff
		return phi

	@cached_method
	def bettiNumbers(this):
		r'''
		Computes the Betti numbers of the poset, that is, the ranks of the homology of the order complex (the simplicial complex of all chains).

		@section@Invariants@
		'''
		chains = this.chains()[1:] #nonempty chains
		C = [[] for i in range(max([len(c) for c in chains]))]
		for c in chains: C[len(c)-1].append(c)
		if len(C)==1: return [len(C[0])]

		def e(i,n):
			return np.array([0]*(i)+[1]+[0]*(n-i-1))

		def bddyMap(i):
			dom = C[i]
			cod = C[i-1]

			images = [[[c[:i]+c[i+1:],(-1)**i] for i in range(len(c))] for c in dom]
			mat = [sum([e(cod.index(c),len(cod))*coeff for (c,coeff) in images[i]]) for i in range(len(images))]
			return np.transpose(np.array(mat))

		#don't do bddyMap(0) cause it's a zero map and np can't take rank of a zero map
		bddy = [bddyMap(i) for i in range(1,len(C))]
		bddy_rks = [np.linalg.matrix_rank(b) for b in bddy]
		ret = []
		#compute homology ranks for all interior maps
		for i in range(0,len(bddy)-1):
			ret.append(np.shape(bddy[i])[1] - bddy_rks[i] - bddy_rks[i+1])
		#add highest homology rank
		ret.append(np.shape(bddy[-1])[1] - bddy_rks[-1])
		#add lowest homology rank and return
		return [len(C[0]) - np.linalg.matrix_rank(bddy[0])]+ret
	##############
	#End Invariants
	##############
	##############
	#Maps
	##############
	def buildIsomorphism(this, that, indices=False):
		r'''
		Returns an isomorphism from \verb|this| to \verb|that| as a dictionary
		or \verb|None| if the posets are not isomorphic.

		If \verb|indices| is \verb|True| then the dictionary keys and values
		are indices, otherwise they are elements.

		@section@Maps@
		'''
		d1 = {i: this.rank(i,indices=True) for i in range(len(this))}
		d2 = {i: that.rank(i,indices=True) for i in range(len(that))}
		if collections.Counter(d1.values()) != collections.Counter(d2.values()): return None
		this_ranks = [0 for i in this]
		for r in range(len(this.ranks)):
			for i in this.ranks[r]: this_ranks[i] = r
		that_ranks = [0 for i in that]
		for r in range(len(that.ranks)):
			for i in that.ranks[r]: that_ranks[i] = r
		def set_comp_rks(P,d,ranks):
			for i in range(len(P)):
				row = P.zeta.row(i)
				col = list(P.zeta.col(i))
				upset = frozenset(collections.Counter(ranks[i+j] for j in range(len(row)) if row[j] != 0).items())
				downset = frozenset(collections.Counter(ranks[j] for j in range(len(col)) if col[j] != 0).items())
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
		if set(d1.values())!=set(d2.values()): return None
		d1Inv = invert(d1)
		d2Inv = invert(d2)


		def nextchoice(map, i, jstart):
			if i==len(this): return map, -1, True
			cands = d2Inv[d1[i]]
			for j_ in range(jstart,len(cands)):
				j = cands[j_]
				if j in [m[1] for m in map]: continue
				if all(this.zeta[tuple(sorted((m[0],i)))] == that.zeta[tuple(sorted((m[1],j)))] for m in map):
					return map+[[i,j]], j_+1, False
			return None, j_+1, True

		def prevchoice(map, i, jstarts):
			while i>=0 and jstarts[i] >= len(d2Inv[d1[i]]):
				jstarts[i] = 0
				i -= 1
			if i==-1:
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


	def is_isomorphic(this, that):
		r'''
		Returns \verb|True| if the posets are isomorphic and \verb|False| otherwise.

		@section@Maps@
		'''
		return this.buildIsomorphism(that) != None
	##############
	#End Maps
	##############
	##############
	#Misc
	##############
	@cached_method
	def __hash__(this):
		r'''
		Hashes the poset, dependent on \verb|this.elements| and relations between
		ranks.

		@section@Miscellaneous@
		'''
		return hash(frozenset((e,f,0 if i>j else this.zeta[i,j]) for i,e in enumerate(this.elements) for j,f in enumerate(this.elements)))

	def copy(this):
		r'''
		Returns a shallow copy of the poset.

		Making a shallow copy via the copy module i.e. \verb|Q = copy.copy(P)|
		doesn't update the self reference in \verb|Q.hasseDiagram| (in this example
		\verb|Q.hasseDiagram.P| is \verb|P|). This doesn't matter if you treat posets as immutable,
		but otherwise could cause issues when displaying or generating hasse diagrams.
		The returned poset has the self reference updated.

		@section@Miscellaneous@
		'''
		P = copy.copy(this)
		P.hasseDiagram.P = P
		return P

	def latex(this, **kwargs):
		r'''
		Returns a string of tikz code to draw the Hasse diagram of the poset for use in a \LaTeX{} document.

		This is a wrapper for \verb|HasseDiagram.latex|.

		For a full list of keyword arguments see \verb|HasseDiagram|. The most common arguments are:
		\begin{itemize}

			\item[]{\verb|height| -- The height in tikz units of the diagram.

				The default value is 10.
				}
			\item[]{\verb|width| -- The width in tikz units of the diagram.

				The default value is 8.
				}

			\item[]{\verb|labels| -- If \verb|False| elements are represented by filled circles.
				If \verb|True| by default elements are labeled by the result
				of casting the poset element to a string.

				The default value is \verb|True|.
				}

			\item[]{\verb|ptsize| -- When labels is \verb|False| this is the size of the circles used
				to represent elements. This has no effect if labels is \verb|True|.

				The default value is \verb|'2pt'|.
				}

			\item[]{\verb|nodescale| -- Each node is wrapped in \verb|'\\scalebox{'+nodescale+'}'|.

				The default value is \verb|'1'|.
				}

			\item[]{\verb|standalone| -- When \verb|True| a preamble is added to the beginning and
				\verb|'\\end{document}'| is added to the end so that the returned string
				is a full \LaTeX{} document that can be compiled. Compiling requires
				the \LaTeX{} packages tikz (pgf) and preview. The resulting figure can be
				incorporated into another \LaTeX{} document with \verb|\includegraphics|.

				When \verb|False| only the code for the figure is returned; the return value
				begins with \verb|\begin{tikzpicture}| and ends with \verb|\end{tikzpicture}|.

				The default is \verb|False|.
				}

			\item[]{\verb|nodeLabel| -- A function that takes the \verb|HasseDiagram| object and an index
				and returns the label for the indicated element as a string.
				For example, the default implementation \verb|HasseDiagram.nodeLabel|
				returns the element cast to a string and is defined as below:
				\begin{center}\begin{verbatim}
					def nodeLabel(H, i):
						return str(H.P[i])
				\end{verbatim}\end{center}

				Note \verb|H.P| is \verb|this|.
			}
		\end{itemize}

		@section@Miscellaneous@
		'''
		return this.hasseDiagram.latex(**kwargs)

	def img(this, tmpfile='a.tex', tmpdir=None, **kwargs):
		r'''
		Produces latex code (via calling \verb|latex|) compiles it with pdflatex and returns a \verb|wand.image.Image| object constructed from the pdf.

		In a Jupyter notebook calling \verb|display| on the return value will show the Hasse diagram in the output cell.
		By default \verb|tmpdir| is \verb|tempfile.gettempdir()|.

		This function converts the compiled pdf to an image using imagemagick, this may fail due to imagemagick's default security policies.
		For more info and how to fix the issue see \url{https://askubuntu.com/questions/1127260/imagemagick-convert-not-allowed}

		Keyword arguments are passed to \verb|latex()| but \verb|standalone| is alwasy set
		to \verb|True| (otherwise the pdf would not compile).
		Note this function may hang if \verb|pdflatex| fails
		to compile.

		@section@Miscellaneous@
		'''
		from wand.image import Image as WImage
		import os
		import tempfile
		if tmpdir==None: tmpdir = tempfile.gettempdir()
		kwargs['standalone']=True #otherwise it won't compile
		with open(os.path.join(tmpdir, tmpfile),'w') as f: f.write(this.latex(**kwargs))
		os.system('echo "" | pdflatex --output-directory={} {} >/dev/null 2>&1'.format(tmpdir,os.path.join(tmpdir,tmpfile)))
		return WImage(filename=os.path.join(tmpdir,tmpfile.replace('.tex','.pdf')))

	def show(this, **kwargs):
		r'''
		Opens a window displaying the Hasse diagram of the poset.

		This is a wrapper for \verb|HasseDiagram.tkinter|.

		For a full list of keyword arguments see \verb|HasseDiagram|. The most common arguments are:
		\begin{itemize}
			\item[]{\verb|height| -- The height of the diagram.

				The default value is 10.
			}

			\item[]{\verb|width| -- The width of the diagram.

				The default width is 8.
			}

			\item[]{\verb|labels| -- If \verb|False| elements are represented as filled circles.
				If \verb|True| by default elements are labeled by the result of
				casting the poset element to a string.

				The default value is \verb|True|.
			}

			\item[]{\verb|ptsize| -- When labels is \verb|False| controls the size of the circles
				representing elements. This can be an integer or a string,
				if the value is a string the last two characters are ignored.

				The default value is \verb|'2pt'|.
			}

			\item[]{\verb|scale| -- Scale of the diagram.

				The default value is 1.
			}

			\item[]{\verb|padding| -- A border of this width is added around all sides of the diagram.

				The default value is 1.
			}

			\item[]{\verb|nodeLabel| -- A function that takes the \verb|HasseDiagram| object and an index
				and returns the label for the indicated element as a string.
				For example, the default implementation \verb|HasseDiagram.nodeLabel|
				returns the element cast to a string and is defined as below:
				\begin{center}\begin{verbatim}
					def nodeLabel(H, i):
						return str(H.P[i])
				\end{verbatim}\end{center}
				Note \verb|H.P| is \verb|this|.
			}
		\end{itemize}

		@section@Miscellaneous@
		'''
		return this.hasseDiagram.tkinter(**kwargs)

	def chains(this, indices=False):
		r'''
		Returns a list of all nonempty chains of the poset (subsets $p_1<\dots<p_r$).

		@section@Miscellaneous@
		'''
		def expand(C):
			ret = []
			for c in C:
				for i in range(len(this.elements)):
					if this.less(c[-1], i, indices=True):
						ret.append(c+(i,))
			return ret

		C = set([(i,) for i in range(len(this.elements))])
		new_C = C.union(expand(C))

		while len(C) != len(new_C):
			C = new_C
			new_C = C.union(expand(C))
		if indices: return list(C)

		return [tuple()]+[tuple([this.elements[i] for i in c]) for c in C]

	def orderComplex(this, indices=False):
		r'''
		Returns the poset of all chains ordered by inclusion.

		@section@Invariants@
		'''

		elements = sorted(this.chains(indices),key=len)

		ranks = [[] for i in range(max([len(c) for c in elements])+1)]
		for i in range(len(elements)):
			ranks[len(elements[i])].append(i)

		zeta = TriangularArray(1 if (i==j or all(x in elements[j] for x in elements[i])) else 0 for i in range(len(elements)) for j in range(i,len(elements)))

		args = {'elements': elements, 'ranks': ranks, 'zeta': zeta, 'trans_close': False}
		if hasattr(this,'name'): args['name'] = 'Order complex of '+this.name
		P = Poset(**args)

		return P

	def relations(this, indices=False):
		r'''
		Returns a list of all pairs $(e,f)$ where $e\le f$.

		@section@Miscellaneous@
		'''
		if indices:
			return [(i,j) for i in range(len(this.elements)-1) for j in range(i+1,len(this.elements)) if this.zeta[i, j] != 0] 

		return [(this.elements[i],this.elements[j]) for i in range(len(this.elements)-1) for j in range(i+1,len(this.elements)) if this.zeta[i, j] != 0]

	def relabel(this, elements=None):
		r'''
		Returns a new \verb|Poset| object with
		the \verb|elements| attribute as given.

		If \verb|elements| is \verb|None| then
		the returned poset has \verb|elements| attribute set to \verb|list(range(len(this)))|.

		@section@Miscellaneous@
		'''
		if elements == None:
			elements = list(range(len(this)))
		ret = Poset(this)
		ret.elements = elements
		return ret

	def reorder(this, perm, indices=False):
		r'''
		Returns a new \verb|Poset| object (representing the same poset) with the elements reordered.

		\verb|perm| should be a list of elements if \verb|indices| is \verb|False| or a list of indices if \verb|True|.
		The returned poset has elements in order of a linear
		extension created from the given permutation.
		If the list \verb|perm| defines a linear
		extension then \verb|perm[i]| is the $i$th element.

		If \verb|perm| is not a linear extension it is coerced into a
		linear extension by repeatedly removing the first
		element whose ideal has already been removed.

		@section@Miscellaneous@
		'''
		if not indices:
			perm = [this.elements.index(p) for p in perm]
		else:
			perm = list(perm)
		#########################################################
		zeta = this.zeta
		linear_perm = []
		while len(perm)>0:
			for ip,p in enumerate(perm):
				col = list(zeta.col(p))[:-1]
				if all(i in linear_perm for i,coli in enumerate(col) if coli==1):
					linear_perm.append(p)
					del perm[ip]
					break
		perm = linear_perm
		#########################################################
		zeta = TriangularArray((this.zeta[i,j] if i<j else this.zeta[j,i] for ii,i in enumerate(perm) for j in perm[ii:]))
		ranks = [sorted([perm.index(i) for i in rk]) for rk in this.ranks]

		P = Poset(elements = [this.elements[i] for i in perm], zeta = zeta, ranks = ranks, trans_close = False)
		P.hasseDiagram = this.hasseDiagram
		P.hasseDiagram.P = P

		return P


	def sort(this, key = None, indices=False):
		r'''
		Returns a new \verb|Poset| object (representing the same poset) with the elements sorted.
		
		If the given key does not deifine a linear
		extension then the ordering is coerced
		to one in the same way as \verb|reorder|.

		@section@Miscellaneous@
		'''
		if indices:
			perm = sorted(this.elements, key = lambda p: key(this.elements.index(p)) )
		else:
			perm = sorted(this.elements,key=key)
		return this.reorder(perm)

	def shuffle(this):
		r'''
		Returns a new \verb|Poset| object (representing the same poset) with the elements in a random order.

		@section@Miscellaneous@
		'''
		perm = list(range(len(this)))
		random.shuffle(perm)
		return this.reorder(perm, True)

	def toSage(this):
		r'''
		Converts the poset to an instance of \verb|sage.combinat.posets.posets.FinitePoset|.

		@section@Miscellaneous@
		'''
		import sage
		return sage.combinat.posets.posets.Poset((this.elements,this.relations()),facade=False)
	def fromSage(P):
		r'''
		Convert an instance of \verb|sage.combinat.posets.poset.FinitePoset| to an instance of \verb|Poset|.

		@section@Miscellaneous@
		'''
		rels = [[x,y] for x,y in P.relations() if x!=y]
		return Poset(relations=rels)

	def make_ranks(zeta):
		r'''
		Used by the constructor to compute the ranks list for a poset when it isn't provided.

		@section@Miscellaneous@
		'''
		if zeta.size==0: return []
		left = list(range(zeta.size))
		preranks = {} #keys are indices values are preranks
		while len(left)>0:
			new_left = list(left)
			for l in left:
				zetal = list(zeta.col(l))[:-1]
				loi = [i for i in range(len(zetal)) if zetal[i]!=0]
				if all(i in preranks for i in loi):
					preranks[l] = 1+max([-1]+[preranks[i] for i in loi])
					new_left.remove(l)
			left = new_left
		ranks = [[] for _ in range(1+max(preranks.values()))]
		for k,v in preranks.items():
			ranks[v].append(k)
		return ranks
	def isoClass(this):
		r'''
		Returns an instance of \verb|PosetIsoClass| representing the isomorphism class
		of the poset.

		@section@Miscellaneous@
		'''
		return PosetIsoClass(this)
	##############
	#End Misc
	##############
##########################################
#End Poset Class
##########################################
##############
#Poset Iso Class
##############
class PosetIsoClass(Poset):
	r'''
	This class encodes the isomorphism type of a poset.

	Internally, this class inherits from \verb|Poset| and thus instances of
	\verb|PosetIsoClass| are also instances of \verb|Poset|.
	The major differences between this class
	and \verb|Poset| are that \verb|PosetIsoClass.__eq__| returns \verb|True|
	when the two posets are isomorphic and all methods in \verb|Poset|
	that return a \verb|Poset| object in \verb|PosetIsoClass| instead return
	an instance of \verb|PosetIsoClass|.

	Construct an instance of \verb|PosetIsoClass| the same way as you would an
	instance of \verb|Poset| or given a poset \verb|P| use \verb|P.isoClass()|.

	@is_section@no_children@
	'''
	def __init__(this, *args, **kwargs):
		super().__init__()
		if '__hash__()' in this.cache: del this.cache['__hash__()']
	def __eq__(this, that):
		return isinstance(that,Poset) and this.is_isomorphic(that)
	@cached_method
	def __hash__(this):
		X=set()
		for r in range(len(this.ranks)-1):
			for p in this.ranks[r]:
				X.add((r,len([q for q in this.ranks[r+1] if this.zeta[p, q]!=0])))
		return hash(frozenset(X))
	def __str__(this):
		return "Isomorphism class of "+Poset.__str__(this)
	def __repr__(this):
		return "PosetIsoClass("+Poset.__repr__(this)[len('Poset('):]
@decorator.decorator
def return_iso(f, *args, **kwargs):
	r'''
	Used to define \verb|PosetIsoClass| this decorator causes a function returning
	\verb|ret| to return \verb|ret.isoClass()| when \verb|ret| is an instance of \verb|Poset|.

	@section@Utilities@
	'''
	ret = f(*args, **kwargs)
	if isinstance(ret, Poset):
		return ret.isoClass()
	return ret
for f in dir(PosetIsoClass):
	if f.startswith('_'): continue
	if f=='isoClass': continue
	attr = getattr(PosetIsoClass, f)
	if not callable(attr): continue
	setattr(PosetIsoClass, f, return_iso(attr))
del attr
##############
#End Poset Iso Class
##############

##############
#Genlatt class
##############
class Genlatt(Poset):
	r'''
	A class to encode a ``generator-enriched lattice'' which is a lattice $L$
	along with a set $G\subseteq L\wout\{\zerohat\}$ that generates $L$
	under the join operation.

	This class is mainly provided for the \verb|minorPoset| method.

	Constructor arguments are the same as for \verb|Poset| except that
	this constructor accepts two additional keyword only arguments:
		\begin{itemize}
			\item{\verb|G| - An iterable specifying the
				generating set, can either contain
				elements of the lattice or indices into
				the lattice. The join irreducibles are automatically
				added and may be omitted. If \verb|G| is not
				provided the generating set will consist of the
				join irreducibles.}
			\item{\verb|G_indices| - If \verb|True| then the provided
				argument \verb|G| should
				consist of indices otherwise \verb|G| should
				consist of elements.}
		\end{itemize}

	Note, a lattice $L$ enriched with a generating set $G$ is denoted
	as the pair $(L,G)$.
	See \cite{gustafson-23} for more on {generator-enriched lattices}\footnote{perhaps too much more}.

	@is_section@section_key@1@subclass@
	'''
	def __init__(this, *args, G=None, G_indices=False, **kwargs):
		r'''
		See \verb|Genlatt|.
		'''
		super().__init__(*args, **kwargs)
		covers = super().covers()
		if 'covers(False,)' in this.cache: del this.cache['covers(False,)']
		if 'covers(True,)' in this.cache: del this.cache['covers(True,)']
		this.hasseDiagram.P = this
		irrs = [k for (k,v) in this.dual().covers(G_indices).items() if len(v)==1]
		if G==None:
			this.G = irrs
		else:
			this.G = tuple(set([this.elements[i] for i in G]+irrs)) if G_indices else tuple(set(list(G)+irrs))
		this.G = tuple(sorted([g for g in this.G if g not in this.min()], key=lambda x:this.elements.index(x)))
	#overwrite L's covers function so hasse diagram does all edges
	@cached_method
	def covers(this, indices=False):
		r'''
		Returns a dictionary specifying the diagram edges of the genlatt (the name is a misnomer but is used for compatibility with \verb|HasseDiagram|).

		The edges of the diagram are pairs ($\ell,\ell\vee g)$ where
		$\ell$ is a lattice element and $g$ is a generator (and
		$\ell\ne\ell\vee g$. The return value is a dictionary in
		the same form as the return value of \verb|Poset.covers|,
		that is, keys are nonmaximal elements $\ell$ and values
		are lists of elements $k$ such that there is an edge from
		$\ell$ to $k$.
		'''
		edges={}
		Gens = [this.elements.index(g) for g in this.G] if indices else this.G

		L = this.relabel() if indices else Poset(this)
		L = L.complSubposet(L.max())
		for l in L:
			edges[l] = []
			for g in Gens:
				lg = this.join(l,g,indices)
				if lg == l or lg in edges[l]: continue
				edges[l].append(lg)
		return edges

	@cached_method
	def isRanked(this):
		return all(all(this.rank(v)==this.rank(k)+1 for v in V)for (k,V) in super().covers().items())

	def minor(this, H, z):
		r'''
		Given an iterable \verb|H| of generators and an element \verb|z| returns the \verb|Genlatt| with minimum \verb|z| and generating
		set \verb|H| and with the same order as \verb|this|.
		'''
		elements = set()
		for I in subsets(H):
			for l in itertools.accumulate(I, this.join, initial=z): pass
			elements.add(l)
		return Genlatt(this.subposet(elements),G=H)

	def Del(this, K):
		r'''
		Return the deletion set of the minor \verb|K|.

		The argument \verb|K| should be an instance of \verb|Genlatt|.

		The deletion set of a minor $(K,H)$ of $(L,G)$ is the set
		\[\text{Del}(K,H)=\{g\in G:g\join\zerohat_K\not\in H\cup\{\zerohat_K\}\}\]
		This is the minimal set of generators that must be deleted
		to form $(K,H)$ from $(L,G)$.
		'''
		z = K.min()[0]
		return [g for g in this.G if (not this.lesseq(g,z)) and (this.join(g,z) not in K.G)]

	def contract(this, g, weak=False, L=None):
		r'''
		Return the \verb|Genlatt| obtained by contracting the generator \verb|g|, if \verb|weak| is \verb|True| performs the weak contraction with respect to \verb|L| (default value for \verb|L| is \verb|this|).
		'''
		H = [this.join(g,h) for h in this.G]
		H = [h for h in H if h!=g]
		if weak and L!=None:
			D = [L.join(g,h) for h in L.Del(this)]
			if g in D: return None
			H = [h for h in H if h not in D]
		return this.minor(H,g)

	def delete(this, g):
		r'''
		Return the \verb|Genlatt| obtained by deleting the generator \verb|g|.
		'''
		H = [h for h in this.G if h!=g]
		return this.minor(H,this.min()[0])

	def _minors(this, minors, rels, i, weak, L):
		r'''
		Recursion backend to \verb|minors|.
		'''
		rels[i]=[]
		for g in this.G:
			for M in (this.delete(g),this.contract(g,weak,L)):
				if M==None: continue #weak contraction not defined
				if M in minors:
					Mi = minors.index(M)
				else:
					Mi = len(minors)
					minors.append(M)
				rels[i].append(Mi)
				M._minors(minors,rels,Mi,weak,L)
		return

	def minors(this, weak=False):
		r'''
		Returns a list of minors of the given \verb|Genlatt| instance
		and an incomplete dictionary of relations.

		The relations when transitively closed yield the relations
		for the minor poset.
		'''
		minors = [this]
		rels = {}
		this._minors(minors, rels, 0,weak,this)
		return minors, rels

	def __eq__(this,that):
		return isinstance(that,Genlatt) and Poset.__eq__(this,that) and set(this.G) == set(that.G)

	@cached_method
	def __hash__(this):
		return hash((super().__hash__(), frozenset(this.G)))
	def __str__(this):
		return super().__str__() + '\nG = '+str(this.G)
	def __repr__(this):
		return 'Genlatt'+super().__repr__()[len('Poset'):-1] + ', G='+repr(this.G)+')'
	def minorPoset(this, weak=False, **kwargs):
		r'''
		Returns the minor poset of the given \verb|Genlatt| instance.

		When generating a Hasse diagram with \verb|latex()| use
	the prefix \verb|L_| to control options for the node diagrams.
		'''
		minors, rels = this.minors(weak)
		if this.name=='': name = 'Minor poset of a generator-enriched lattice'
		else: name = 'Minor poset of '+this.name
		M = Poset(
			elements=minors,
			relations=rels,
			indices=True,
			name=name,
			)
		if not weak:
			M.cache['isRanked()']=True
			M.cache['isEulerian()']=True
			M.cache['isGorenstein()']=True

		#sort ranks correctly for nice plotting: use the cube sorting (revlex from 0<*<1)
		#and the weak minor poset join injection (0's are deletion setand 1's are zerohat ideal)
		def sort_key(K):
			D = this.Del(K)
			z = K.min()[0]
			C = [g for g in this.G if this.lesseq(g,z)]
			return [0 if g in D else 2 if g in C else 1 for g in this.G][::-1]
		M = M.sort(sort_key)
		M = M.dual().adjoin_zerohat() #wrecks hasseDiagram

		nodeLabel = lambda this,i : '$\\emptyset$' if i in this.P.min(True) else nodeLabel(this,i)
		kwargs.update({
			'draw_min' : False,
			'prefix' : 'L',
			'width' : 8,
			'height' : 10,
			'L_width' : 0.8,
			'L_height' : 1,
			'L_height' : 2,
			'L_width' : 1,
			'L_nodescale' : 0.5,
			})
		M.hasseDiagram = SubposetsHasseDiagram(M,this,**kwargs)
		return M

	def cartesianProduct(this,that):
		r'''
		Computes the Cartesian product.

		The Cartesian product of two generator-enriched lattices $(L,G)$ and $(K,H)$ is
		\[(L\times K,(G\times\{\zerohat_K\})\cup(\{\zerohat_L\times H)).\]
		'''
		zthis = this.min()[0]
		zthat = that.min()[0]
		return Genlatt(
			Poset.cartesianProduct(this,that),
			G=tuple((g,zthat) for g in this.G)+tuple((zthis,h) for h in that.G)
			)

	def diamondProduct(this,that):
		r'''
		Computes the diamond product.

		The diamond product of two generator-enriched lattices $(L,G)$ and $(K,H)$ is
		\[(L\diamond K, G\times H)\]
		where
		\[L\diamond K = ((L\wout\{\zerohat\})\times(K\wout\{\zerohat\}))\cup\{\zerohat\}.\]
		'''
		return Genlatt(Poset.diamondProduct(this,that), G=itertools.product(this.G,that.G))

	def pyr(this):
		r'''
		Computes the pyramid over a generator-enriched lattice.

		The pyramid over a generator-enriched lattice $(L,G)$ is the generator-enriched
		lattice $(L,G)\times(B_1,\irr(B_1))$.
		'''
		return this.cartesianProduct(Genlatt(Chain_1()))

	def prism(this):
		r'''
		Computes the prism over a generator-enriched lattice.

		The prism over a generated-enriched lattice $(L,G)$ is the generator-enriched lattice
		$(L,G)\diamond(B_2,\irr(B_2))$.
		'''
		return this.diamondProduct(Genlatt(Cube_1()))

	def adjoin_onehat(this):
		r'''
		Returns the generator-enriched lattice obtained by adjoining a new maximum.
		'''
		ret = Poset.adjoin_onehat(this)
		return Genlatt(ret,G=this.G+tuple(ret.max()))

	def adjoin_zerohat(this):
		r'''
		Returns the generator-enriched lattice obtained by adjoining a new minimum.
		'''
		return Genlatt(Poset.adjoin_zerohat(this),G=this.G+tuple(this.min()))

##############
#End Genlatt class
##############
