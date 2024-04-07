##########################################
#TODO
##########################################
#Bug: latex and show for rank 0 posets
#
#standardize incMat convention (sign and diagonal)
#
#standardize choice of camelcase versus underscores
#
#I had in adjoin_onehat() this.copy_from(...)
#and it made it so that if you save a poset P
#and then call adjoin_onehat() (keeping P the same value)
#then you can't iterate over P (for p in P: throws an error)
#What the heck is that?
	#kind of moot cause like I said it's fixed
	#but I want to know cause that's so strange
#
#Find and add non-eulerian dehn sommerville posets?
#
#add a nodeName function for built ins (default just does index)
#
#make latex() dump its arguments in the first line
#
#make a HasseDiagram subclass for MinorPoset
#	needs to use the HasseDiagram on the lattice to find node placement for the
#	minors' diagrams.
#
#	Maybe we should add an option to tkinter to draw to a canvas and also make
#	HasseDiagram have an edge set it computes that we can set to be different
#	for genlatts
#
#Set HasseDiagram defaults for Bqn
#
#Make HasseDiagram subclass for Distributive lattices
#
#Set HasseDiagram defaults for lattices of flats, should be same as BooleanAlgebra
#for polymatroids and graph is using partitions
#
#We should have different functions for text for tikz and tkinter with the defaults the
#same function
#
#Uniform matroids
#
#Uniform q-matroids
#
#Antichain example (fix latex and show)
##########################################
'''
A module for creating and manipulating partially ordered sets.

This module provides a class Poset that encodes a finite poset. The Poset class provides:

	1. Operations between posets such as disjoint  unions, Cartesian products
	star products and diamond products.

	2. Methods to calculate data about a given poset such as flag f and h vectors,
	the cd-index and mobius function values.

	3. Examples of posets such as Chains, Boolean algebras, face lattices of cubes,
	type A Bruhat orders, uncrossing posets.

Definitions of a few terms in this documentation:

	Poset - A set equipped with a relation <= that is reflexive, transitive and antisymmetric.

	Chain - A totally ordered set. A chain of a poset is a totally ordered subset (under the same ordering)

	zeta matrix - The matrix Z satisfying Z[i][j] is 1 when i<=j and 0 otherwise.

	mobius function - The function mu on PxP whose values mu(i,j) give the entries of the inverse of the zeta matrix.

	rank - A poset is ranked if the length of all inclusionwise maximal chains is the same.
		The rank of the poset is the length of the longest chain. The rank of an
		element is the length of the longest chain ending at said element.

	length - The length of an element is the length of the longest chain ending at said
		element. If the poset is ranked this is the same as rank of the element.

	minimum - 0 is a minimum of P if 0 <= p for all p in P.

	minimal - x is minimal in P if p <= x only holds for p = x.

	maximum - 1 is a maximum of P if p <= 1 for all p in P.

	maximal - x is maximal in P if x <= p only holds for p = x.

	order complex - The order complex of a poset is the (abstract) simplicial complex whose
		faces are all the chains of P.
'''
import itertools
import copy
import decorator
import collections

import time
class Timer:
	def __init__(this, running=True):
		this.elapsed_time = 0
		this.running = False
		if running: this.start()

	def start(this):
		this.running = True
		this.start_time = time.time()

	def stop(this):
		this.stop_time = time.time()
		if this.running: this.elapsed_time += this.stop_time - this.start_time
		this.running = False

	def pause(this):
		the_time = time.time()
		this.elapsed_time += the_time - this.start_time
		this.running = False

	def reset(this, *args):
		this.stop()
		this.elapsed_time = 0
		this.start()

	def __str__(this):
		return str(this.elapsed_time)

@decorator.decorator
def cached_method(f, *args, **kwargs):
	'''
	This is an attribute for class functions that will cache the return result in a cache property on the object.

	When the attribute is placed on a class function e.g.

		@cached_method
		def f(this,...):

	calls to the function will first check the dictionary this.cache for the function call
	and if it is stored instead of calling the function the cache value is returned.
	If the value is not present the function is called then the return value is cached
	and then that value is returned.

	The key in this.cache for a function call f(this,'a',7,[1,2,3],b= ['1'],a = 3) is
		"f('a', 7, [1,2,3], a=3, b=['1'])"
	The key for a function call f(this) is 'f()'.

	This attribute should not be placed on a function whose return value is not solely
	dependent on its arguments (including the argument this).
	'''
	key = f.__name__ + str(args[1:])[:-1] + ((', ' + ', '.join([str(k)+'='+repr(v) for (k,v) in sorted(kwargs.items(),key = lambda x:x[1])])) if len(kwargs)>0 else '')+')'
	try:
		if key in args[0].cache:
			return args[0].cache[key]
	except:
		pass
	ret = f(*args, **kwargs)
	try:
		args[0].cache[key] = ret
	except:
		pass
	return ret

def log_func(f):
	def g(*args, **kwargs):
		print(f.__name__,*['\t'+str(a) for a in args],*['\t'+k+'='+str(v) for (k,v) in kwargs.items()],sep='\n')
		ret = f(*args,**kwargs)
		print('\treturn value:',ret)
		return ret
	return g

class Poset:
	'''
	A class representing a finite partially ordered set.

	Constructor arguments:
		incMat - A matrix whose i,j value is 1 if the ith element is strictly less
			than the jth element.

		elements - A list specifying the elements of the poset.

			The default value is [0,...,len(incMat)-1]

		ranks - A list of lists. The ith list is a list of indices of element of
			length i.

		relations - Either a list of pairs (x,y) such that x<y or a dictionary
			whose values are lists of elements greater than the associated key.
			This is used to construct incMat if it is not provided.

		less - A function that given two elements p,q returns True when p < q.
			This is used to construct incMat if neither incMat nor relations
			are provided.

		indices - A boolean indicating indices instead of elements are used in relations
			and less. The default is False.

		name - An optional identifier if not provided no name attribute is set.

		hasse_class - An instance of hasse_class is constructed with arguments being
			this poset plus all keyword arguments passed to Poset, i.e.:
				this.hasseDiagram = hasse_class(this, **kwargs)
			If you subclass HasseDiagram to change default drawing behavior pass
			your subclass when constructing a poset.

			The default value is HasseDiagram.

		trans_close - If True the transitive closure of incMat is computed, this should
			be False only if the provided matrix satisfies
				incMat[i][j] ==
					1 when i<j
					-1 when i>j
					0 otherwise.

		The keyword arguments are passed to HasseDiagram (or hasse_class if specified).

	To construct a poset you must pass at least either incMat or less.

	In the constructor the transitive closure of incMat is computed (unless trans_close=False is set)
	so it is only essential that incMat[i][j] == 1 when i is covered by j. The diagonal
	entries incMat[i][i] should always be zero.

	If elements is not provided it will be set to [0,...,len(incMat)]

	ranks is inessential, if not provided it will be computed.

	Function calls to several of the more costly computations are cached. Generally
	functions in this class do not change the poset but instead return a new poset.
	Posets may be considered immutable (this is not enforced in any way), or if you alter
	a poset you should clear the cache via:
		this.cache = {}
	'''
	def __init__(this, incMat=None, elements=None, ranks=None, less=None, name='', hasse_class=None, trans_close=True, relations=None, indices=False, **kwargs):
		'''
		See Poset
		'''
		#this.__slots__ = ('cache','hasseDiagram','incMat','elements','ranks','name')
		if elements !=None: elements = list(elements) #can take any iterable but need indexing
		#####
		#make incMat
		#####
		if type(relations) == list:
			if elements == None:
				this.elements = list(set(tuple(itertools.chain(*relations))))
			else:
				this.elements = elements

			this.incMat = [[0]*len(this.elements) for e in this.elements]
			if indices:
				for rel in relations:
					this.incMat[rel[0]][rel[1]] = 1
					this.incMat[rel[1]][rel[0]] = -1
			else:
				elems_to_indices = {}
				for i in range(len(this.elements)): elems_to_indices[this.elements[i]] = i

				for rel in relations:
					i = elems_to_indices[rel[0]]
					j = elems_to_indices[rel[1]]
					this.incMat[i][j] = 1
					this.incMat[j][i] = -1

		elif type(relations)==dict:
			if elements == None:
				this.elements = list(set(itertools.chain(*relations.values())).union(relations.keys()))
			else:
				this.elements = elements
			elems_to_indices = {}
			for i in range(len(this.elements)): elems_to_indices[this.elements[i]] = i

			this.incMat = [[0]*len(this.elements) for e in this.elements]
			if indices:
				for i in relations:
					for j in relations[i]:
						this.incMat[i][j] = 1
						this.incMat[j][i] = -1
			else:
				elems_to_indices = {}
				for i in range(len(this.elements)): elems_to_indices[this.elements[i]] = i

				for p in relations:
					i = elems_to_indices[p]

					for q in relations[p]:
						j = elems_to_indices[q]
						this.incMat[i][j] = 1
						this.incMat[j][i] = -1

		elif less != None:
			this.incMat = [[0]*len(elements) for e in elements]
			Less = less if indices else (lambda x,y: less(elements[x],elements[y]))
			for i in range(len(elements)):
				for j in range(i+1,len(elements)):
					if Less(i,j):
						this.incMat[i][j] = 1
						this.incMat[j][i] = -1
					elif Less(j,i):
						this.incMat[i][j] = -1
						this.incMat[j][i] = 1
		elif incMat != None:
			this.incMat = incMat

		else: #no data provided poset is (possibly empty) antichain
			if elements == None:
				this.incMat = []
			else:
				this.incMat = [[0]*len(elements) for e in elements]

		if len(this.incMat)>0 and trans_close: transClose(this.incMat)
		#####
		#####
		if not hasattr(this, 'elements'):
			this.elements = list(range(len(this.incMat))) if elements == None else elements
		this.ranks = ranks if ranks!=None else Poset.make_ranks(this.incMat)
		this.cache = {}
		this.name = name
		if hasse_class == None: hasse_class = HasseDiagram
		this.hasseDiagram = hasse_class(this, **kwargs)

	def make_incMat(relations=None, less=None, incMat=None, trans_close=True, elements=None, indices=False):
		'''
		Returns a valid incidence matrix given any of a list of relations, a less than function or a (possibly incomplete) incidence matrix.
		'''
		#Efff... we need to put this all back in the constructor (:/ whatever the if logic isn't really complicated)
		#because we need to set elements via relations
		#then when setting elements normal way check if it is set yet


	def __str__(this):
		'''
		Returns a nicely formatted string listing the zeta matrix, the ranks list and the elements of the poset.
		'''
		ret = ['zeta = ['] + [' '.join([(' ' if x>=0 else '')+str(x) for x in z]) for z in this.zeta()]+[']']
		if hasattr(this, 'name'):
			ret = [this.name]+ret
		ret.append('ranks = '+str(this.ranks))
		ret.append('elements = '+str(this.elements))
		return '\n'.join(ret)

	def __repr__(this):
		'''
		Gives a string that can be evaluated to recreate the poset.

		To eval the returned string Poset must be in the namespace and repr(this.elements)
		must return a suitable string for evaluation.
		'''
		return 'Poset(incMat='+repr(this.incMat)+', elements='+repr(this.elements)+', ranks='+repr(this.ranks)+')'

	def __eq__(this,that):
		'''
		Checks whether the two posets have the same elements in the same order and the same relations.
		'''
		return isinstance(that, Poset) and set(this.elements) == set(that.elements) and all([this.less(p,q) == that.less(p,q) for p in this for q in this])

	def __iter__(this):
		'''
		Wrapper for this.elements__iter__.
		'''
		return this.elements.__iter__()

	def __getitem__(this, i):
		'''
		Wrapper for this.elements.__getitem__.
		'''
		return this.elements.__getitem__(i)
	def __contains__(this, p):
		'''
		Wrapper for this.elements.__contains__.
		'''
		return this.elements.__contains__(p)

	def __len__(this):
		'''
		Wrapper for this.elements.__len__.
		'''
		return this.elements.__len__()

	##############
	#Operations
	##############
	def adjoin_zerohat(this, label=None):
		'''
		Returns a new poset with a new minimum adjoined.

		By default the label is 0 and if 0 is already an element the default label is
		the first positive integer that is not an element.
		'''
		incMat = [[0]+[1 for p in this.elements]]+[[-1]+this.incMat[i] for i in range(len(this.elements))]
		if label==None:
			label=0
			while label in this.elements:
				label += 1
		assert(not label in this.elements)
		ranks = [[0]]+[[r+1 for r in row] for row in this.ranks]
		elements=[label]+this.elements
		P = Poset(incMat = incMat, elements = elements, ranks = ranks, trans_close = False)
		if 'isRanked()' in this.cache:
			P.cache['isRanked()'] = this.cache['isRanked()']
		if 'isGorenstein()' in this.cache and this.cache['isGorenstein()']:
			P.cache['isGorenstein()'] = False
		if 'isEulerian()' in this.cache and this.cache['isEulerian()']:
			P.cache['isEulerian()'] = False
		return P

	def adjoin_onehat(this, label=None):
		'''
		Returns a new poset with a new maximum adjoined.

		The label default is the same as Poset.adjoin_zerohat()
		'''
		return this.dual().adjoin_zerohat().dual()

	def identify(this, X, indices=False):
		'''
		Returns a new poset after making identifications indicated by X.

		The new relation is p <= q when there exists any representatives p' and q' such that p' <= q'. The result may not
		truly be a poset as it may not satisfy the antisymmetry axiom (p < q < p implies p = q).

		X should either be a dictionary where keys are the representatives and the value is a list of elements
		to identify with the key, or a list of lists where the first element of each list is the representative.
		Trivial equivalence classes need not be specified.
		'''
		if type(X)==dict:
			X = [[k]+list(v) for (k,v) in X.items()]
		if indices:
			X = [[this.elements[X[i][j]] for j in range(len(X[i]))] for i in range(len(X))]
		X = [[x[0]]+[y for y in x[1:] if y!=x[0]] for x in X]
		removed = list(itertools.chain(*[X[i][1:] for i in range(len(X))]))
#		removed = sum([X[i][1:] for i in range(len(X))], start=[])
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
		'''
		Returns the dual poset which has the same elements and relation p <= q when q <= p in the original poset.
		'''
		P = Poset()
		P.elements = this.elements
		P.ranks = this.ranks[::-1]
		P.incMat = [[this.incMat[j][i] for j in range(len(this.elements))] for i in range(len(this.elements))]
		P.hasseDiagram = copy.copy(this.hasseDiagram)
		P.hasseDiagram.P = P
		if 'isRanked()' in this.cache:
			P.cache['isRanked()'] = this.cache['isRanked()']
		if 'isEulerian()' in this.cache:
			P.cache['isEulerian()'] = this.cache['isEulerian()']
		if 'isGorenstein()' in this.cache:
			P.cache['isGorenstein()'] = this.cache['isGorenstein()']
		return P

	def element_union(E, F):
		'''
		Computes the disjoint union of E and F.

		If E and F have empty intersection return value is E+F
		otherwise return value is (Ex{0})+({0}xF). This is used by operation methods
		such as Poset.union and Poset.starProduct.
		'''
		if any([e in F for e in E]+[f in E for f in F]):
			return [(e,0) for e in E] + [(0,f) for f in F]
		else:
			return E+F

	def union(this, that):
		'''
		Computes the disjoint union of two posets.
		'''
		elements = Poset.element_union(this.elements, that.elements)
		incMat = [z+[0]*len(that.elements) for z in this.incMat] + [[0]*len(this.elements)+z for z in that.incMat]

		that_ranks = [[r+len(this.elements) for r in rk] for rk in that.ranks]
		if len(this.ranks)>len(that_ranks):
			small_ranks = that_ranks
			big_ranks = this.ranks
		else:
			small_ranks = this.ranks
			big_ranks = that_ranks
		small_ranks = small_ranks + [[] for i in range(len(big_ranks) - len(small_ranks))]

		ranks = [small_ranks[i]+big_ranks[i] for i in range(len(small_ranks))]

		return Poset(incMat, elements, ranks)

	def bddUnion(this, that):
		'''
		Computes the disjoint union of two posets with maximum and minimums adjoined.
		'''
		this_proper = this.complSubposet(this.max(True)+this.min(True), True)
		that_proper = that.complSubposet(that.max(True)+that.min(True), True)

		return this_proper.union(that_proper).adjoin_zerohat().adjoin_onehat()

	def starProduct(this, that):
		'''
		Computes the star product of two posets.

		This is the union of this with the maximum removed and that with the minimum
		removed and all relations p < q for p in this and q in that.
		'''
		that_part = that.complSubposet(that.min())
		this_part = this.complSubposet(this.max())
		elements = Poset.element_union(this_part.elements, that_part.elements)
		incMat = [z+[1]*len(that_part.elements) for z in this_part.incMat]+[[-1]*len(this_part.elements)+z for z in that_part.incMat]
		ranks = this_part.ranks + [[r+len(this_part.elements) for r in rk ] for rk in that_part.ranks]

		return Poset(incMat, elements, ranks)

	def cartesianProduct(this, that):
		'''
		Computes the cartesian product.
		'''
		elements = [(p,q) for p in this.elements for q in that.elements]
		ranks = [[] for i in range(len(this.ranks)+len(that.ranks)-1)]
		for i in range(len(this.elements)):
			for j in range(len(that.elements)):
				ranks[rankOf(this.ranks, i) + rankOf(that.ranks, j)].append(i*len(that.elements) + j)

		def less(x,y):
			return x!=y and this.lesseq(x[0],y[0]) and that.lesseq(x[1],y[1])

		return Poset(elements=elements, ranks=ranks, less=less)

	def diamondProduct(this, that):
		'''
		Computes the diamond product which is the Cartesian product of the two posets with their minimums removed and then adjoined with a new minimum.
		'''
		this_proper = this.complSubposet(this.min(True), True)
		that_proper = that.complSubposet(that.min(True), True)
		return this_proper.cartesianProduct(that_proper).adjoin_zerohat()

	def pyr(this):
		'''
		Computes the pyramid of a poset, that is, the Cartesian product with a length 1 chain.
		'''
		return this.cartesianProduct(Chain(1))

	def prism(this):
		'''
		Computes the prism of a poset, that is, the diamond product with Cube(1).
		'''
		return this.diamondProduct(Cube(1))
	##############
	#End Operations
	##############
	##############
	#Queries
	##############
	@cached_method
	def isRanked(this):
		'''
		Checks whether the poset is ranked.
		'''
		for i in range(len(this)):
			row = this.incMat[i]
			rk = this.rank(i,indices=True)
			found = True
			for j in range(len(this)):
				if row[j]==1:
					if this.rank(j,indices=True)==rk+1:
						found = True
						break
					else:
						found = False
			if not found:
				return False
		return True

	@cached_method
	def isEulerian(this):
		'''
		Checks whether the given poset is Eulerian (every nontrivial interval has an equal number of odd and even rank elements).
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
		'''
		Checks if the poset is Gorenstein*.

		A poset is Gorenstein* if the proper part of all intervals with more than
		two elements have sphere homology. In other words, thsi function checks that
		this.interval(p,q).properPart().bettiNumbers() is either [2] or [1,0,...,0,1] for all p<=q such that
		this.rank(q) - this.rank(p) >= 2.
		'''
		if len(this.ranks)==1: return len(this.elements)==1
		if len(this.ranks)==2: return len(this.elements)==2
		def check_homol(P):
			if len(P.elements)==2: return True
			b = P.properPart().bettiNumbers()
			return b==[2] or (b[0] == 1 and b[-1] == 1 and all([x == 0 for x in b[1:-1]]))
		return all([check_homol(this.interval(x,y)) for x in this.elements for y in this.elements if this.less(x,y)])

	@cached_method
	def covers(this):
		if this.isRanked():
			ret = {}
			non_max = [i for i in range(len(this)) if i not in this.max(indices=True)]
			for i in non_max:
				p = this[i]
				ret[p] = []
				for j in this.ranks[this.rank(p)+1]:
					q = this[j]
					if this.less(i,j,indices=True):
						ret[p].append(q)
			return ret
		else:
			return {p : this.filter((p,),indices=False,strict=True).min(indices=False) for p in this if p not in this.max()}
	##############
	#End Queries
	##############
	##############
	#Subposet selection
	##############
	@cached_method
	def min(this, indices=False):
		'''
		Returns a list of the minimal elements of the poset.
		'''
		return this.dual().max(indices)

	@cached_method
	def max(this, indices=False):
		'''
		Returns a list of the maximal elements of the poset.
		'''
		ret_indices = [i for i in range(len(this.incMat)) if 1 not in this.incMat[i]]
		return ret_indices if indices else [this.elements[i] for i in ret_indices]

	def complSubposet(this, S, indices=False):
		'''
		Returns the subposet of elements not contained in S.
		'''
		if not indices:
			S = [this.elements.index(s) for s in S]
		return this.subposet([i for i in range(len(this.incMat)) if i not in S],True)

	def subposet(this, S, indices=False):
		'''
		Returns the subposet of elements in S.
		'''
		if not indices:
			S = [this.elements.index(s) for s in S]
		elements = [this.elements[s] for s in S]
		incMat = [[this.incMat[s][r] for r in S] for s in S]
		return Poset(incMat, elements)
#		ranks = [[S.index(r) for r in rk if r in S] for rk in this.ranks]
#		ranks = [rk for rk in ranks if len(rk)>0]
#		return Poset(incMat,elements,ranks)

	def interval(this, i, j, indices=False):
		'''
		Returns the closed interval [i,j].
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
		element_indices = [k for k in range(len(this.elements)) if k in (i,j) or \
		(this.incMat[i][k] == 1 and 1 == this.incMat[k][j]) ]

		return this.subposet(element_indices, indices=True)

	def filter(this, x, indices=False, strict=False):
		'''
		Returns the subposet of elements greater than or equal to any element of x.

		If strict is True then x is not included in the returned poset and
		if it is False x is included.
		'''
		if indices: x = [this[i] for i in x]

		comp = this.less if strict else this.lesseq

		ret = this.subposet([p for p in this if any([comp(y,p) for y in x])])
		if indices: ret.elements = [this.elements.index(p) for p in ret]
		return ret

	def ideal(this, x, indices=False, strict=False):
		'''
		Returns the subposet of elements less than or equal to any element of x.

		Wrapper for this.dual().filter.
		'''
		return this.dual().filter(x,indices,strict)

	@cached_method
	def properPart(this):
		'''
		Returns the subposet of all elements that are neither maximal nor minimal.
		'''
		P = this
		if len(P.min()) == 1:
			P = P.complSubposet(P.min())
		if len(P.max()) == 1:
			P = P.complSubposet(P.max())
		return P

	def rankSelection(this, S):
		'''
		Returns the subposet of elements whose rank is contained in S.

		Does not automatically include the minimum or maximum.
		'''
		ret = this.subposet([x for x in itertools.chain(*[this.ranks[s] for s in S])], indices=True)
		if 'isRanked()' in this.cache:
			ret.cache['isRanked()'] = this.cache['isRanked()']
		return ret
	##############
	#End Subposet selection
	##############
	##############
	#Internal computations
	##############
	def less(this, i, j, indices=False):
		'''
		Returns whether i is strictly less than j.
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)

		return this.incMat[i][j]==1

	def lesseq(this, i, j, indices=False):
		'''
		Returns whether i is less than or equal to j.
		'''
		return i==j or this.less(i,j, indices)

	def isAntichain(this, A, indices=False):
		'''
		Returns whether the given set is an antichain.
		'''
		for i in range(len(A)):
			for j in range(i+1,len(A)):
				if this.lesseq(A[i],A[j],indices) or this.lesseq(A[j],A[i],indices): return False
		return True

	@cached_method
	def join(this, i, j, indices=False):
		'''
		Computes the join of i and j, if it does not exist returns None.
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
			ret = join(i, j, this.incMat)
			if ret == None: return None
			return this.elements[ret]
		return this.join(i, j, this.incMat)

	def mobius(this, i=None, j=None, indices=False):
		'''
		Computes the value of the Mobius function from i to j.

		If i or j is not provided computes the mobius from the minimum to the maximum
		and throws an exception of type ValueError f there is no minimum or maximum.
		'''
		if i==None or j==None:
			bottom = this.min()
			top = this.max()
			if len(bottom)!=1 or len(top)!=1:
				raise ValueError("To compute the Mobius value for the whole poset the poset must be bounded.")
			i = this.elements.index(bottom[0])
			j = this.elements.index(top[0])
			indices = True
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)

		if i == j: return 1
		if not this.less(i,j,True) and not this.less(j,i,True): return 0

		@cached_method
		def calc_mobius(this, i,j):
			if i == j: return 1
			ret = 1
			for k in range(len(this)):
				if this.incMat[i][k]==1 and this.incMat[k][j]==1:
					ret += calc_mobius(this, i,k)
			return -ret

		return calc_mobius(this, i,j)

	@cached_method
	def rank(this, i, indices=False):
		'''
		Returns the length of i (the length of the longest chain ending at i).
		'''
		if not indices:
			i = this.elements.index(i)

		return rankOf(this.ranks, i)
	##############
	#End Internal computations
	##############
	##############
	#Invariants
	##############
	@cached_method
	def flagVectors(this):
		'''
		Returns the table of flag f and h vectors as a list with elements [S, f_S, h_S].
		'''
		return makeFlagVectorsTable(this.incMat, this.ranks)

	@cached_method
	def abIndex(this):
		return cdIndex(this.incMat, this.ranks, convert_to_cd=False)

	@cached_method
	def cdIndex(this):
		'''
		Returns the cd-index.

		The cd-index is encoded as a list where each item is a list [c,w] where w is a cd-monomial
		as a string and c is the coefficient. If the poset does not have a cd-index then the ab-index
		is returned.

		You can convert this to a string suitable for latex with posets.cdIndexLatex.

		For more info on the cd-index see https://arxiv.org/abs/1901.04939
		'''
		return cdIndex(this.incMat, this.ranks)

	@cached_method
	def cdIndex_IA(this, v=None):
		'''
		Returns the cd-index as calculated via Karu's incidence algebra formula.

		See Proposition 1.2 in https://doi.org/10.1112/S0010437X06001928

		The argument v should be a vector indexed by the poset and by default is the all 1's vector. This is
		the vector the incidence algebra functions are applied to.
		'''
		return cdIndex_incidence_algebra(this.incMat, this.ranks, np.array(v) if v!=None else v)


	@cached_method
	def cd_coeff_mat(this, u):
		'''
		Returns the matrix for the incidence algebra element giving the u coefficient of the cd-index via Karu's formula.
		'''
		return cdIndex_coefficient_mat(this.incMat, this.ranks, u)

	def cd_op(this, u, v=None):
		'''
		Applies the incidence algebra operation corresponding to the cd-monomial u to the vector v.
		'''
		X = this.cd_coeff_mat(u)
		if type(v)==type(None): v = np.array([[1] for p in this])
		else: v = np.array(v)
		X = np.matmul(X, v)
		return {this[i] : X[i][0] for i in range(len(this)) if X[i]!=0}

	def zeta(this):
		'''
		Returns the zeta matrix, the matrix whose i,j entry is 1 if elements[i] <= elements[j] and 0 otherwise.
		'''
		return [[1 if i==j or this.incMat[i][j]==1 else 0 for j in range(len(this.incMat))] for i in range(len(this.incMat))]

	@cached_method
	def bettiNumbers(this):
		'''
		Computes the Betti numbers of the poset, that is, the ranks of the homology of the order complex (the simplicial complex of all chains).
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
	def buildIsomorphism(this, that):
		d1 = {i: this.rank(i,indices=True) for i in range(len(this))}
		d2 = {i: that.rank(i,indices=True) for i in range(len(that))}
		if collections.Counter(d1.values()) != collections.Counter(d2.values()): return None
		def get_lengths(l):
			return tuple([len(x) for x in l])
		def set_comp_rks(P,d):
			for i in range(len(P)):
				d[i] = (d[i], get_lengths(P.filter([i],indices=True).ranks), get_lengths(P.ideal([i],indices=True).ranks))
			return d
		if 'set_comp_rks()' in this.cache:
			d1 = this.cache['set_comp_rks()']
		else:
			set_comp_rks(this, d1)
			this.cache['set_comp_rks()'] = d1
		if 'set_comp_rks()' in that.cache:
			d2 = that.cache['set_comp_rks()']
		else:
			set_comp_rks(that, d2)
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
			if i==len(P): return map
			if dP[i] not in dQinv: return None
			cands = dQinv[dP[i]] #candidates for image of P[i]
			for j in cands:
				if j in [m[1] for m in map]: continue
				#check if i->j is ordering-preserving
				if all(P.incMat[m[0]][i] == Q.incMat[m[1]][j] for m in map):
					new_map = iso(P,Q,map+((i,j),),dP,dQ,dPinv,dQinv,i+1)
					if new_map != None: return new_map
			return None
		ret = iso(this, that, tuple(), d1, d2, d1Inv, d2Inv, 0)
		if ret != None:
			ret = {r[0] : r[1] for r in ret}
		return ret

	def is_isomorphic(this, that):
		return this.buildIsomorphism(that) != None
	##############
	#End Maps
	##############
	##############
	#Misc
	##############
	@cached_method
	def _hash_tuple(this):
		d = {i: this.rank(i,indices=True) for i in range(len(this))}
		def get_lengths(l):
			return tuple([len(x) for x in l])
		ret = set()
		for i in range(len(this)):
			ret.add( (d[i], get_lengths(this.filter([i],indices=True).ranks), get_lengths(this.ideal([i],indices=True).ranks)) )
		return frozenset(ret)

	@cached_method
	def __hash__(this):
		return hash(tuple(tuple(term) for term in this.abIndex()))

	def copy(this):
		'''
		Returns a shallow copy of the poset.

		Making a shallow copy via the copy module i.e. Q = copy.copy(P)
		doesn't update the self reference in Q.hasseDiagram (in this example
		Q.hasseDiagram.P is P). This doesn't matter if you treat posets as immutable,
		but otherwise could cause issues when displaying or generating hasse diagrams.
		'''
		P = copy.copy(this)
		P.hasseDiagram.P = P
		return P

	def latex(this, **kwargs):
		r'''
		Returns a string of tikz code to draw the Hasse diagram of the poset for use in a latex document.

		This is a wrapper for HasseDiagram.latex.

		For a full list of keyword arguments see HasseDiagram. The most common arguments are:

			height - The height in tikz units of the diagram.

				The default value is 30.

			width - The width in tikz units of the diagram.

				The default value is 18.

			labels - If False elements are represented by filled circles.
				If True by default elements are labeled by the result
				of casting the poset element to a string.

				The default value is True.

			ptsize - When labels is False this is the size of the circles used
				to represent elements. This has no effect if labels is True.

				The default value is '2pt'.

			nodescale - Each node is wrapped in '\\scalebox{'+nodescale+'}'.

				The default value is '1'.

			standalone - When True a preamble is added to the beginning and
				'\\end{document}' is added to the end so that the returned string
				is a full latex document that can be compiled. Compiling requires
				the latex packages tikz (pgf) and preview. The resulting figure can be
				incorporated into another latex document with \includegraphics.

				When False only the code for the figure is returned; the return value
				begins with \begin{tikzpicture} and ends with \end{tikzpicture}.

				The default is False.

			nodeLabel - A function that takes the HasseDiagram object and an index
				and returns the label for the indicated element as a string.
				For example, the default implementation HasseDiagram.nodeLabel
				returns the element cast to a string and is defined as below:

					def nodeLabel(H, i):
						return str(H.P[i])

				note H.P is this poset.
		'''
		return this.hasseDiagram.latex(**kwargs)

	def show(this, **kwargs):
		'''
		Opens a window displaying the Hasse diagram of the poset.

		This is a wrapper for HasseDiagram.tkinter.

		For a full list of keyword arguments see HasseDiagram. The most common arguments are:

			height - The height of the diagram.

				The default value is 30.

			width - The width of the diagram.

				The default width is 18.

			labels - If False elements are represented as filled circles.
				If True by default elements are labeled by the result of
				casting the poset element to a string.

				The default value is True.

			ptsize - When labels is False controls the size of the circles
				representing elements. This can be an integer or a string,
				if the value is a string the last two characters are ignored.

				The default value is '2pt'.

			scale - Scale of the diagram.

				The default value is 1.

			padding - A border of this width is added around all sides of the diagram.

				The default value is 1.

			nodeLabel - A function that takes the HasseDiagram object and an index
				and returns the label for the indicated element as a string.
				For example, the default implementation HasseDiagram.nodeLabel
				returns the element cast to a string and is defined as below:

					def nodeLabel(H, i):
						return str(H.P[i])

				note H.P is this poset.
		'''
		return this.hasseDiagram.tkinter(**kwargs)

	def chains(this, indices=False):
		'''
		Returns a list of all nonempty chains of the poset (subsets p_1<...<p_r).
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
		'''
		Returns the poset of all chains ordered by inclusion.
		'''

		elements = this.chains(indices)

		ranks = [[] for i in range(max([len(c) for c in elements])+1)]
		for i in range(len(elements)):
			ranks[len(elements[i])].append(i)

		incMat = [[0]*len(elements) for e in elements]
		for i in range(len(elements)):
			for j in range(i+1, len(elements)):
				if all([x in elements[j] for x in elements[i]]):
					incMat[i][j] = 1
					incMat[j][i] = -1
				elif all([x in elements[i] for x in elements[j]]):
					incMat[j][i] = 1
					incMat[i][j] = -1

		args = {'elements': elements, 'ranks': ranks, 'incMat': incMat, 'trans_close': False}
		if hasattr(this,'name'): args['name'] = 'Order complex of '+this.name
		P = Poset(**args)

		return P

	def relations(this, indices=False):
		'''
		Returns a list of all pairs (e,f) where e <= f.
		'''
		if indices:
			return [(i,j) for j in range(len(this.elements)) for i in range(len(this.elements)) if this.incMat[i][j] == 1]

		return [(this.elements[i],this.elements[j]) for j in range(len(this.elements)) for i in range(len(this.elements)) if this.incMat[i][j]==1]

	def reorder(this, perm, indices=False):
		'''
		Returns a new Poset object (representing the same poset) with the elements reordered.

		perm should be a list of elements if indices is False or a list of indices if True. The returned poset has elements in the given order, i.e. perm[i] is the ith element.
		'''
		if not indices:
			perm = [this.elements.index(p) for p in perm]

		elements = [this.elements[i] for i in perm]
		incMat = [[this.incMat[i][j] for j in perm] for i in perm]
		ranks = [sorted([perm.index(i) for i in rk]) for rk in this.ranks]

		P = Poset(elements = elements, incMat = incMat, ranks = ranks, trans_close = False)
		P.hasseDiagram = this.hasseDiagram
		P.hasseDiagram.P = P

		return P


	def sort(this, key = None, indices=False):
		'''
		Returns a new Poset object (representing the same poset) with the elements sorted.
		'''
		perm = sorted(this.elements, key = key)
		return this.reorder(perm, indices)

	def shuffle(this):
		'''
		Returns a new Poset object (representing the same poset) with the elements in a random order.
		'''
		perm = list(range(len(this)))
		random.shuffle(perm)
		return this.reorder(perm, True)

	def toSage(this):
		'''
		Converts this to an instance of sage.combinat.posets.posets.FinitePoset.
		'''
		import sage
		return sage.combinat.posets.posets.Poset((this.elements,[[p,q] for p in this for q in this if this.lesseq(p,q)]), facade=False)
	def fromSage(P):
		'''
		Convert an instance of sage.combinat.posets.poset.FinitePoset to an instance of Poset.
		'''
		rels = [ [x,y] for x,y in P.relations() if x!=y]
		return Poset(relations=rels)
#		return Poset(incMat = P.lequal_matrix(), elements =  P.list())

	def make_ranks(incMat):
		'''
		Used by the constructor to compute the ranks list for a poset when it isn't provided.
		'''
		if len(incMat)==0: return []
		left = list(range(len(incMat)))
		preranks = {} #keys are indices values are ranks
		while len(left)>0:
			for l in left:
				loi = [i for i in range(len(incMat)) if incMat[i][l]==1]
				if all([i in preranks.keys() for i in loi]): break
#			try:
#				assert(all([i in preranks.keys() for i in loi]))
#			except Exception as e:
#				print('Assertion in Poset.make_ranks failed')
#				print('loi',loi)
#				print('preranks',preranks)
#				raise e
			preranks[l] = 1+max([-1]+[preranks[i] for i in loi])
			left.remove(l)
		return [[j for j in range(len(incMat)) if preranks[j]==i] for i in range(1+max(preranks.values()))]
	##############
	#End Misc
	##############
##############
#Built in posets
##############
def Empty():
	'''
	Returns an empty poset.
	'''
	return Poset(elements = [], ranks = [], incMat = [])

def Bruhat(n):
	r'''
	Returns the type A_{n-1} Bruhat order (symmetric group S_n).

	Bruhat(3)

	   321
	 /    \ 
	231  312
	|  \/  |
	|  /\  |
	132  213
	 \    /
	   123
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

def Butterfly(n):
	r'''
	Returns the rank n+1 bounded poset where ranks 1...n have two elements and all comparisons between ranks.

	Butterfly(2)

	  1
	 / \ 
	a2 b2
	|\ /|
	| X |
	|/ \|
	a1 b1
	 \ /
	  0
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

def Chain(n):
	'''
	Returns the poset on 0...n ordered linearly.

	Chain(3)

	3
	|
	2
	|
	1
	|
	0
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
	Returns the poset of subsets of X, default is {1...n}, ordered by inclusion.

	Boolean(2)

	    (1,2)
	   /    \ 
	(1,)   (2,)
	   \    /
	     ()
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
	'''
	Returns Boolean(n+1) the face lattice of the n-dimensional simplex.
	'''
	return Boolean(n+1)

def Polygon(n):
	r'''
	Returns the face lattice of the n-gon.

	Polygon(3)

	        1
	    /   |   \ 
	(1,2) (3,1) (2,3)
	   |\/     \/|
	   |/\     /\|
	(1,)  (2,)  (3,)
	   \    |   /
	        0
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
	Returns the face lattice of the n-dimensional cube.

	Cube(2)

	      **
	  /  /  \  \ 
	0*  *0  *1  1*
	| \/  \/  \/ |
	| /\  /\  /\ |
	00  01  10  11
	  \  \  /   /
	       0
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
	'''
	Returns the face poset of a cubical complex homeomorphic to the n-dimensional Torus.

	This poset is isomorphic to the cartesian product of n copies of P_m with minimum and maximum adjoined
	where P_m is the face lattice of an m-gon with its minimum and maximum removed.

	When m<= 26 the set is {0,1,...,m-1,A,B,...,[mth leter of the alphabet]}^n and otherwise is {0,...,m-1,*0,...*[m-1]}^n.
	The order relation is
	componentwise where 0<A,[mth letter of the alphabet], 1<A,B,..., m-1<[m-1st letter of the alphabet],[mth letter of the alphabet]  for m<=26, and 0<*1,*2 ... m-1<*[m-1],*0 for m>26
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
	'''
	Returns the face poset of the cubical complex obtained from a 2x...x2 grid of n-cubes via a series of gluings as indicated by the parameter orientations.

	If orientations is [1,...,1] the n-Torus is constructed and if orientations is [-1,...,-1] the
	projective space of dimension n is constructed.

	The dimension of the cubes is len(orientations).

	If orientations[i] == 1 the two ends of the large cube are glued so that points with the same
	image under projecting out the ith coordinate are identified.

	If orientations[i] == -1 points on the two ends of the large cube are identified with their antiodes.

	If orientations[i] is any other value no gluing is performed for that component.
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
	'''
	Returns the face poset of a cubical complex homeomorphic to the KleinBottle().

	Pseudonym for GluedCube([-1,1]).
	'''
	P = GluedCube([-1,1])
	P.name = "Klein Bottle"
	return P

def ProjectiveSpace(n=2):
	'''
	Returns the face poset of a Cubical complex homeomorphic to Project space of dimension n.

	Pseudonym for GluedCube([-1,...,-1]).
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
	'''
	Returns the face poset of the cubical complex that forms a d[0]x...xd[-1] grid of n-cubes.
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
	'''
	Returns either a lower interval [0,t] or the upper interval [t,1] in the uncrossing poset.

	The parameter t should be either a pairing encoded as a list [s_1,t_1,...,s_n,t_n] where
	s_i is paired to t_i or an integer greater than 1. If t is an integer the entire uncrossing
	poset is returned.

	For more info on the uncrossing poset see: https://arxiv.org/abs/1406.5671
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
		transClose(M)
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
	'''
	Returns the poset of subspaces of the vector space F_q^n where F_q is the field with q elements.

	Currently only implemented for q a prime. Raises a not implemented error if q is not prime.
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

def DistributiveLattice(P):
	'''
	Returns the lattice of ideals of a given poset.
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
	elements = []
	for I in ideals:
		ranks[len([c for c in bin(I) if c=='1'])].append(I)
		elements.append(tuple(P[i] for i in range(len(P)) if (1<<i)&I!=0))
	def less(I,J):
		return I!=J and all(i in J for i in I)
	return Poset(elements = elements, less = less)

def LatticeOfFlats(data):
	'''
	Returns the lattice of flats given either a list of edges of a graph or a the rank function of a (poly)matroid.

	When the input represents a graph it should be in the format [[i_1,j_1],...,[i_n,j_n]]
	where the pair [i_k,j_k] represents an edge between i and j in the graph.

	When the input represents a (poly)matroid the input should be a list of the ranks of
	sets ordered reverse lexicographically (i.e. binary order). For example, if f is the
	rank function of a (poly)matroid with ground set size 3 the input should be
		[f({}),f({1}),f({2}),f({1,2}),f({3}),f({1,3}),f({2,3}),f({1,2,3})].

	Input representing a polymatroid need not actually represent a polymatroid, no checks
	are done for the axioms. This function may return a poset that isn't a lattice if
	the input function isn't submodular or a preorder that isn't a poset if the input
	is not order-preserving.
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

def MinorPoset(L,genL=None):
	'''
	Returns the minor poset given a lattice L and a list of generators genL.

	The join irreducibles are automatically added to L. If genL is not provided the generating set will be only the
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
			k = join(i,j,L)
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
	transClose(minors_M)

	class GenlattDiagram:

		def __init__(this, L, G, **kwargs):
			super().__init__(L, **kwargs)
			this.G = G
			this.edges = set()
			#find all extra edges of diagram
			for l in L:
				for g in G:
					lg = L.join(l,g)
					if lg == l or len(L.interval(l,lg))==2: continue
					this.edges.add( (l,lg) )

		def latex(this, **kwargs):
			ret = super().latex(**kwargs)
			end_index = ret.index('\\end{tikzpicture}')
			head = ret[:end_index]
			tail = ret[ret.index:]



	class MinorPosetHasseDiagram:

		def __init__(this, P, **kwargs):
			super().__init__(P, **kwargs)
			if 'nodewidth' in kwargs:
				this.nodewidth = kwargs['nodewidth']
			else:
				this.nodewidth = 0.9*(this.width / max([len(r) for r in this.P.ranks]))

			if 'nodeheight' in kwargs:
				this.nodeheight = kwargs['nodeheight']
			else:
				this.nodeheight = 0.5*(this.height / len(this.P.ranks))

		def nodeLabel(this, i):
			if this.in_latex:
				beginning = '\\langle'
				ending = '\\rangle'
				sep = '\\vert'
			else:
				beginning = '<'
				ending = '>'
				sep = '|'

			return beginning + ','.join([str(x) for x in this.P[i][1]]) + setp + str(this.P[i][0])+ending

		def nodeName(this, i):
			return '/'.join(str(this.P[i][0])+[str(x) for x in this.P[i][1]])

		def nodeDraw(this, i):
			pass
			#Draw the lattice in gray

			#Draw the minor in black



	P = Poset(minors_M, minors, minors_ranks)
	P.elements = [tuple([L_set[M[0]],tuple(L_set[g] for g in M[1])]) for M in P]
	P = P.adjoin_zerohat()
	#cache some values for queries
	P.cache['isRanked()']=True
	P.cache['isEulerian()']=True
	P.cache['isGorenstein()']=True
	return P

##############
#End Built in posets
##############
##############
#Hasse drawer
##############
import math
import random
import tkinter as tk
class HasseDiagram:
	r'''
	A class that can produce latex/tikz code for the Hasse diagram of a poset or display the diagram in a window using tkinter.


	##########################################
	#Overview
	##########################################

	An instance of this class is attached to each instance of Poset. This class
	is used to produce latex code for a poset when Poset.latex() is called or
	to display a poset in a new window when Poset.show() is called. These functions
	are wrappers for HasseDiagram.latex() and HasseDiagram.tkinter().

	The constructor for this class takes keyword arguments that control how the
	Hasse diagram is drawn. These keyword arguments set the default options
	for that given instance of HasseDiagram. When calling latex() to produce latex code
	or tkinter() to draw the diagram in a tkinter window the same keyword arguments can
	be passed to control how the diagram is drawn during that particular operation.

	Options such as height, width, scale, etc. are values whose defaults can
	be overriden by passing keyword arguments  either to
	the constructor or the functions themselves.

	Other options such as loc_x, loc_y, nodeLabel or nodeDraw are functions.
	The default values for these functions are class methods.

	##########################################
	#Overriding function parameters
	##########################################

	Function parameters can be overriden in two ways. The first option is to
	make a function with the same signature as the default function and to pass that
	function as a keyword argument to the constructor or latex()/tkinter() when called.

	For example:

		def nodeLabel(this, i):
			return str(this.P.mobius(0, this.P[i]))

		#P is a Poset already constructed that has a minimum 0
		P.hasseDiagram.tkinter(nodeLabel = nodeLabel)

	The code above will show a Hasse Diagram of P with the elements labeled by
	the mobius values $\mu(0,p)$.

	When overriding function parameters the first argument is always the HasseDiagram
	instance. HasseDiagram has an attribute for each option described below as well
	as the following attributes:

		P - The poset to be drawn.

		in_tkinter - Boolean indicating whether tkinter() is being executed.

		in_latex - Boolean indicating whether latex() is being executed.

		canvas - While tkinter() is being executed this is the tkinter.Canvas
			object being drawn to.

	Note that any function parameters, such as nodeLabel, are set via
		this.nodeLabel = #provided function
	so if you intend to call these functions you must pass this as an argument via
		this.nodeLabel(this, i)
	The class methods remain unchanged of course, for example HasseDiagram.nodeLabel
	always refers to the default implementation.

	##############
	#Subclassing
	##############

	The second way to override a function parameter is via subclassing. This is more
	convenient if overriding several function parameters at once or if the computations
	are more involved. It is also useful for adding extra parameters. Any variables initialized
	in the constructor are saved at the beginning or latex() or tkinter(), overriden during
	execution of the function by any provided keyword arguments, and restored at the end of
	execution. The mobius example above can be accomplished by subclassing as
	follows:
		class MobiusHasseDiagram(HasseDiagram):

			def nodeLabel(this, i):
				zerohat = this.P.min()[0]
				return str(this.P.mobius(zerohat, this.P[i]))

		P.hasseDiagram = MobiusHasseDiagram(P)
		P.hasseDiagram.tkinter()

	To provide an option that changes what element the mobius value is computed
	from just set the value in the constructor.

		class MobiusHasseDiagram(HasseDiagram):

			def __init__(this, P, z = None, **kwargs):
				super().__init__(P, **kwargs)

				if z == None:
					this.z = this.P.min()[0] #z defaults to first minimal element
				else:
					this.z = z

			def nodeLabel(this, i):
				return str(this.P.mobius(this.z, this.P[i]))

		#P is a Poset with minimum 0
		P.hasseDiagram = MobiusHasseDiagram(P)
		P.hasseDiagram.tkinter() #labels are $\mu(0, x)$
		P.hasseDiagram.tkinter(z = P[0]) #labels are $\mu(P_0, x)$
		P.hasseDiagram.tkinter() #labels are $\mu(0, x)$

	Note you can pass a class to the Poset constructor to construct a poset with
	a hasseDiagram of that class.

	##########################################
	#Keyword arguments
	##########################################

	Options that affect about latex() and tkinter():

		width - Width of the diagram. When calling latex() this is the width
			in tikz units, for tkinter() the units are 1/10th of tkinter's units.

			The default value is 18.

		height - Height of the diagram, uses the same units as width.

			The default value is 30.

		labels - Whether to display labels for elements or just a filled circle.

			The default value is True.

		ptsize - No effect when labels is True, when labels is False this is the size
			of the circles shown for elements. When calling tkinter() this can
			be either a number or a string; if a string the last two characters
			are ignored.

			The default value is '2pt'.

		indices_for_nodes - If True HasseDiagram.nodeLabel is not called and the
			node text is the index of the element in the poset.
			If labels is False this argument has no effect.

			The default value is False.

		nodeLabel - A function that given this and an index returns a string to label
			the corresponding element by.

			The default value is HasseDiagram.nodeLabel.

		loc_x - A function that given this and an index returns the x-coordinate
			of the element in the diagram as a string. 0 represents the middle
			of the screen, positive values rightward and negative leftward.
			Returned value should lie in the range [-width/2, width/2].

			The default value is HasseDiagram.loc_x.

		loc_y - A function that given this and an index returns the y-coordinate
			of the element in the diagram as a string. 0 represents the bottom
			of the screen, positive values extend upward. Returned value should
			lie in the range [0, height].

			The default value is HasseDiagram.loc_y.

		jiggle
		jiggle_x
		jiggle_y  - Coordinates of all elements are perturbed by a random vector in
			the rectangle
				-jiggle-jiggle_x <= x <= jiggle+jiggle_x
				-jiggle-jiggle_y <= y <= jiggle+jiggle_y
			This can be useful if you want to prevent cover lines from successive ranks
			aligning to form the illusion of a line crossing between two ranks;
			or when drawing unranked posets if a line happens to cross over an
			element. The perturbation occurs in loc_x and loc_y so if these are
			overwritten and you want to preserve this behaviour add a line
			to the end of loc_x such as
				x = x+random.uniform(-jiggle-jiggle_x,jiggle+jiggle_x)

			The default values are 0.

	Options that affect only tkinter():

		scale - All coordinates are scaled by 10*scale.

			The default value is 1.

		padding - A border of this width is added around all sides of the diagram.
			This is affected by scale.

			The default value is 1.

		offset - Cover lines start above the bottom element and end below the top
			element, this controls the separation.

			The default value is 1.

		nodeDraw - When labels is False this function is called instead of placing
			anything for the node. The function is passed this and an index to
			the element to be drawn. nodeDraw should use the tkinter.Canvas
			object this.canvas to draw. The center of your diagram should be
			at the point
				x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
				y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))
			For larger diagrams make sure to increase height and width as well as offset.

			The default value is HasseDiagram.nodeDraw.

	Options that affect only latex():

		extra_packages - A string that when calling latex() is placed in the preamble.
			It should be used to include any extra packages or define commands
			needed to produce node labels. This has no effect when standalone is False.

			The default value is ''.

		nodescale - Each node is wrapped in '\\scalebox{'+nodescale+'}'.

			The default value is '1'.

		tikzscale - This is the scale parameter for the tikz environment, i.e. the
			tikz environment containing the figure begins
			'\\begin{tikzpicture}[scale='+tikzscale+']'.

			The default value is '1'.

		line_options - Tikz options to be included on every line drawn, i.e. lines
			will be written as '\\draw['+line_options+'](...'.

			The default value is ''.

		northsouth - If True lines are not drawn between nodes directly but from
			node.north to node.south which makes lines come together just beneath
			and above nodes. When False lines are drawn directly to nodes which
			makes lines directed towards the center of nodes.

			The default is True.

		lowsuffix - When this is nonempty lines will be drawn to node.lowsuffix instead of
			directly to nodes for the higher node in each cover. If northsouth
			is True this has no effect, '.south' is used for the low suffix.

			The default is ''.

		highsuffix - This is the suffix for the bottom node in each cover. If northsouth
			is True this has no effect, '.north' is used for the high suffix.

			The default is ''.

		decoration - A function that takes this and indices
			i and j representing a cover this.P[i]<this.P[j] to be drawn and
			returns a string of tikz line options to be included (along with any
			line options from line_options) on that draw command.

			The default value is HasseDiagram.decoration which
			returns an empty string.

		nodeName - A function that takes this and an index i representing an element
			whose node is to be drawn and returns the name of the node in tikz.
			This does not affect the image but is useful if you intend to edit
			the latex code and want the node names to be human readable.

			The default value HasseDiagram.nodeName returns str(i).

		standalone - When True a preamble is added to the beginning and
			'\\end{document}' is added to the end so that the returned string
			is a full latex document that can be compiled. Compiling requires
			the latex packages tikz (pgf) and preview. The resulting figure can be
			incorporated into another latex document with \includegraphics.

			When False only the code for the figure is returned; the return value
			begins with \begin{tikzpicture} and ends with \end{tikzpicture}.

			The default is False.
	'''
	def __init__(this, P, **kwargs):
		'''
		See HasseDiagram.
		'''
		this.P = P
		this.in_latex = False
		this.in_tkinter = False

		defaults = {
			'extra_packages':'',
			'nodescale':'1',
			'scale':'1',
			'tikzscale':'1',
			'line_options':'',
			'northsouth':True,
			'lowsuffix':'',
			'highsuffix':'',
			'labels': True,
			'ptsize': '2pt',
			'height': 30,
			'width': 18,
			'decoration': type(this).decoration,
			'loc_x': type(this).loc_x,
			'loc_y': type(this).loc_y,
			'nodeLabel': type(this).nodeLabel,
			'nodeName': type(this).nodeName,
			'indices_for_nodes': False,
			'jiggle': 0,
			'jiggle_x': 0,
			'jiggle_y': 0,
			'standalone': False,
			'padding': 20,
			'nodeDraw': type(this).nodeDraw,
			'offset': 1
			}

		for (k,v) in defaults.items():
			if k in kwargs: this.__dict__[k] = kwargs[k]
			else: this.__dict__[k] = v

	def decoration(this,i,j):
		'''
		This is the default implementation of decoration, it returns an empty string.
		'''
		return ''

	def loc_x(this, i):
		'''
		This is the default implementation of loc_x.

		This spaces elements along each rank evenly. The length of a rank is the
		ratio of the logarithms of the size of the rank and the size of the largest rank.

		The return value is a string.
		'''
		len_P=len(this.P)
		rk = this.P.rank(i, True)
		if len(this.P.ranks[rk])==1: return '0'
		rkwidth=math.log(float(len(this.P.ranks[rk])))/math.log(float(this.maxrksize))*float(this.width)
		index=this.P.ranks[rk].index(i)
		ret = (float(index)/float(len(this.P.ranks[rk])-1))*rkwidth - (rkwidth/2.0)
		jiggle = this.jiggle_x + this.jiggle
		return str( ret + random.uniform(-jiggle, jiggle) )

	def loc_y(this,i):
		'''
		This is the default value of loc_y.

		This evenly spaces ranks.

		The return value is a string.
		'''
		#TODO make max rank return height
		rk = this.P.rank(i, True)
		delta = float(this.height)/float(len(this.P.ranks)-1)
		jiggle = this.jiggle_y + this.jiggle
		return str( rk*delta + random.uniform(-jiggle,jiggle) )
#		if this.P.min(True) == [i]: return '0'
#		if this.P.max(True) == [i]: return str(this.height)
#		if i in this.P.max(True): return str(this.height)
#		rk = this.P.rank(i, True)
#		delta=float(this.height)/float(len(this.P.ranks)-2)
#		ret = rk*delta-delta/2.0
#		jiggle = this.jiggle_y + this.jiggle
#		return str( ret + random.uniform(-jiggle, jiggle) )

	def nodeLabel(this,i):
		'''
		This is the default implementation of nodeLabel.

		The ith element is returned cast to a string.
		'''
		return str(this.P.elements[i])
#		return str(i) if this.indices_for_nodes else str(this.P.elements[i])
#		return str(i)

	def nodeName(this,i):
		'''
		This is the default implementation of nodeName.

		i is returned cast to a string.
		'''
		return str(i)

	def nodeDraw(this, i):
		'''
		This is the default implementation of nodeDraw.

		This draws a filled black circle of radius ptsize/2.
		'''
		ptsize = this.ptsize if type(this.ptsize)==int else float(this.ptsize[:-2])

		x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
		y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))

		this.canvas.create_oval(x-ptsize/2,y-ptsize/2,x+ptsize/2,y+ptsize/2, fill='black')
		return

	def tkinter(this, **kwargs):
		'''
		Opens a window using tkinter and draws the Hasse diagram.

		The keyword arguments are described in HasseDiagram.
		'''
		#save default parameters to restore aferwards
		defaults = this.__dict__.copy()
		#update parameters from kwargs
		this.__dict__.update(kwargs)
		this.in_tkinter = True

		this.maxrksize = max([len(r) for r in this.P.ranks])

		root = tk.Tk()
		root.title("Hasse diagram of "+(this.P.name if hasattr(this.P,"name") else "a poset"))
		scaling = float(this.scale)*10 #TODO get rid of
		padding = float(this.padding) #TODO get rid of
		this.scale = float(this.scale)*10
		this.padding = float(this.padding)
		width = float(this.width)*scaling
		height = float(this.height)*scaling
		canvas = tk.Canvas(root, width=width+2*padding, height=height+2*padding)
		this.canvas = canvas
		canvas.pack(fill = "both", expand = True)
		for r in range(len(this.P.ranks)):
			for i in this.P.ranks[r]:
				x = float(this.loc_x(this,i))*scaling + width/2 + padding
				y = 2*padding+height-(float(this.loc_y(this,i))*scaling + padding)
				if not this.labels:
#					canvas.create_oval(x-ptsize/2,y-ptsize/2,x+ptsize/2,y+ptsize/2, fill='black')
					this.nodeDraw(this, i)
				else:
					canvas.create_text(x,y,text=str(i) if this.indices_for_nodes else this.nodeLabel(this,i),anchor='c')
				if r == len(this.P.ranks)-1: continue
				#TODO: should we worry about efficiency? (I mean probably not but this feels ugly)
				for j in this.P.ranks[r+1] if this.P.isRanked() else this.P.filter([i], indices = True, strict = True).min():
					if not this.P.less(i,j,True): continue
					xj = float(this.loc_x(this,j))*scaling + width/2 + padding
					yj = 2*padding+height-(float(this.loc_y(this,j))*scaling + padding)
					canvas.create_line(x,y-scaling*this.offset,xj,yj+scaling*this.offset)
		root.mainloop() #makes this function blocking so you can actually see the poset when ran in a script
		this.__dict__.update(defaults)

	def latex(this, **kwargs):
		'''
		Returns a string to depict the Hasse diagram in Latex.

		The keyword arguments are described in HasseDiagram.
		'''
		this.maxrksize = max([len(r) for r in this.P.ranks])
		defaults = this.__dict__.copy()
		this.__dict__.update(kwargs)
		this.in_latex = True

		if this.northsouth:
			this.lowsuffix = '.north'
			this.highsuffix = '.south'
		#write preamble
		ret=[]
#		ret.append('%'+' '.join([a for a in sys.argv])+'\n') #TODO: put in actual parameters here
		if this.standalone:
			ret.append('\\documentclass{article}\n\\usepackage{tikz}\n')
			ret.append(this.extra_packages)
			ret.append('\n\\usepackage[psfixbb,graphics,tightpage,active]{preview}\n')
			ret.append('\\PreviewEnvironment{tikzpicture}\n\\usepackage[margin=0in]{geometry}\n')
			ret.append('\\begin{document}\n\\pagestyle{empty}\n')
		ret.append('\\begin{tikzpicture}\n')

		#write nodes for the poset elements
		if not this.labels:
			for rk in this.P.ranks:
				for r in rk:
					name=this.nodeName(this, r)
					ret.append('\\coordinate('+name+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+');\n')
					ret.append('\\fill('+name+')circle('+this.ptsize+');\n')
		else:
			for rk in this.P.ranks:
				for r in rk:
					ret.append('\\node('+this.nodeName(this, r)+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+')\n{')
					ret.append('\\scalebox{'+this.nodescale+"}{")
					ret.append(str(r) if this.indices_for_nodes else this.nodeLabel(this, r))
					ret.append('}};\n\n')

		#draw lines for covers
		if this.P.isRanked():
			for r in range(0,len(this.P.ranks)-1):
				for i in this.P.ranks[r]:
					for j in this.P.ranks[r+1]:
#			for r in range(0,len(ranks)-1):
#				for i in ranks[r]:
#					for j in ranks[r+1]:
						if this.P.less(i,j,True):
							options=this.decoration(this, i,j)+(','+this.line_options if this.line_options!='' else "")
							if len(options)>0: options='['+options+']'
							ret.append('\\draw'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")

		#for unranked version for each element we have to check all the higher length elements
		if not this.P.isRanked():
#			for r in range(0,len(this.P.ranks)-1):
#				for i in this.P.ranks[r]:
#					for s in this.P.ranks[r+1:]: #<--there's 2 colons this time
			for r in len(0,len(ranks)-1):
				for i in ranks[r]:
					uoi=[] #elements above i
					for s in ranks[r+1:]:
						for j in s:
							if this.P.less(i,j, True):
								uoi.append(j)
					covers=poset_min(uoi,lambda i,j: this.P.less(i,j, True))
					for j in covers:
						options=this.decoration(this, i,j)+(','+this.line_options if this.line_options!='' else "")
						if len(options)>0: options='['+options+']'
						ret.append('\\draw'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")
		ret.append('\\end{tikzpicture}')
		if this.standalone:
			ret.append('\n\\end{document}')

		this.__dict__.update(defaults)
		return ''.join(ret)


##############
#End Hasse drawer
##############
##############
#cdIndex functions
##############
##########################################
#code below is copied from cdIndex.py
#from https://github.com/WilliamGustafson/cdIndexCalculator
##########################################
#a few general poset utilities
#
#within this module a poset is represented via matrix M
#whose entry M[i][j] is 1 if i<j, -1 if i>j and 0 otherwise.
#
#many functions also require a rank list provided whose ith entry
#is a list of the indices corresponding to rank i elements.
##########################################
import sys

def debugPrint(s, x=None):
	if 'debug' in sys.argv:
		print('')
		if x ==None: print(s)
		else:
			print(s + ": ",end='')
			print(x)
		print('')

def rankOf(ranks,x):
	for i in range(0, len(ranks)):
		if x in ranks[i]:
			return i
	return -1

#given a matrix encoding all the cover relations of a poset this function
#alters the input to compute the matrix representing the poset
def transClose(M):
	for i in range(0,len(M)):
		#uoi for upper order ideal
		uoi = [x for x in range(0,len(M)) if M[i][x] == 1]
		while True:
			next = [x for x in uoi]
			for x in uoi:
				for y in range(0,len(M)):
					if M[x][y] == 1 and y not in next: next.append(y)
			if uoi == next: break
			uoi = next

		for x in uoi:
			M[i][x] = 1
			M[x][i] = -1

#computes the join of i and j in M, returns -1 if it does not exist
def join(i,j,M):
	if i==j: return i
	if M[i][j] == -1: return i
	if M[i][j] == 1: return j
	m = [x for x in range(0,len(M)) if M[i][x] == 1 and M[j][x] == 1]
	for x in range(0,len(m)):
		isJoin = True
		for y in range(0,len(m)):
			if x!=y and M[m[x]][m[y]] != 1:
				isJoin = False
				break
		if isJoin: return m[x]
	return None

#computes mobius function of [i,j], useful for debugging
def mobius(M, ranks, i, j):
	if i == j: return 1
	if M[i][j] != 1: return None
	ret = 0
	for x in range(0,len(M)):
		if M[x][j] == 1 and M[x][i] == -1:
			ret -= mobius(M,ranks,i,x)
	ret -= 1 #mobius(M,rank,i,i)
	return ret


##########################################
#helper functions for polynomials with noncommutative variables
#
#monomials are stored as a list containing the coefficient and a string for the variables
#polynomials are lists of monomials
##########################################

def multPolys(p,q):
	r=[[x[0]*y[0],x[1]+y[1]] for x in p for y in q]
	#collect terms
	ret=[]
	for x in r:
		monoms=[y[1] for y in ret]
		if x[1] not in monoms:
			ret.append(x)
			continue
		ret[monoms.index(x[1])][0]+=x[0]
	return ret

def multManyPolys(P):
	ret=P[0]
	while len(P)>1:
		P=P[1:]
		ret=multPolys(ret,P[0])
	return p

def addPolys(p,q):
	ret=[x for x in p]
	for x in q:
		temp=[y[1] for y in ret]
		if x[1] in temp:
			ret[temp.index(x[1])][0]+=x[0]
		else:
			ret.append(x)
	return [x for x in ret if x[0]!=0]

def subPolyForMonom(x,p,m):
	X=[[y[0],y[1].replace(m,'*')] for y in x]
	ret=[]
	for y in X:
		q=[[y[0],'']]
		for i in range(0,len(y[1])):
			if y[1][i]=='*':
				q=multPolys(q,p)
			else: #mult by the monomial
				for j in range(0,len(q)):
					q[j][1]+=y[1][i]
		ret=addPolys(ret,q)
	return ret

##########################################
#methods to calculate flag vector table
##########################################

#fVector returns the flag f vector f_S(M)
#fVectorCalc calculates this recursively

def fVectorCalc(ranks,S,M, i, count):
	newCount = count
	if S == []: return 1
	for j in ranks[S[0]]:
		if M[i][j] == 1:
			newCount += fVectorCalc(ranks, S[1:], M, j, count)
	return newCount

def fVector(M,ranks,S):
	return fVectorCalc(ranks,S,M,ranks[0][0],0)

#returns a list whose entries are lists containing a subset of the ranks
#and the flag f and h vector entries
def makeFlagVectorsTable(M,ranks):
	table = [[[],1,1]]

	if len(ranks)<=2: return table

	#iterate over all subsets of the ranks
	for i in range(1,1<<(len(ranks)-1)-1):
		#construct the corresponding set S
		pad = 1
		elem = 1
		S = []
		while pad <= i:
			if pad&i:
				S.append(elem)

			pad <<= 1
			elem += 1
		table.append([S,fVector(M,ranks,S),0])


	#do PIE to get the flag h vectors
	for i in range(1,len(table)):
		sign = (2*(len(table[i][0])%2)) - 1 #is -1 if even number of elements, 1 if odd
		for j in range(0,i+1):
			if set(table[j][0]).issubset(table[i][0]):
				table[i][2] += sign*(2*(len(table[j][0])%2)-1)*table[j][1]
	return table

##########################################
#methods for computing the ab and cd indices
##########################################

def abIndex(table, rank):
	ab = []
	for x in table:
		u = ['a']*(rank-1)
		for s in x[0]: u[s-1] = 'b'
		ab.append([x[2],''.join(u)])

	return ab

#returns a given ab polynomial expressed in terms of c and d if possible
#if not possible returns ab
def abToCd(ab):
	if len(ab)==0: return ab
	#substitue a->c+e and b->c-e
	#where e=a-b
	#this scales by a factor of 2^n (n+1 is number of ranks)
	ce=subPolyForMonom(subPolyForMonom(ab,[[1,'c'],[1,'e']],'a'),[[1,'c'],[-1,'e']],'b')
	debugPrint('ce',ce)

	cd=subPolyForMonom(ce,[[1,'cc'],[-2,'d']],'ee')
	leftover_e=False
	for m in cd:
		if 'e' in m[1]:
#			break
			if 'e' in m[1][:-1]: return ab
			leftover_e = True
	#divide coefficients by 2^n
	power=sum([2 if cd[0][1][i]=='d' else 1 for i in range(len(cd[0][1]))])
	if leftover_e:
		return ab
#		ret = [[],[],[]]
#		for x in cd:
#			coeff = x[0]>>power
#			monom = x[1][:-1]
#			if 'e' not in x[1]:
#				if x[1][-1] == 'c':
#					ret[1] = addPolys(ret[1], [[coeff,monom]])
#					ret[2] = addPolys(ret[2], [[coeff, monom]])
##					ret[1].append([[coeff, monom]])
##					ret[2].append([[coeff, monom]])
#				else: # x[1][-1] == 'd':
#					ret[0] = addPolys(ret[0], [[coeff, monom]])
##					ret[0].append([[coeff, monom]])
#			else:
#				ret[1] = addPolys(ret[1], [[coeff, monom]])
#				ret[2] = addPolys(ret[2], [[-coeff, monom]])
##				ret[1].append([coeff, monom])
##				ret[2].append([-coeff,monom])
		return ret
	return [[x[0]>>power,x[1]] for x in cd]

#returns the cd-index of the poset encoded by M
#
#optionally provide a list for table or ab and these will be
#filled with the flag vectors table and ab-index respectively

def cdIndex(M, ranks, table=None, ab=None, convert_to_cd=True):
	if table == None: table = []
	if ab == None: ab = []

	table += makeFlagVectorsTable(M,ranks)

	debugPrint('table', table)

	ab += abIndex(table, len(ranks)-1)

	debugPrint('ab',ab)
	if not convert_to_cd: return ab
	cd=abToCd(ab)
	#actual cd-index
	if type(cd[0][0])==int: cd.sort(key=lambda x:x[1])
	else:
		#lower Gorenstein* cdd,cda,cdb index
		cd = (sorted(cd[0]), sorted(cd[1]), sorted(cd[2]))
	debugPrint('cd', cd)
	return cd

##########################################
#formatting methods
##########################################

#returns a string representing the given polynomial in latex
def cdIndexLatex(cd):
	s = ""
	for i in range(0,len(cd)):
		if cd[i][0] == 0: continue
		if cd[i][0] == -1: s+= '-'
		elif cd[i][0] != 1: s += str(cd[i][0])
		current = ''
		power = 0
		for c in cd[i][1]:
			if current == '':
				current = c
				power = 1
				continue
			if c == current:
				power += 1
				continue
			s += current
			if power != 1: s += '^{' + str(power) + '}'
			current = c
			power = 1
		s += current
		if power != 1 and power != 0: s += '^{' + str(power) + '}'
		if power == 0 and current == "": s += '1'

		if i != len(cd)-1:
			if cd[i+1][0] >= 0: s += "+"
	if s == '': return '0'
	return s

#returns a string with latex code to display the given flag vectors table
def flagVectorsLatex(table):
	ret = "\\begin{longtable}{c|c|c}\n\t$S$&$f_S$&$h_S$\\\\\n\t\\hline\n\t\\endhead\n"
	for t in table:
		ret += "\t\\{"
		ret += ','.join([str(x) for x in t[0]])
		ret = ret+"\\} & " + str(t[1]) + " & " + str(t[2]) + "\\\\\n\t\\hline\n"
	return ret + "\\end{longtable}"


import numpy as np

def cdIndex_incidence_algebra(M,ranks,v=None):
#	incMat=np.array([[1 if M[i][j]==1 or i==j else 0 for j in range(len(M))] for i in range(len(M))])
#	mu=np.linalg.inv(incMat)
	#alpha=np.array([[(-1)**rankOf(ranks,y)*incMat[x][y] for y in range(len(M))] for x in range(len(M))])
	#find zerohat
	zh = [i for i in range(len(M)) if -1 not in M[i]][0]
	alpha=np.array([[(-1)**rankOf(ranks, y) if x==y or M[x][y]==1 else 0 for y in range(len(M))] for x in range(len(M))])
#	deltas=[np.array([[1 if i==j and i in [x for r in ranks[:k+1] for x in r] else 0 for j in range(len(M))]for i in range(len(M))])for k in range(len(ranks))]
	deltas=[np.array([[1 if i==j and i in itertools.chain(*ranks[:k+1]) else 0 for j in range(len(M))] for i in range(len(M))]) for k in range(len(ranks))]
	def make_cd_ops(n,ops=[]):
		ret=[]
		if n>-1:
			C=deltas[n]
#			c_ops=[(np.matmul(C,O),'c'+w) for O,w in ops]
			c_ops=[(O,'c'+w) for O,w in ops] #you don't actually have to do the C ops they're already accounted for
			ret+=make_cd_ops(n-1,c_ops)
			if n>0:
#				D=np.matmul(deltas[n-1],np.matmul((-1)**n*alpha-deltas[-1],deltas[n]))
				D=np.matmul((-1)**n*alpha-deltas[-1],deltas[n])
				d_ops=[(np.matmul(D,O),'d'+w) for O,w in ops]
				ret+=make_cd_ops(n-2,d_ops)
			return ret
		return ops
	if type(v) == type(None):
		v = np.array([1 for x in M])
	ops=make_cd_ops(len(ranks)-3,[(deltas[len(ranks)-1],'')])
	ret=[[int(np.matmul(O,v)[ranks[0]]),w] for O,w in ops]
	ret = sorted([x for x in ret if x[0]!=0],key=lambda x:x[1])
	return ret
def cdIndex_coefficient_mat(M,ranks,u):
	#find zerohat
	zh = [i for i in range(len(M)) if -1 not in M[i]][0]
	alpha=np.array([[(-1)**rankOf(ranks, y) if x==y or M[x][y]==1 else 0 for y in range(len(M))] for x in range(len(M))])
	deltas=[np.array([[1 if i==j and i in itertools.chain(*ranks[:k+1]) else 0 for j in range(len(M))] for i in range(len(M))]) for k in range(len(ranks))]
	u = u.replace('d','_d')
	codeg = len(ranks)-2-len(u)
	X = deltas[codeg]
	for i in range(len(u)):
		if u[i] == 'd':
			X = np.matmul(X, np.matmul((-1)**(codeg+i)*alpha-deltas[-1],deltas[codeg+i]))
	return X
##############
#End cdIndex functions
##############
