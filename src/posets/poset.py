##########################################
#TODO
##########################################
#Make Bruhat directly should be faster
#
#PartitionLattice and those after needs figures
#
#get rid of requires decorator and try warpped imports
#
#add comment of what imports are for
#
#latex and draw should get lines via covers
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
#	Subclass HasseDiagram for the Boolean algebra and use that same class for (poly)matroids?
#
#We should have different functions for text for tikz and tkinter with the defaults the
#same function
#
#Write README with some examples.
##########################################
r'''
@no_list@sections_order@Poset@PosetIsoClass@HasseDiagram@Built in posets@Utilities@Timer@@
A module for creating and manipulating partially ordered sets.

This module provides a class \verb|Poset| that encodes a finite poset.
The \verb|Poset| class provides:

\begin{enumerate}
	\item{
	Operations between posets such as disjoint  unions, Cartesian products
	star products and diamond products.
	}
	\item{
	Methods to calculate data about a given poset such as flag $f$- and $h$-vectors,
	the \textbf{cd}-index and M\"obius function values.
	}
	\item{
	Examples of posets such as Chains, Boolean algebras, face lattices of cubes,
	type A Bruhat orders, uncrossing posets and more.
	}
\end{enumerate}
'''
import itertools
import copy
import decorator
import collections
import math
import random
from .polynomial import *
from .hasseDiagram import *
from .utils import *
try:
	import numpy as np
except:
	np = "numpy"

import time
###########################################
#Poset Class
##########################################
class Poset:
	r'''
	@is_section@section_key@0@
	@sections_order@Operations@Subposet Selection@Internal Computations@Queries@Invariants@Maps@Miscellaneous@PosetIsoClass@@
	A class representing a finite partially ordered set.

	Constructor arguments:

	\begin{itemize}
		\item[]{
		\verb|incMat| -- A matrix whose $i,j$ entry is 1 if the $i$th element is strictly less
			than the $j$th element.
		}
		\item[]{
		\verb|elements| -- A list specifying the elements of the poset.

			The default value is \verb|[0,...,len(incMat)-1]|
		}
		\item[]{
		\verb|ranks| -- A list of lists. The $i$th list is a list of indices of element of
			length $i$.
		}
		\item[]{
		\verb|relations| -- Either a list of pairs $(x,y)$ such that $x<y$ or a dictionary
			whose values are lists of elements greater than the associated key.
			This is used to construct incMat if it is not provided.
		}
		\item[]{
		\verb|less| -- A function that given two elements $p,q$ returns \verb|True| when
			$p < q$. This is used to construct incMat if neither \verb|incMat|
			nor \verb|relations| are provided.
		}
		\item[]{
		\verb|indices| -- A boolean indicating indices instead of elements are used in
			\verb|relations| and \verb|less|. The default is \verb|False|.
		}
		\item[]{
		\verb|name| -- An optional identifier if not provided no name attribute is set.
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
		\verb|trans_close| -- If \verb|True| the transitive closure of \verb|incMat| is
			computed, this should be \verb|False| only if the provided matrix satisfies
				\[
				\verb|incMat[i][j] ==|\begin{cases}
						1 & \text{when } i<j\\
						-1 & \text{when } i>j\\
						0 & \text{otherwise}.
					\end{cases}
				\]
		}
	\end{itemize}
	The keyword arguments are passed to \verb|HasseDiagram| (or \verb|hasse_class|
	if specified).

	To construct a poset you must pass at least either \verb|incMat| or \verb|less|.

	In the constructor the transitive closure of \verb|incMat| is computed
	(unless \verb|trans_close=False| is set)
	so it is only essential that \verb|incMat[i][j] == 1| when $i$ is covered by $j$. The diagonal
	entries \verb|incMat[i][i]| should always be zero.

	If \verb|elements| is not provided it will be set to \verb|[0,...,len(incMat)-1]|

	\verb|ranks| is inessential, if not provided it will be computed.

	Function calls to several of the more costly computations are cached. Generally
	functions in this class do not change the poset but instead return a new poset.
	\verb|Poset| objects may be considered immutable (this is not enforced in any way),
	or if you alter a poset you should clear the cache via: \verb|this.cache = {}|.
	'''
	__slots__ = ('cache','hasseDiagram','incMat','elements','ranks','name')
	def __init__(this, incMat=None, elements=None, ranks=None, less=None, name='', hasse_class=None, trans_close=True, relations=None, indices=False, **kwargs):
		r'''
		@section@Miscellaneous@
		See \verb|Poset|.
		'''
		if isinstance(incMat, Poset):
			for s in Poset.__slots__:
				setattr(this, s, getattr(incMat, s))
			return

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

		if len(this.incMat)>0 and trans_close: Poset.transClose(this.incMat)
		#####
		#####
		if not hasattr(this, 'elements'):
			this.elements = list(range(len(this.incMat))) if elements == None else elements
		this.ranks = ranks if ranks!=None else Poset.make_ranks(this.incMat)
		this.cache = {}
		this.name = name
		if hasse_class == None: hasse_class = HasseDiagram
		this.hasseDiagram = hasse_class(this, **kwargs)

	def transClose(M):
		r'''
		@section@Miscellaneous@
		Given a matrix with entries $1,-1,0$ encoding a relation computes the transitive closure.
		'''
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

	def __str__(this):
		r'''
		@section@Miscellaneous@
		Returns a nicely formatted string listing the zeta matrix, the ranks list and the elements of the poset.
		'''
		ret = ['zeta = ['] + [' '.join([(' ' if x>=0 else '')+str(x) for x in z]) for z in this.zeta()]+[']']
		if hasattr(this, 'name'):
			ret = [this.name]+ret
		ret.append('ranks = '+str(this.ranks))
		ret.append('elements = '+str(this.elements))
		return '\n'.join(ret)

	def __repr__(this):
		r'''
		@section@Miscellaneous@
		Gives a string that can be evaluated to recreate the poset.

		To \verb|eval| the returned string \verb|Poset| must be in the namespace and \verb|repr(this.elements)|
		must return a suitable string for evaluation.
		'''
		return 'Poset(incMat='+repr(this.incMat)+', elements='+repr(this.elements)+', ranks='+repr(this.ranks)+',name='+repr(this.name)+')'

	def __eq__(this,that):
		r'''
		@section@Miscellaneous@
		Returns \verb|True| when \verb|that| is a \verb|Poset| representing the same poset as \verb|this| and \verb|False| otherwise.
		'''
		if not isinstance(that,Poset): return False
		if set(this.elements)!=set(that.elements): return False
		inds = [that.elements.index(this[i]) for i in range(len(this))]
		return all(this.incMat[i][j] == that.incMat[inds[i]][inds[j]] for i in range(len(this)) for j in range(i+1,len(this)) )

	def __iter__(this):
		r'''
		@section@Miscellaneous@
		Wrapper for \verb|this.elements__iter__|.
		'''
		return this.elements.__iter__()

	def __getitem__(this, i):
		r'''
		@section@Miscellaneous@
		Wrapper for \verb|this.elements.__getitem__|.
		'''
		return this.elements.__getitem__(i)
	def __contains__(this, p):
		r'''
		@section@Miscellaneous@
		Wrapper for \verb|this.elements.__contains__|.
		'''
		return this.elements.__contains__(p)

	def __len__(this):
		r'''
		@section@Miscellaneous@
		Wrapper for \verb|this.elements.__len__|.
		'''
		return this.elements.__len__()

	##############
	#Operations
	##############
	def adjoin_zerohat(this, label=None):
		r'''
		@section@Operations@
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
		r'''
		@section@Operations@
		Returns a new poset with a new maximum adjoined.

		The label default is the same as \verb|Poset.adjoin_zerohat()|
		'''
		return this.dual().adjoin_zerohat().dual()

	def identify(this, X, indices=False):
		r'''
		@section@Operations@
		Returns a new poset after making identifications indicated by \verb|X|.

		The new relation is $p\le q$ when there exists any representatives $p'$ and $q'$ such that $p'\le q'$. The result may not
		truly be a poset as it may not satisfy the antisymmetry axiom ($p<q<p$ implies $p = q$).

		\verb|X| should either be a dictionary where keys are the representatives and the value is a list of elements
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
		r'''
		@section@Operations@
		Returns the dual poset which has the same elements and relation $p\le q$ when $q\le p$ in the original poset.
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
		r'''
		@section@Operations@
		Computes the disjoint union of lists \verb|E| and \verb|F|.

		If \verb|E| and \verb|F| have empty intersection return value is \verb|E+F|
		otherwise return value is ($E\times\{0\})\cup(\{0\}\times F$). This is used by operation methods
		such as \verb|Poset.union| and \verb|Poset.starProduct|.
		'''
		if any([e in F for e in E]+[f in E for f in F]):
			return [(e,0) for e in E] + [(0,f) for f in F]
		else:
			return E+F

	def union(this, that):
		r'''
		@section@Operations@
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
		r'''
		@section@Operations@
		Computes the disjoint union of two posets with maximum and minimums adjoined.
		'''
		this_proper = this.complSubposet(this.max(True)+this.min(True), True)
		that_proper = that.complSubposet(that.max(True)+that.min(True), True)

		return this_proper.union(that_proper).adjoin_zerohat().adjoin_onehat()

	def bddProduct(this, that):
		r'''
		@section@Operations@
		Computes the Cartesian product of two posets with maximum and minimum adjoined.
		'''
		this_proper = this.complSubposet(this.max(True)+this.min(True), True)
		that_proper = that.complSubposet(that.max(True)+that.min(True), True)

		return this_proper.cartesianProduct(that_proper).adjoin_zerohat().adjoin_onehat()

	def starProduct(this, that):
		r'''
		@section@Operations@
		Computes the star product of two posets.

		This is the union of \verb|this| with the maximum removed and \verb|that| with the minimum
		removed and all relations $p < q$ for $p$ in \verb|this| and $q$ in \verb|that|.
		'''
		that_part = that.complSubposet(that.min())
		this_part = this.complSubposet(this.max())
		elements = Poset.element_union(this_part.elements, that_part.elements)
		incMat = [z+[1]*len(that_part.elements) for z in this_part.incMat]+[[-1]*len(this_part.elements)+z for z in that_part.incMat]
		ranks = this_part.ranks + [[r+len(this_part.elements) for r in rk ] for rk in that_part.ranks]

		return Poset(incMat, elements, ranks)

	def cartesianProduct(this, that):
		r'''
		@section@Operations@
		Computes the cartesian product.
		'''
		elements = [(p,q) for p in this.elements for q in that.elements]
		ranks = [[] for i in range(len(this.ranks)+len(that.ranks)-1)]
		for i in range(len(this.elements)):
			for j in range(len(that.elements)):
				ranks[this.rank(i,True)+that.rank(j,True)].append(i*len(that.elements)+j)

		def less(x,y):
			return x!=y and this.lesseq(x[0],y[0]) and that.lesseq(x[1],y[1])

		return Poset(elements=elements, ranks=ranks, less=less)

	def diamondProduct(this, that):
		r'''
		@section@Operations@
		Computes the diamond product which is the Cartesian product of the two posets with their minimums removed and then adjoined with a new minimum.
		'''
		this_proper = this.complSubposet(this.min(True), True)
		that_proper = that.complSubposet(that.min(True), True)
		return this_proper.cartesianProduct(that_proper).adjoin_zerohat()

	def pyr(this):
		r'''
		@section@Operations@
		Computes the pyramid of a poset, that is, the Cartesian product with a length 1 chain.
		'''
		return this.cartesianProduct(Chain(1))

	def prism(this):
		r'''
		@section@Operations@
		Computes the prism of a poset, that is, the diamond product with \verb|Cube(1)|.
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
		r'''
		@section@Queries@
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
		r'''
		@section@Queries@
		Checks whether the given poset is Eulerian (every interval with at least 2 elements has an equal number of odd and even rank elements).
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
		@section@Queries@
		Checks if the poset is Gorenstein*.

		A poset is Gorenstein* if the proper part of all intervals with more than
		two elements have sphere homology. In other words, this function checks that
		\begin{center}
			\verb|this.interval(p,q).properPart().bettiNumbers()|
		\end{center}
		is either \verb|[2]| or \verb|[1,0,...,0,1]| for all $p<=q$ such that
		\verb|this.rank(q) - this.rank(p) >= 2|.
		'''
		if len(this.ranks)==1: return len(this.elements)==1
		if len(this.ranks)==2: return len(this.elements)==2
		def check_homol(P):
			if len(P.elements)==2: return True
			b = P.properPart().bettiNumbers()
			return b==[2] or (b[0] == 1 and b[-1] == 1 and all([x == 0 for x in b[1:-1]]))
		return all([check_homol(this.interval(x,y)) for x in this.elements for y in this.elements if this.less(x,y)])

	@cached_method
	def covers(this, indices=False):
		r'''
		@section@Queries@
		Returns the list of covers of the poset.

		We say $q$ covers $p$ when $p<q$ and $p<r<q$ implies $r=p$ or $r=q$.
		'''
		if this.isRanked():
			ret = {}
			non_max = [i for i in range(len(this)) if i not in this.max(indices=True)]
			for i in non_max:
				p = i if indices else this[i]
				ret[p] = []
				for j in this.ranks[this.rank(p,indices)+1]:
					q = j if indices else this[j]
					if this.less(i,j,indices=True):
						ret[p].append(q)
			return ret
		else:
			if indices:
				P = this.relabel()
				return {i : P.filter((i,),strict=True).min() for i in P if i not in P.max()}
			else:
				return {p : this.filter((p,),strict=True).min() for p in this if p not in this.max(indices=indices)}
	##############
	#End Queries
	##############
	##############
	#Subposet selection
	##############
	@cached_method
	def min(this, indices=False):
		r'''
		@section@Subposet Selection@
		Returns a list of the minimal elements of the poset.
		'''
		return this.dual().max(indices)

	@cached_method
	def max(this, indices=False):
		r'''
		@section@Subposet Selection@
		Returns a list of the maximal elements of the poset.
		'''
		ret_indices = [i for i in range(len(this.incMat)) if 1 not in this.incMat[i]]
		return ret_indices if indices else [this.elements[i] for i in ret_indices]

	def complSubposet(this, S, indices=False):
		r'''
		@section@Subposet Selection@
		Returns the subposet of elements not contained in \verb|S|.
		'''
		if not indices:
			S = [this.elements.index(s) for s in S]
		return this.subposet([i for i in range(len(this.incMat)) if i not in S],True)

	def subposet(this, S, indices=False):
		r'''
		@section@Subposet Selection@
		Returns the subposet of elements in \verb|S|.
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
		r'''
		@section@Subposet Selection@
		Returns the closed interval $[i,j]$.
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
		element_indices = [k for k in range(len(this.elements)) if k in (i,j) or \
		(this.incMat[i][k] == 1 and 1 == this.incMat[k][j]) ]

		return this.subposet(element_indices, indices=True)

	def filter(this, x, indices=False, strict=False):
		r'''
		@section@Subposet Selection@
		Returns the subposet of elements greater than or equal to any element of \verb|x|.

		If \verb|strict| is \verb|True| then \verb|x| is not included in the returned poset and
		if it is \verb|False| \verb|x| is included.
		'''
		if indices: x = [this[i] for i in x]

		comp = this.less if strict else this.lesseq

		ret = this.subposet([p for p in this if any([comp(y,p) for y in x])])
		if indices: ret.elements = [this.elements.index(p) for p in ret]
		return ret

	def ideal(this, x, indices=False, strict=False):
		r'''
		@section@Subposet Selection@
		Returns the subposet of elements less than or equal to any element of \verb|x|.

		Wrapper for \verb|this.dual().filter|.
		'''
		return this.dual().filter(x,indices,strict)

	@cached_method
	def properPart(this):
		r'''
		@section@Subposet Selection@
		Returns the subposet of all elements that are neither maximal nor minimal.
		'''
		P = this
		if len(P.min()) == 1:
			P = P.complSubposet(P.min())
		if len(P.max()) == 1:
			P = P.complSubposet(P.max())
		return P

	def rankSelection(this, S):
		r'''
		@section@Subposet Selection@
		Returns the subposet of elements whose rank is contained in \verb|S|.

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
		r'''
		@section@Internal Computations@
		Returns whether $i$ is strictly less than $j$.
		'''
		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)

		return this.incMat[i][j]==1

	def lesseq(this, i, j, indices=False):
		r'''
		@section@Internal Computations@
		Returns whether $i$ is less than or equal to $j$.
		'''
		return i==j or this.less(i,j, indices)

	def isAntichain(this, A, indices=False):
		r'''
		@section@Internal Computations@
		Returns whether the given set is an antichain ($i\not<j$ for all $i$ and $j$).
		'''
		for i in range(len(A)):
			for j in range(i+1,len(A)):
				if this.lesseq(A[i],A[j],indices) or this.lesseq(A[j],A[i],indices): return False
		return True

	@cached_method
	def join(this, i, j, indices=False):
		r'''
		@section@Internal Computations@
		Computes the join of $i$ and $j$, if it does not exist returns \verb|None|.
		'''
		def _join(i,j,M):
			if i==j: return i
			if M[i][j] == -1: return i
			if M[i][j] == 1: return j
			m = [x for x in range(0,len(M)) if M[i][x] == 1 and M[j][x] == 1]
			for x in range(0,len(m)):
				is_join = True
				for y in range(0,len(m)):
					if x!=y and M[m[x]][m[y]] != 1:
						is_join = False
						break
				if is_join: return m[x]
			return None

		if not indices:
			i = this.elements.index(i)
			j = this.elements.index(j)
			ret = _join(i, j, this.incMat)
			if ret == None: return None
			return this.elements[ret]
		return _join(i, j, this.incMat)

	def mobius(this, i=None, j=None, indices=False):
		r'''
		@section@Internal Computations@
		Computes the value of the M\"obius function from $i$ to $j$.

		If $i$ or $j$ is not provided computes the mobius from the minimum to the maximum
		and throws an exception of type \verb|ValueError| if there is no minimum or maximum.
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
		r'''
		@section@Internal Computations@
		Returns the length of $i$ (the length of the longest chain ending at $i$).

		Returns \verb|None| if \verb|i| is not an element
		(or \verb|i| is not a valid index if \verb|indices| is \verb|True|).
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
	@cached_method
	def flagVectors(this):
		r'''
		@section@Invariants@
		Returns the table of flag $f$- and $h$-vectors as a list with elements \verb|[S, f_S, h_S]|.

		This method should only be used with a poset that has a unique minimum and maximum.
		'''
		def fVectorCalc(ranks,S,M, i, count):
			newCount = count
			if S == []: return 1
			for j in ranks[S[0]]:
				if M[i][j] == 1:
					newCount += fVectorCalc(ranks, S[1:], M, j, count)
			return newCount
		table = [[[],1,1]]

		if len(this.ranks)<=2: return table

		#iterate over all subsets of the ranks
		for i in range(1,1<<(len(this.ranks)-1)-1):
			#construct the corresponding set S
			pad = 1
			elem = 1
			S = []
			while pad <= i:
				if pad&i:
					S.append(elem)

				pad <<= 1
				elem += 1
			table.append([S,fVectorCalc(this.ranks,S,this.incMat,this.ranks[0][0],0),0])


		#do PIE to get the flag h vectors
		for i in range(1,len(table)):
			sign = (2*(len(table[i][0])%2)) - 1 #is -1 if even number of elements, 1 if odd
			for j in range(0,i+1):
				if set(table[j][0]).issubset(table[i][0]):
					table[i][2] += sign*(2*(len(table[j][0])%2)-1)*table[j][1]
		return table


	def flagVectorsLatex(this):
		r'''
		@section@Invariants@
		Returns a string of latex code representing the table of flag vectors of the poset.

		Requires the package longtable to compile.
		'''
		table = this.flagVectors()
		ret = "\\begin{longtable}{c|c|c}\n\t$S$&$f_S$&$h_S$\\\\\n\t\\hline\n\t\\endhead\n"
		for t in table:
			ret += "\t\\{"
			ret += ','.join([str(x) for x in t[0]])
			ret = ret+"\\} & " + str(t[1]) + " & " + str(t[2]) + "\\\\\n\t\\hline\n"
		return ret + "\\end{longtable}"


	@cached_method
	def abIndex(this):
		r'''
		@section@Invariants@
		Returns the \text{ab}-index of the poset.

		If the poset has a unique minimum and maximum but isn't ranked
		this computes the \text{ab}-index considering the poset to be
		quasigraded (in the sense of \url{https://doi.org/10.1016/j.aim.2014.09.008}) with $\overline{\zeta}=\zeta$ and $\rho$ the length function.

		For more information on the \text{ab}-index see: \url{https://arxiv.org/abs/1901.04939}
		'''
		ab = []
		for x in this.flagVectors():
			u = ['a']*(len(this.ranks)-2)
			for s in x[0]: u[s-1] = 'b'
			ab.append([x[2],''.join(u)])

		return Polynomial(ab)

	@cached_method
	def cdIndex(this):
		r'''
		@section@Invariants@
		Returns the \textbf{cd}-index.

		The \textbf{cd}-index is encoded as a \verb|Polynomial|. If the given poset
		does not have a \textbf{cd}-index then a \text{cd}-polynomial is still
		returned, but this is not meaningful. If you wish to check whether a poset has
		a \textbf{cd}-index the Boolean below:
		\begin{center}
			\verb|this.cdIndex().sub('c',Polynomial({'a':1,'b':1})).sub('d',Polynomial({'ab':1,'ba':1})) == this.abIndex()|
		\end{center}

		For more info on the \textbf{cd}-index see \url{https://arxiv.org/abs/1901.04939}
		'''
		n = len(this.ranks)-2
		flag = {tuple():1}
		###########################################################

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

		###########################################################
		def fVectorCalc(ranks,S,M, i, count):
			newCount = count
			if len(S)==0: return 1
			for j in ranks[S[0]]:
				if M[i][j] == 1:
					newCount += fVectorCalc(ranks, S[1:], M, j, count)
			return newCount

		if len(this.ranks)<=2: return table

		n = len(this.ranks)-2
		if n%2==1:
			v = tuple(i for i in range(2,n+1,2))
		else:
			v = tuple(i for i in range(1,n,2))
		for i in range(0,len(v)):
			u = v[i:]
			for S in domIdeal(u,1,True):
				flag[S] = fVectorCalc(this.ranks,S,this.incMat,this.ranks[0][0],0)
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


	@requires(np)
	@cached_method
	def cdIndex_IA(this, v=None):
		r'''
		@section@Invariants@
		Returns the \textbf{cd}-index as calculated via Karu's incidence algebra formula.

		See Proposition 1.2 in \url{https://doi.org/10.1112/S0010437X06001928}

		The argument \verb|v| should be a vector indexed by the poset and by default is the all 1's vector.
		This is the vector the incidence algebra functions are applied to.
		'''
		#find zerohat
		M = this.incMat
		zh = this.min(True)[0]
		alpha=np.array([[(-1)**this.rank(y,True) if x==y or M[x][y]==1 else 0 for y in range(len(M))] for x in range(len(M))])
		deltas=[np.array([[1 if i==j and i in itertools.chain(*this.ranks[:k+1]) else 0 for j in range(len(M))] for i in range(len(M))]) for k in range(len(this.ranks))]
		def make_cd_ops(n,ops=[]):
			ret=[]
			if n>-1:
				C=deltas[n]
				c_ops=[(O,'c'+w) for O,w in ops] #you don't actually have to do the C ops they're already accounted for
				ret+=make_cd_ops(n-1,c_ops)
				if n>0:
					D=np.matmul((-1)**n*alpha-deltas[-1],deltas[n])
					d_ops=[(np.matmul(D,O),'d'+w) for O,w in ops]
					ret+=make_cd_ops(n-2,d_ops)
				return ret
			return ops
		if type(v) == type(None):
			v = np.array([1 for x in M])
		ops=make_cd_ops(len(this.ranks)-3,[(deltas[len(this.ranks)-1],'')])
		ret=[[int(np.matmul(O,v)[zh]),w] for O,w in ops]
		ret = sorted([x for x in ret if x[0]!=0],key=lambda x:x[1])
		return Polynomial(ret)


	@requires(np)
	@cached_method
	def cd_coeff_mat(this, u):
		r'''
		@section@Invariants@
		Returns the matrix for the incidence algebra element giving the \verb|u| coefficient of the \textbf{cd}-index via Karu's formula.
		'''
		#find zerohat
		M = this.incMat
		zh = this.min(True)[0]
		alpha=np.array([[(-1)**this.rank(y,True) if x==y or M[x][y]==1 else 0 for y in range(len(M))] for x in range(len(M))])
		deltas=[np.array([[1 if i==j and i in itertools.chain(*this.ranks[:k+1]) else 0 for j in range(len(M))] for i in range(len(M))]) for k in range(len(this.ranks))]
		u = u.replace('d','_d')
		codeg = len(this.ranks)-2-len(u)
		X = deltas[codeg]
		for i in range(len(u)):
			if u[i] == 'd':
				X = np.matmul(X, np.matmul((-1)**(codeg+i)*alpha-deltas[-1],deltas[codeg+i]))
		return X

	@requires(np)
	def cd_op(this, u, v=None):
		r'''
		@section@Invariants@
		Applies the incidence algebra operation corresponding to the \textbf{cd}-monomial $u$ to the vector $v$ (encoded as a string and a list/tuple respectively).
		'''
		X = this.cd_coeff_mat(u)
		if type(v)==type(None): v = np.array([[1] for p in this])
		else: v = np.array(v)
		X = np.matmul(X, v)
		return {this[i] : X[i][0] for i in range(len(this)) if X[i]!=0}

	def zeta(this):
		r'''
		@section@Invariants@
		Returns the zeta matrix, the matrix whose $i,j$ entry is $1$ if \verb|elements[i] <= elements[j]| and $0$ otherwise.
		'''
		return [[1 if i==j or this.incMat[i][j]==1 else 0 for j in range(len(this.incMat))] for i in range(len(this.incMat))]

	@requires(np)
	@cached_method
	def bettiNumbers(this):
		r'''
		@section@Invariants@
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
	def buildIsomorphism(this, that, indices=False):
		r'''
		@section@Maps@
		Returns an isomorphism from \verb|this| to \verb|that| as a dictionary
		or \verb|None| if the posets are not isomorphic.

		If \verb|indices| is \verb|True| then the dictionary keys and values
		are indices, otherwise they are elements.
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


	def is_isomorphic(this, that):
		r'''
		@section@Maps@
		Returns \verb|True| if the posets are isomorphic and \verb|False| otherwise.
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
		@section@Miscellaneous@
		Hashes the poset, dependent on \verb|this.elements| and relations between
		ranks.
		'''
		X=set()
		for r in range(len(this.ranks)-1):
			for p in this.ranks[r]:
				X.add((r,len([q for q in this.ranks[r+1] if this.incMat[p][q]==1])))
		return hash((tuple(this.elements),frozenset(X)))

	def copy(this):
		r'''
		@section@Miscellaneous@
		Returns a shallow copy of the poset.

		Making a shallow copy via the copy module i.e. \verb|Q = copy.copy(P)|
		doesn't update the self reference in \verb|Q.hasseDiagram| (in this example
		\verb|Q.hasseDiagram.P| is \verb|P|). This doesn't matter if you treat posets as immutable,
		but otherwise could cause issues when displaying or generating hasse diagrams.
		'''
		P = copy.copy(this)
		P.hasseDiagram.P = P
		return P

	def latex(this, **kwargs):
		r'''
		@section@Miscellaneous@
		Returns a string of tikz code to draw the Hasse diagram of the poset for use in a \LaTeX{} document.

		This is a wrapper for \verb|HasseDiagram.latex|.

		For a full list of keyword arguments see \verb|HasseDiagram|. The most common arguments are:
		\begin{itemize}

			\item[]{\verb|height| -- The height in tikz units of the diagram.

				The default value is 30.
				}
			\item[]{\verb|width| -- The width in tikz units of the diagram.

				The default value is 18.
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

				note \verb|H.P| is \verb|this|.
			}
		\end{itemize}
		'''
		return this.hasseDiagram.latex(**kwargs)

	def show(this, **kwargs):
		r'''
		@section@Miscellaneous@
		Opens a window displaying the Hasse diagram of the poset.

		This is a wrapper for \verb|HasseDiagram.tkinter|.

		For a full list of keyword arguments see \verb|HasseDiagram|. The most common arguments are:
		\begin{itemize}
			\item[]{\verb|height| -- The height of the diagram.

				The default value is 30.
			}

			\item[]{\verb|width| -- The width of the diagram.

				The default width is 18.
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
				note \verb|H.P| is \verb|this|.
			}
		\end{itemize}
		'''
		return this.hasseDiagram.tkinter(**kwargs)

	def chains(this, indices=False):
		r'''
		@section@Miscellaneous@
		Returns a list of all nonempty chains of the poset (subsets $p_1<\dots<p_r$).
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
		@section@Invariants@
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
		r'''
		@section@Miscellaneous@
		Returns a list of all pairs $(e,f)$ where $e\le f$.
		'''
		if indices:
			return [(i,j) for j in range(len(this.elements)) for i in range(len(this.elements)) if this.incMat[i][j] == 1]

		return [(this.elements[i],this.elements[j]) for j in range(len(this.elements)) for i in range(len(this.elements)) if this.incMat[i][j]==1]

	def relabel(this, elements=None):
		r'''
		@section@Miscellaneous@
		Returns a new \verb|Poset| object with
		the \verb|elements| attribute as given.

		If \verb|elements| is \verb|None| then
		the returned poset has \verb|elements| attribute set to \verb|list(range(len(this)))|.
		'''
		if elements == None:
			elements = list(range(len(this)))
		ret = Poset(this)
		ret.elements = elements
		return ret

	def reorder(this, perm, indices=False):
		r'''
		@section@Miscellaneous@
		Returns a new \verb|Poset| object (representing the same poset) with the elements reordered.

		\verb|perm| should be a list of elements if \verb|indices| is \verb|False| or a list of indices if \verb|True|. The returned poset has elements in the given order, i.e. \verb|perm[i]| is the $i$th element.
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
		r'''
		@section@Miscellaneous@
		Returns a new \verb|Poset| object (representing the same poset) with the elements sorted.
		'''
		perm = sorted(this.elements, key = key)
		return this.reorder(perm, indices)

	def shuffle(this):
		r'''
		@section@Miscellaneous@
		Returns a new \verb|Poset| object (representing the same poset) with the elements in a random order.
		'''
		perm = list(range(len(this)))
		random.shuffle(perm)
		return this.reorder(perm, True)

	def toSage(this):
		r'''
		@section@Miscellaneous@
		Converts this to an instance of \verb|sage.combinat.posets.posets.FinitePoset|.
		'''
		import sage
		return sage.combinat.posets.posets.Poset((this.elements,[[p,q] for p in this for q in this if this.lesseq(p,q)]), facade=False)
	def fromSage(P):
		r'''
		@section@Miscellaneous@
		Convert an instance of \verb|sage.combinat.posets.poset.FinitePoset| to an instance of \verb|Poset|.
		'''
		rels = [ [x,y] for x,y in P.relations() if x!=y]
		return Poset(relations=rels)
#		return Poset(incMat = P.lequal_matrix(), elements =  P.list())

	def make_ranks(incMat):
		r'''
		@section@Miscellaneous@
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
	def isoClass(this):
		r'''
		@section@Miscellaneous@
		Returns an instance of \verb|PosetIsoClass| representing the isomorphism class
		of the poset.
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
	@is_section@no_children@
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
	'''
	def __eq__(this, that):
		return this.is_isomorphic(that)
	@cached_method
	def __hash__(this):
		X=set()
		for r in range(len(this.ranks)-1):
			for p in this.ranks[r]:
				X.add((r,len([q for q in this.ranks[r+1] if this.incMat[p][q]==1])))
		return hash(frozenset(X))
	def __str__(this):
		return "Isomorphism class of "+Poset.__str__(this)
	def __repr__(this):
		return "PosetIsoClass("+Poset.__repr__(this)[6:]
@decorator.decorator
def return_iso(f, *args, **kwargs):
	r'''
	@section@Utilities@
	Used to define \verb|PosetIsoClass| this decorator causes a function returning
	\verb|ret| to return \verb|ret.isoClass()| when \verb|ret| is an instance of \verb|Poset|.
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
