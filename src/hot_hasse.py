from posets import *
import copy

##########################################
#Notes and TODOS and Whatnot
##########################################
#Having layers be alternating is annoying,
#it appears the only necessary property (more
#like desirable it makes things smaller) is
#that each run of segments is maximal (left and
#right of a container of segments is a vertex or
#the end of the layer list). With this change you
#have to make a tweak to the step where you
#merge p-vertices into segment containers; the only
#change is that if p has a vertex to the left or to
#the right but not both you merge p into the container
#to its right or left resp. and if both adjacent elements
#to p are vertices you put p into a SplayTree by itself.
#I don't think any other changes are needed so that is what
#I will do. I'm not going back and fixing all the notes about
#alternating layers right now but that is TODO.
#
#SplayTree takes linearly ordered obejcts for data, TODO
#make it takes hashable objects instead and makes Segments
#hashable also vertices
#Ensure that hash method is compatible with equals which sometimes
#ignores Segment.
#
#Add references to each function's doc string
#
#fix all liar comments pay particular attention
#to anything about (alternating) layers
#
#write tests for the cross reduction inside out (Start with cross counting use example from paper)
#when doing cross step testing use the example from the paper too
##########################################

class SplayTree:

	def __init__(this, data=None, l=None, r=None, p=None):
		this.data = data
		this.l = l
		if this.l is not None: this.l.p = this
		this.r = r
		if this.r is not None: this.r.p = this
		this.p = p
		this._set_size()

	def _set_size(this):
		this.size = 1 + (0 if this.l is None else this.l.size) + (0 if this.r is None else this.r.size)

	def rotate(this):
		'''
		Safe fronted for \verb|_rotate|.
		'''
		if this.p is None: return this
		return this._rotate()

	def _rotate(this):
		'''
		Rotates a binary tree at the edge from the given node to its parent.

		This method assumes the given node is not the root or the child of the root.
		If this is not the case an \verb|AttributeError| exception will be thrown
		(trying to access an attribute on \verb|None|).
		The tree is altered in place and the original node, now in a new position,
		is returned.

		In detail, the given node is placed where its parent originally was and the
		former parent placed where the given node originally was in the tree.
		Also, the two original branch of the given node are swapped (left for right
		and vice versa).

		If editing a tree you may need to call \verb|root| in order to still have the full
		tree. If the root changes your tree object will be a proper subtree.
		'''
		##############################
		#Here's what rotating does (all unpictured nodes and
		#edges are unaffected)
		#this is the middle node (x on left y on right)
		##############################
		#      y            x
		#     /              \
		#    x    <------>    y
		#     \              /
		#      z            z
		##############################
		pp = this.p.p
		p = this.p
		if p.l is this:
			r = this.r

			this.r = p
			p.p = this

			p.l = r
			if r is not None: r.p = p

		if p.r is this:
			l = this.l

			this.l = p
			p.p = this

			p.r = l
			if l is not None: l.p = p

		this.p = pp
		if pp is not None:
			if pp.l is p: pp.l = this
			else: pp.r = this
		p._set_size()
		this._set_size()
		if pp is not None: pp._set_size()
		return this

	def splay_step(this):
		'''
		Single splay step
		'''
		if this.p is None: return
		#zig
		if this.p.p is None:
			this._rotate()

		#zig-zig
		elif (this.p is this.p.p.l and this is this.p.l) or\
		(this.p is this.p.p.r and this is this.p.r):
			this.p._rotate()
			this._rotate()

		#zig-zag
		#if (this.p is this.p.p.l and this is this.p.r) or\
		#(this.p is this.p.p.r and this is this.p.l):
		else:
			this._rotate()
			this._rotate()
		return this
	def splay(this):
		'''
		Whole splay operation
		'''
		while this.p is not None:
			this.splay_step()
		return this

	def search(this, data):
		'''
		Standard binary search (no splaying)
		'''
		if this.data == data: return this
		if data < this.data:
			if this.l is None: return this
			return this.l.search(data)
		else:
			if this.r is None: return this
			return this.r.search(data)

	def get(this, data):
		if data == this.data:
			this.splay()
			return this
		if data < this.data:
			if this.l is None:
				this.splay()
				return this
			return this.l.get(data)
		else: #data > this.data
			if this.r is None:
				this.splay()
				return this
			return this.r.get(data)

	def add(this, data):
		t = this.search(data)
		if t.data == data: s = t
		else:
			s = SplayTree(data=data,p=t)
			if data<t.data: t.l = s
			if data>t.data: t.r = s
			t.size+=1
		return s.splay()

	def join(this, that):
		t = this.root()
		s = this.get(t.data)
		this.r = s.r
		this.size += s.size
		return this

	def split(this, data):
		t = this.get(data)
		return (t.l,t.r)

	def remove(this, data):
		t = this.search(data)
		s = t.r if t.l is None else t.l.join(t.r)
		s.p = t.p
		if t.p is not None:
			if t.p.l is t:
				t.p.l = s
			else:
				t.p.r = s
			t.p.size -= 1
			return t.p.splay()
		return s

	def root(this):
		'''
		Returns the root node.
		'''
		while this.p is not None: this = this.p
		return this

	def __iter__(this):
		return this.__next__()

	def __next__(this):
		if this.l is not None: yield from next(this.l)
		yield this.data
		if this.r is not None: yield from next(this.r)

	def __eq__(this,that):
#		Assumes that==that.root() and compares this as a subtree to that
		if not isinstance(that,SplayTree): return False
		if len(this)!=len(that): return False
		return all(x in that.root() for x in this)

	def __neq__(this,that):
		return not this==that

	def __repr__(this):
		return 'SplayTree(data={}, l={}, r={})'.format(repr(this.data),this.l,this.r)

	def __len__(this):
		return this.size
	def __contains__(this,key):
		return key == this.get(key).data


	#Debugging methods
	def traverse(this):
		print('traversing node:',this.data,'left:','' if this.l is None else this.l.data,'right:','' if this.r is None else this.r.data)
		if this.l is not None: this.l.traverse()
		if this.r is not None: this.r.traverse()

class Vertex:
	def __init__(this,id,p=False,q=False,S=None):
		this.id = id
		this.p = p
		this.q = q
		this.S = S
		assert(not (this.p and this.q))
		assert((this.S is None) == (not this.p and not this.q))
	def __str__(this):
		return ('p-vertex' if this.p else 'q-vertex' if this.q else 'vertex') + ' ' + str(this.id)
	def __eq__(this,that):
		return isinstance(that,Vertex) and this.p==that.p and this.q==that.q and this.id==that.id
	def __repr__(this):
		return f'Vertex({this.id},p={this.p},q={this.q})'
	def __lt__(this,that):
		return this.id < that.id if type(that) is Vertex else this.id <= that.p.id if type(that) is Segment else NotImplemented
	def __gt__(this,that):
		return this.id > that.id if type(that) is Vertex else this.id > that.p.id if type(that) is Segment else NotImplemented
	def __hash__(this):
		return this.id
class Segment:
	def __init__(this,p=None,q=None):
		this.p = Vertex(id=p,p=True,q=False,S=this)
		this.q = Vertex(id=q,p=False,q=True,S=this)
	def __str__(this):
		return f'segment from {this.p.id} to {this.q.id}'
	def __eq__(this,that):
		return isinstance(that,Segment) and this.p==that.p and this.q==that.q
	def __repr__(this):
		return f'Segment(p={repr(this.p)},q={repr(this.q)})'
	def __lt__(this,that):
		return (this.p.id <= that.p.id and this.q.id < that.q.id) if type(that) is Segment else this.p.id < that.id if type(that) is Vertex else NotImplemented
	def __gt__(this,that):
		return (this.p.id >= that.p.id and this.q.id > that.q.id) if type(that) is Segment else this.p.id >= that.id if type(that) is Vertex else NotImplemented
	def __hash__(this):
		return hash((this.p.id,this.q.id))

def cross_sort(P,ranks=None,agg=np.mean):
	'''
	Given a poset reorder ranks to reduce crossings in the Hasse diagram.

	If provided \verb|ranks| is used as an initial ordering, otherwise the
	linear extension of \verb|P| is used (the linear ordering \verb|P.elements|).

	If provided \verb|agg| is passed to \verb|cross_reduction|.
	'''
	#TODO: rk_to_layer currently expects a rank list containing integers and tuples
	#and converts these to Vertex and Segment objects, but cross_reduction return a list
	#containing Vertex and Segment objects. Decide where to do the conversion and how
	#to handle the inconsistencies (converting at top level sounds most appropriate?)
	Q = P.dual()
	ranks = copy.deepcopy(P.ranks) if ranks is None else copy.deepcopy(ranks)
	long_edges = [[(x,y) for x in covers if P.rank(x,True)<i for y in covers[x] if P.rank(y,True)>i] for i in range(len(P.ranks))]
	ranks = [rank_to_layer(rk,long_edges) for rk in ranks]
	crosses = cross_count(P,ranks)

	old_crosses = 0
	while old_crosses != crosses:
		old_crosses = crosses
		#upwards pass
		for L,K,i in zip(ranks[:-1],ranks[1:],range(1,len(L))):
			ranks[i] = cross_reduction(P,L,K,agg)
		#downwards pass
		for L,K,i in zip(ranks[-2::-1],ranks[-1:0:-1],range(len(L)-2,-1,-1)):
			ranks[i] = cross_reduction(Q,K,L,agg)
			#TODO I'm pretty sure on the downward pass notions of p and q swap
			#maybe add a dual method to vertices and Segments or a dual flag in cross_reduction
		crosses = cross_count(P,ranks)
	perm = itertools.chain(([v.id for v in L if isinstance(v,Vertex)] for L in ranks))
	P.reorder(perm=perm,indices=True)
	return P

def rk_to_layer(L):
	'''
	Given a rank, meaning a list of indices to elements and pairs of indices indicating segments, return a layer (list of indices and \verb|SplayTree| instances replacing the runs of pairs/$p$-vertices.
	'''
	long_edges=[x for x in L if type(x) is tuple]
	ret = []
	T = None
	for x in L:
		if type(x) is int:
				if T is not None:
					ret.append(T.root())
					T = None
				p = x in [e[0] for e in long_edges]
				q = x in [e[1] for e in long_edges]
				S = Segment(*next(e for e in long_edges if e[0]==x)) if p else Segment(*next(e for e in long_edges if e[1] == x)) if q else None
				ret.append(Vertex(x,p=p,q=q,S=S))
		
		if type(x) is tuple:
			if T is None:
				T = SplayTree(data=Segment(p=x[0],q=x[1]))
			else:
				T.add(Segment(p=x[0],q=x[1]))
	if T is not None: ret.append(T.root())
	return ret

def rk_to_next_layer(L, long_edges):
	'''
	Given a rank, meaning a list of indices to elements and pairs of indices indicating segments, return a layer (list of indices and \verb|SplayTree| instances replacing the runs of pairs/$p$-vertices.
	'''
	ret = []
	T = None
	for x in L:
		if type(x) is int:
			p = x in [e[0] for e in long_edges]
			if p:
				S = Segment(*next(e for e in long_edges if e[0]==x))
				if T is None:
					T = SplayTree(data=S)
				else:
					T.add(S)
			else: #not p
				if T is not None:
					ret.append(T)
					T = None
				q = x in [e[1] for e in long_edges]
				S = Segment(*next(e for e in long_edges if e[1] == x)) if q else None
				ret.append(Vertex(x,p=p,q=q,S=S))

		if type(x) is tuple:
			if T is None:
				T = SplayTree(data=x)
			else:
				T.add(x)
	if T is not None: ret.append(T)
	return ret

def cross_count(P,layers):
	'''
	Given a poset and a list of layers returns the number of crossings in the Hasse diagram of the poset.

	A layer encodes the vertices and segments along a rank and consists of \verb|Vertex| and \verb|Segment| objects.
	'''
	raise NotImplementedError
def cross_count_layer(L,K,edges):
	'''
	Given two adjacent ranks of a poset count the number of crossings in the Hasse diagram between those two ranks.

	\verb|covers| is a list of tuples \verb|(l,k)| where either these elements are vertices with a cover between them or are the two eneds of a long edge. This list should be in lexicographic order as induced by the desired order on the ranks in the poset.

	\verb|long_edges| is a list of the covers that cross gap between \verb|L| and \verb|K| but have a vertex in neither.
	'''
	#Accumulator tree algorithm works like this:
		#Assume $\abs{L}\ge\abs{K}$.
		#$\pi$ should be the graph as a sequence, $\pi_i$ is the element in $K$ adjacent to the $i$th edge
		#when the edges are sorted lexicographically (considered as elements of $L\times K$).
		#Let n be the length of the smaller of the two layers
		#Let $q=2^m$ be the smallest power of two greater than or equal to 2n
		#We have a fully balanced binary tree T with q-1 entries which we view
		#as a poset (we can label the poset by binary sequences of length between 0 and $m-1$
		#and say $x<y$ when $y$ is a prefix of $x$).
		#
		#Now for the actual accumulation we proceed as follows. For each element $\pi_i$ we add one to
		#the corresponding leaf $x$ and to each element $y>x$. Do this in order of the tree and as you
		#go up increment a cross count by the value on the sibling for each left node encountered.
		#
		#Binary tree is stored as a list, in terms of the tree the entries go left to
		#right across ranks:
		#						0
		#				1				2
		#			3		4		5		6
		#a child is left iff odd and the parent of i is (i//2)+(i%2)-1
		#the children of i are 2*i+1 and 2*i+2
		#the sibling of i is i+2*(i%2)-1
		#leaves are the last q/2 (q-1 is the size of the tree) elements of the list
	#turn ranks into layers
	#L = rk_to_layer(L,long_edges)
	#K = rk_to_layer(K,long_edges)
	edges = list(edges)
	swapped = len(L)>len(K)	
	if swapped: 
		L,K = K,L
		edges = [e[::-1] for e in edges]
#	long_edges = [e for e in edges if type(e) is tuple]
#This line is wrong. Need to go over all covers and long edges and grab indices in the smaller of the two ranks (which is L regardless of \verb|swapped| because we swapped to make that happen)
	pi = [j for i,j in sorted([(K.index(y),L.index(x)) for x,y in edges])]
	del swapped
	n = len(L)
#	q is a power of 2, q-1 is the size of the binary tree, q/2 must be at least $\abs{L}$
#	so that $L$ can inject into the leaves
#	$2^{m-1}<\abs{L}\le 2^m = q$
	q = 1
	while q<=(n<<1):
		q<<=1
	T = [0 for _ in range(q-1)]
	count = 0 #cross count to be accumulated
	for x in pi:
		i=q//2+x-1 #index to node
		#corresponding leaf is in same index, increment it
		increment = 1 if type(L[x]) is Vertex else len(L[x])
		#go up tree and increment, also, for each left node add sibling's value to cross count
		while i>=0:
			T[i]+=increment
			if i%2==1: #if left add to count
				count+=T[i+1]
			i = i//2 + i%2 - 1 #get next node
	return count

def cross_reduction(P,L,K,agg=np.mean):
	'''
	Given two rank lists \verb|L| and \verb|K| computes a new
	ordering for \verb|K| to reduce crossings.

	Elements of \verb|L| and \verb|K| should be either indices to poset
	elements indicating vertices or tuples of two indices indicating a segment
	(the first is the $p$-vertex the second the $q$-vertex).
	'''
	#append segment S(p) for each p-vertex in L to container preceding p,
	#then join the container with the next one and remove p.
	#L = rk_to_next_layer(L) #L is now basically a hybrid of K as a layer and L as a layer (verts from L containers from K)
	#K = rk_to_layer(K)

	#for each p vertex in L replace with a segment and merge any adjacent containers
	new_L = []
	T = None
	for x in L:
		if type(x) is Vertex and x.p:
			if T is None: T = SplayTree(x.S)
			else: T.add(x)
		elif type(x) is Vertex:
			if T is not None: new_L.append(T)
			new_L.append(x)
		else:
			if T is not None: T.add(x)
			else: T = SplayTree(x)
	if T is not None: new_L.append(T)
	L = new_L


#set position values for L:
#		pos(v_{i_0}) = len(S_{i_0})
#		pos(v_{i_j}) = pos(v_{i_{j-1}}) + len(S_{i_j}) + 1
#		pos(S_{i_j}) = pos(v_{i_j-1})+1 (if len(S_{i_j})>0)
#		pos(S_{i_0}) = 0 if nonempty
	pos = {}
	last = -1
	for i in range(1,len(L)):
		if type(L[i]) is SplayTree:
			pos[L[i]] = last+1
		if type(L[i]) is Vertex:
			pos[L[i]] = (len(L[i-1]) if type(L[i-1]) is SplayTree else 0) + last + 1
			last = pos[L[i]]

	#aggregate position values over covered elements to get a measure on K
	#below we use SplayTree's as dictionary keys so we need a hash TODO can we get around that?
	SplayTree.__hash__ = lambda this: hash(frozenset(this))
	
	meas = {}
	for k in K:
		if type(k) is SplayTree:
			meas[k] = pos[k]
		else: #type(k) is Vertex:
			if k.q: continue
			covers = P.covers(True)[k.id]
			meas[k] = agg([Vertex(x) for x in covers])
	#THE BELOW COMMENTS ARE A LIE: TODO PUT SOMETHING TRUTHFUL HERE
	#sort all non-q vertices and all segment containers in K
	#then merge the two in the following way:
	# - If $m(K^V_0)\le p(K^S_0)$ then $push(K,pop(K^V))$.}
	# - If $m(K^V_0)\ge p(K^S_0) + \abs{K^S_0} - 1$
	#	then $push(K,pop(K^S))$.
	# - otherwise $k=ceil(m(pop(K^V))-pos(S)$ $T,R =split(S,k)$
	#	$push(K,T)$
	KV = sorted(((k,v) for k,v in meas.items() if type(k) is Vertex and not k.q),key = lambda x:x[1])
	KS = sorted(((k,v) for k,v in meas.items() if type(k) is SplayTree),key = lambda x:x[1])
	ret = []
	while len(KV)>0 and len(KS)>0:
		if meas[KV[0]] <= pos[KS[0]]:
			ret.append(KV.pop())
		elif measK[v[0]] >= pos[KS[0]] + len(KS[0]) - 1:
			ret.append(KS.pop())
		else:
			S = KS.pop()
			v = KV.pop()
			k = math.ceil(meas[v] - pos[S])
			T,R = S.split(k)
			ret.append(T)
			ret.append(v)
			pos[R] = pos[S] + k
			KS.append(R)
			k = math.ceil(meas(KV.pop()))-pos[S]

	#place q-vertices ``according to the position of their segment''
	#``Note that the representation of the layer $L_i$ is lost, since the containers
	#are reused for the layer $L_{i+1}$.'' I don't see this pointed out clearly in any
	#way whatsoever, but these two quotes together suggest to me that what is meant
	#is that when the new order on K is constructed we append the containers from L
	#and don't use the containers from K. This works out because the containers in the two
	#are very similar, the only difference is some segments in L turn in to q-vertices
	#in K and this step is where we fix that and also get the q-vertices in the right position
	#by simply turning segments into q-vertices where needed by calling S.split(q) for
	#each q-vertex in K where S is the container containing it at the time.
	new_ret = []
	for T in ret:
		if type(T) is not SplayTree:
			new_ret.append(T)
			continue
		for x in T:
			qverts = sorted([v for v in T if type(v) is Vertex and v.q],key = lambda v:hash(v))
			new_Ts = []
			for q in qverts:
				S,T = T.split(q)
				if len(S)>0: new_Ts.append(S)
				new_Ts.append(q)
			if len(T)>0: new_Ts.append(T)
		new_ret+=new_Ts
	return new_ret
	
	#ensure K is an alternating layer by merging consecutive containers and inserting empty
	#containers in between vertices
	#TODO names
	new_new_ret = []
	T = None
	for x in new_ret:
		if type(x) is SplayTree:
			if T is None:
				T = x
			else:
				T = T.join(x)
		else:
			if T is not None:
				new_new_ret.append(T)
				T = None
			new_new_ret.append(x)
	if T is not None: new_new_ret.append(T)
	return list(itertools.chain(*((v.id,) if type(v) is Vertex else (x.id for x in v) for v in new_new_ret)))
