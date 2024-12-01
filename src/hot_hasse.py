from posets import *

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
		print('splay_step')
		#zig
		if this.p.p is None:
			print('zig')
			this._rotate()

		#zig-zig
		elif (this.p is this.p.p.l and this is this.p.l) or\
		(this.p is this.p.p.r and this is this.p.r):
			print('zig-zig')
			this.p._rotate()
			this._rotate()

		#zig-zag
		#if (this.p is this.p.p.l and this is this.p.r) or\
		#(this.p is this.p.p.r and this is this.p.l):
		else:
			print('zig-zag')
			this._rotate()
			this._rotate()
		return this
	def splay(this):
		'''
		Whole splay operation
		'''
		while this.p is not None:
			print('this.data',this.data)
			print('this.p.data',this.p.data)
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
		yield this
		if this.r is not None: yield from next(this.r)

	def __eq__(this,that):
		print('SplayTree.__eq__({},{})'.format(None if this is None else this.data,None if that is None else that.data))
		if not isinstance(that,SplayTree): return False
		if this.data != that.data or this.size != that.size: return False
		return this.l == that.l and this.r == that.r

	def __neq__(this,that):
		return not this==that

	def __repr__(this):
		return 'SplayTree(data={}, l={}, r={})'.format(this.data,this.l,this.r)

	#Debugging methods
	def traverse(this):
		print('traversing node:',this.data,'left:','' if this.l is None else this.l.data,'right:','' if this.r is None else this.r.data)
		if this.l is not None: this.l.traverse()
		if this.r is not None: this.r.traverse()


##########################################
#Notes on crossing reduction algorithm
##########################################
#Input: two successive alternating layers L,K
#$L=S_{i_0},v_{i_0},S_{i_1},\dots,v_{i_k},S_{i_{k+1}}$
#
#1. append the segment s(v) for p-vertex v in L to the
#container preceding v. Then join this container
#with the succeeding container.
#
#2. Compute measure values for elements in K. First assign
#a position value $p(v)$ to all vertices $v$ in $L$.
#$p(v_{i_0})=\abs{S_{i_0}}$ and $p(v_{i_j})=p(v_{i_{j-1}})$+\abs{S_{i_j}}+1$
#
#3. Calculate an initial ordering on $K$ according to measure in list
#$L^V$. Same for the containers toget $L^S$. Merge the two by
#doing the following:
#\begin{itemize}
#	\item{If $m(L^V_0)\le p(L^S_0)$ then $push(K,pop(L^V))$.}
#	\item{If $m(L^V_0)\ge p(L^S_0) + \abs{L^S_0) - 1$
#		then $push(K,pop(L^S))$.}
#	item{otherwise $k=ceil(m(pop(L^V))-pos(S)$ $T,R =split(S,k)$
#		$push(K,T)$ 
