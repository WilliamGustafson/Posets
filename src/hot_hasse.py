from posets import *

class SplayTree:

	def __init__(this, data=None, l=None, r=None, p=None):
		this.data = data
		this.l = l
		if this.l is not None: this.l.p = this
		this.r = r
		if this.r is not None: this.r.p = this
		this.p = p

	def rotate(this):
		'''
		Safe fronted for \verb|_rotate|.
		'''
		if this.p is None: return this
		return this._rotate()
#	def _rotate(this):
#		'''
#		Unsafe rotate
#		'''
#		#swap data between this and parent
#		tmp = this.data
#		this.data = this.p.data
#		this.p.data = tmp
#
#		#swap left and right subtrees on parent
#		tmp = this.p.l
#		this.p.l = this.p.r
#		this.p.r = tmp
#
#		return this.p

	def _rotate(this):
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
		return this
	def a_rotate(this):
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
		#
		#
		#where x is this when this.p.l is this and y is this.p
		#and y is this when this.p.r is this and x is this.p
		#also 
		p = this.p
		pp = this.p.p
		pl = p.l
#		pr = p.r
		l = this.l
		r = this.r

		#right rotation
		if pl is this:
			#l r pr -> l pl pr
			if r is not None: r.p = p
			p.l = r
			p.p = this
			this.r = p
		#left rotation
		else:
			#pl l r -> pl pr r
			if l is not None: l.p = p
			p.r = l
			p.p = this
			this.l = p
		this.p = pp
		if pp is not None:
			if pp.l is p: pp.l = this
			else: pp.l = this
		return this
#		print('rotating',this.data)
#		p = this.p
#		pp = this.p.p
#		pl = p.l
#		pr = p.r
#		l = this.l
#		r = this.r
#		#set this into parent's position
#		this.p = pp
#		if pp is not None:
#			if pp.l is p: pp.l = this
#			else: pp.r = this
#
#		#set parent into this old position
#		if pl is this:
#			this.r = p
#			this.l = pr
#			if pr is not None: pr.p = this
#		else:
#			this.l = p
#			this.r = pl
#			if pl is not None: pl.p = this
#		p.p = this
#		#parent gets subtrees but swapped
#		p.l = r
#		if l is not None: l.p = p
#		p.r = l
#		if r is not None: r.p = p
#
#		return this

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
#			input()

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
		s.splay()
		return s

	def join(this, that):
		t = this
		while t.r is not None: t = t.r
		s = this.get(t.data)
		this.r = s.r
		return this

	def split(this, data):
		t = this.get(data)
		return (t.l,t.r)

	def remove(this, data):
		t = this.search(data)
		s = t.r if t.l is None else t.l.join(t.r)
		if t.p is not None:
			if t.p.l is t:
				t.p.l = s
			else:
				t.p.r = s
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
		print('SplayTree.__eq__({},{})'.format(None if this is None else this.data,None if that is None else that.data))
		if not isinstance(that,SplayTree): return False
		if this.data != that.data: return False
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
