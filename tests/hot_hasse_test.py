import sys
sys.path.append('../src/')
from hot_hasse import *
import pytest

def T():
	return SplayTree(
		data=50,
		l=SplayTree(
			data=30,
			r=SplayTree(data=40),
			l=SplayTree(
				data=10,
				r=SplayTree(
					data=20,
					l=SplayTree(data=15)
					)
				)
			),
		r=SplayTree(
			data=60,
			r=SplayTree(
				data=90,
				r=SplayTree(
					data=100
					),
				l=SplayTree(
					data=70,
					r=SplayTree(data=80)
					)
				)
			)
		)
class TestSplayTree:
	def show_data(this):
		'''
		Plot tree as a poset to make sure we copied it right, looks good.
		'''
		from posets import Poset
		def add_children(t,rels):
			rels[t.data]=[]
			if t.l is not None:
				rels[t.data].append(t.l.data)
				add_children(t.l,rels)
			if t.r is not None:
				rels[t.data].append(t.r.data)
				add_children(t.r,rels)
		rels={}
		t=T()
		add_children(t,rels)
		Poset(relations=rels).dual().show()

	def test_search(this):
		t=T()
		assert(t.search(30).data == 30)
		assert(t.search(10).data == 10)
		assert(t.search(70)==t.r.r.l)
		assert(t.search(12)==t.l.l.r.l)

	def test_iter(this):
		t=T()
		assert([x for x in t]==[10,15,20,30,40,50,60,70,80,90,100])

	def test_rotate(this):
		t=T()
		t_rot = SplayTree(
		data=50,
		l=SplayTree(
			data=10,
			r=SplayTree(
				data=30,
				l=SplayTree(
					data=20,
					l=SplayTree(data=15)
					),
				r=SplayTree(data=40)
				)
			),
		r=SplayTree(
			data=60,
			r=SplayTree(
				data=90,
				r=SplayTree(
					data=100
					),
				l=SplayTree(
					data=70,
					r=SplayTree(data=80)
					)
				)
			)
		)
		t.search(10).rotate()
		assert(t_rot == t)

	def test_rotate60(this):
		t=T()
		t_rot = SplayTree(
			data=60,
			l=SplayTree(
				data=50,
				l=SplayTree(
					data=30,
					l=SplayTree(
						data=10,
						r=SplayTree(
							data=20,
							l=SplayTree(
								data=15
								)
							)
						),
					r=SplayTree(data=40)
					),
				),
			r=SplayTree(
				data=90,
				l=SplayTree(
					data=70,
					r=SplayTree(data=80)
					),
				r=SplayTree(data=100)
				)
			)
		s=t.search(60).rotate()
		assert(s.data == 60)
		assert(s == t_rot)
		assert(t.data == 50)
		assert(t == t_rot.search(50))

	def test_rotate_leaf(this):
		assert(
			SplayTree(data=0,r=SplayTree(data=1)).search(1).rotate().root()
			==
			SplayTree(data=1,l=SplayTree(data=0))
		)

	def test_splay_step(this):
		t_step = SplayTree(
			data=50,
			l=SplayTree(
				data=30,
				l=SplayTree(
					data=10,
					r=SplayTree(
						data=20,
						l=SplayTree(data=15)
						)
					),
				r=SplayTree(data=40)
				),
			r=SplayTree(
				data=60,
				r=SplayTree(
					data=80,
					l=SplayTree(data=70),
					r=SplayTree(data=90,
						r=SplayTree(data=100)
						)
					)
				)
			)
		t=T()
		assert(t.search(80).splay_step().root() == t_step)

	def test_splay(this):
		t=T()
		t_splay = SplayTree(
			data=80,
			r=SplayTree(
				data=90,
				r=SplayTree(data=100)
				),
			l=SplayTree(data=60,
				r=SplayTree(data=70),
				l=SplayTree(
					data=50,
					l=SplayTree(
						data=30,
						r=SplayTree(data=40),
						l=SplayTree(
							data=10,
							r=SplayTree(
								data=20,
								l=SplayTree(data=15)
								)
							)
						)
					)
				)
			)
		t.search(80).splay()
		t.traverse()
		assert(t.root() == t_splay)

	def test_get(this):
		t_splay = SplayTree(
			data=80,
			r=SplayTree(
				data=90,
				r=SplayTree(data=100)
				),
			l=SplayTree(data=60,
				r=SplayTree(data=70),
				l=SplayTree(
					data=50,
					l=SplayTree(
						data=30,
						r=SplayTree(data=40),
						l=SplayTree(
							data=10,
							r=SplayTree(
								data=20,
								l=SplayTree(data=15)
								)
							)
						)
					)
				)
			)
		t=T()
		assert(80 == t.get(80).data)
		assert(t.root() == t_splay)

	def test_add(this):
		t = SplayTree(
			data=3,
			l=SplayTree(
				data=1,
				r=SplayTree(data=2)
				),
			r=SplayTree(data=5)
			)
		s = SplayTree(
			data=3,
			l=SplayTree(
				data=1,
				r=SplayTree(data=2)
				),
			r=SplayTree(
				data=5,
				l=SplayTree(4)
				)
			)
		assert(t.add(4).root() == s.search(4).splay().root())
	
	def test_contains(this):
		s = SplayTree(
			data=3,
			l=SplayTree(
				data=1,
				r=SplayTree(data=2)
				),
			r=SplayTree(
				data=5,
				l=SplayTree(4)
				)
			)
		T = s.search(4).splay().root()
		assert(T.get(2).data == 2)
		assert(T.root().get(1).data == 1)
		assert(T.root().get(2).data == 2)

	def test_join(this):
		s = SplayTree(
			data=3,
			l=SplayTree(
				data=1,
				r=SplayTree(data=2)
				),
			r=SplayTree(data=4)
			)
		t2 = SplayTree(
			data=8,
			l=SplayTree(
				data=6,
				r=SplayTree(data=7)
				),
			r=SplayTree(
				data=10,
				l=SplayTree(data=9)
				)
			)
class TestCrossReduction:
	def make_Q():
		#Example from ``Simple and Efficient Bilary Cross Counting''
		return Poset(relations={'s0':['n0','n2','n3'],'s1':['n1'],'s2':['n1','n3','n5'],'s3':['n2','n4'],'s4':['n2','n5']}).reorder(perm=[f's{i}' for i in range(5)]+[f'n{i}' for i in range(6)])
	def make_P():
	    #3     7     11
	    #|\    |    /|
	    #2 \   6   / 10
	    #|  \  |  /  |
	    #1   \ 5 /   9
	    #|    \|/    |
	    #0     4     8
		C = Chain(3)
		D = Chain(3)
		E = Chain(3)
		D.elements = list(range(max(C)+1,max(C)+1+len(D)))
		E.elements = list(range(max(D)+1,max(D)+1+len(E)))
		P = C.union(D).union(E)
		P.incMat[len(C)][len(C)-1] = 1
		P.incMat[len(C)-1][len(C)-1] = -1
		P.incMat[len(C)][-1] = 1
		P.incMat[-1][len(C)] = -1
		return P
	def make_layers(P):
		covers = P.covers(True)
		long_edges = [[(x,y) for x in covers if P.rank(x,True)<i for y in covers[x] if P.rank(y,True)>i] for i in range(len(P.ranks))]
	def test_rk_to_layer(this):
		P = TestCrossReduction.make_P()
		covers = P.covers(True)
		long_edges = [[(x,y) for x in covers if P.rank(x,True)<i for y in covers[x] if P.rank(y,True)>i] for i in range(len(P.ranks))]
		T = SplayTree(Segment(4,3))
		T.add(Segment(4,11))
		print(rk_to_layer(P.ranks[1]+long_edges[1]))
		print([Vertex(1),Vertex(5),Vertex(9),T])
#		assert(rk_to_layer(P.ranks[1]+long_edges[1],long_edges[1]) == [Vertex(1),Vertex(5),Vertex(9),T])
	def test_cross_count_layer(this):
		P = TestCrossReduction.make_P()
		covers = P.covers(True)
		long_edges = [[(x,y) for x in covers if P.rank(x,True)<i for y in covers[x] if P.rank(y,True)>i] for i in range(len(P.ranks))]
		L = rk_to_layer(P.ranks[1]+long_edges[1])
		K = rk_to_layer(P.ranks[2]+long_edges[2])
		edges = [(x,y) for x in P.ranks[1] for y in P.ranks[2] if y in P.covers(True)[x]] + long_edges[1]
		edges = []
		for l in L:
			if type(l) is Vertex:
				l_covers = P.covers(True)[l.id]
				edges+=[(l,k) for k in K if type(k) is Vertex and k.id in l_covers]
			else:
				edges.append((l,l))
		Q=TestCrossReduction.make_Q()
		assert(cross_count_layer(rk_to_layer(Q.ranks[0]),rk_to_layer(Q.ranks[1]),((Vertex(x),Vertex(y)) for x in Q.ranks[0] for y in Q.covers(True)[x]))==12)
		B=Butterfly(2)
		assert(cross_count_layer(rk_to_layer(B.ranks[1]),rk_to_layer(B.ranks[2]),((Vertex(x),Vertex(y)) for x in B.ranks[1] for y in B.ranks[2]))==1)
	def test_cross_reduction(this):
		B = Bruhat(3,weak=True).reorder([0,1,2,4,3,5],indices=True)
	def test_cross_count(this):
#		assert(1 == cross_count(Bruhat(3)))
#		assert(0 == cross_count(Bruhat(3,weak=True)))
#		assert(2 == cross_count(Boolean(3)))
#		assert(2 == cross_count(Butterfly(3)))
		P = TestCrossReduction.make_P()
		P.incMat[0][7] = 1
		P.incMat[8][7] = 1
		P = Poset(incMat=P.incMat,elements=P.elements)
		assert(2 == cross_count(P))
TestCrossReduction().test_cross_count()
