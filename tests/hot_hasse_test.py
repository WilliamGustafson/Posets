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
		assert(list(t)==[10,15,20,30,40,50,60,70,80,90,100])

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
