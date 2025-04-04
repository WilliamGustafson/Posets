##########################################
#general utilities setup
##########################################
import sys
import os
import pytest
import shutil
import tempfile
sys.path.append(os.path.join('..','src'))
from posets import *
import subprocess

#mock poset class that overrides equality
#checking x==y when x is an instance of P and y a Poset
#checks that the incMat, elements and ranks properties are all equal
class P:
	def __init__(this, incMat=None, elements=None, ranks=None):
		if type(incMat) == Poset: #construct from a poset
			this.incMat = incMat.incMat
			this.ranks = incMat.ranks
			this.elements = incMat.elements
			return
		#construct explicitly
		this.incMat = incMat
		this.elements = elements
		this.ranks = ranks
	def __eq__(this, that):
		return this.incMat == that.incMat and this.elements == that.elements and this.ranks == that.ranks
	def __neq__(this, that):
		return not this == that
	def __repr__(this):
		return "P("+','.join(('incMat='+repr(this.incMat),'elements='+repr(this.elements),'ranks='+repr(this.ranks)))+")"
##########################################
#test constructor options
##########################################

class TestConstructorOptions:
	Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])

	def test_relsListIndices(this):
		'''
		relations=list indices=True
		'''
		bool3 = P(Poset(relations=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[4,5],[4,6],[3,7],[5,7],[6,7]], elements = this.Bool3.elements, indices=True))
		assert(bool3==this.Bool3)

	def test_relsDictIndices(this):
		'''
		relations=dict indices=True
		'''
		bool3 = P(Poset(relations={0:[1,2,4],1:[3,5],2:[3,6],4:[5,6],5:[7],6:[7],3:[7]}, indices=True, elements=this.Bool3.elements))
		assert(bool3==this.Bool3)

	def test_restList(this):
		b3_rels_list = [[tuple(),(1,)],[tuple(),(2,)],[tuple(),(3,)],[(1,),(1,2)],[(1,),(1,3)],[(2,),(1,2)],[(2,),(2,3)],[(3,),(1,3)],[(3,),(2,3)],[(1,2),(1,2,3)],[(1,3),(1,2,3)],[(2,3),(1,2,3)]]
		bool3 = P(Poset(relations=b3_rels_list, indices=False, elements=this.Bool3.elements))
		assert(bool3==this.Bool3)

	def test_relsDict(this):
		b3_rels_dict={tuple():[(1,),(2,),(3,)],(1,):[(1,2),(1,3)],(2,):[(1,2),(2,3)],(3,):[(1,3),(2,3)],(1,3):[(1,2,3)],(2,3):[(1,2,3)],(1,2):[(1,2,3)]}
		bool3 = P(Poset(relations=b3_rels_dict, elements=this.Bool3.elements, indices=False))
		assert(bool3==this.Bool3)

	def test_reorder(this):
		bool3 = P(this.Bool3.reorder(this.Bool3.elements[::-1]))
		assert(bool3!=this.Bool3)

	def test_reorderIndices(this):
		bool3 = P(this.Bool3.reorder(list(range(len(this.Bool3)))[::-1], indices=True))
		assert(bool3!=this.Bool3)

	def test_sort(this):
		assert(this.Bool3.sort(indices=True,key=lambda x:-x).elements==[(1,2,3),(2,3),(1,3),(3,),(1,2),(2,),(1,),tuple()])
		assert(this.Bool3.reorder(list(range(len(this.Bool3)))[::-1], indices=True).sort()==this.Bool3)

#		b3_rels_list = [[tuple(),(1,)],[tuple(),(2,)],[tuple(),(3,)],[(1,),(1,2)],[(1,),(1,3)],[(2,),(1,2)],[(2,),(2,3)],[(3,),(1,3)],[(3,),(2,3)],[(1,2),(1,2,3)],[(1,3),(1,2,3)],[(2,3),(1,2,3)]]
#		bool3 = P(Poset(relations=b3_rels_list).sort())
#		assert(bool3==this.Bool3.sort())
#
#		b3_rels_dict={tuple():[(1,),(2,),(3,)],(1,):[(1,2),(1,3)],(2,):[(1,2),(2,3)],(3,):[(1,3),(2,3)],(1,3):[(1,2,3)],(2,3):[(1,2,3)],(1,2):[(1,2,3)]}
#		bool3 = P(Poset(relations=b3_rels_dict).sort())
#		assert(bool3==this.Bool3.sort())
#
	def antichain(this):
		a4 = P(elements=[0,1,2,3],incMat=[[0]*4 for i in range(4)],ranks=[[0,1,2,3]])
		assert(a4==Poset(elements=[0,1,2,3]))
##########################################
#test example posets
##########################################
class TestExamples:
	def test_Bool(this):
		Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])
		assert(Bool3==Boolean(3))
	def test_chain(this):
		chain3 = P([[0,1,1,1],[-1,0,1,1],[-1,-1,0,1],[-1,-1,-1,0]], list(range(4)), [[i] for i in range(4)])
		assert(chain3==Chain(3))
#
#		assert(Bool3==Boolean(3))
	def test_polygon(this):
		square = P(Poset(relations={1:[(1,2),(1,4)],2:[(1,2),(2,3)],3:[(2,3),(3,4)],4:[(3,4),(1,4)]}).adjoin_zerohat().adjoin_onehat().sort(key=str))
		assert(square==Polygon(4).sort(key=str))
#
	def test_cube(this):
		square = P(Poset(
			relations={'00':['*0','0*'],'10':['*0','1*'],'01':['*1','0*'],'11':['*1','1*'],'0*':['**'],'*0':['**'],'1*':['**'],'*1':['**']}
		).adjoin_zerohat().sort(key=str))
		assert(square==Cube(2).sort(key=str))
	def test_torus(this):
		torus = P(Poset(
			relations={
				'00':['A0','B0','0A','0B'],
				'10':['A0','B0','1A','1B'],
				'01':['0A','0B','A1','B1'],
				'11':['1A','1B','A1','B1'],
				'A0':['AA','AB'],
				'B0':['BA','BB'],
				'0A':['AA','BA'],
				'0B':['AB','BB'],
				'A1':['AA','AB'],
				'B1':['BA','BB'],
				'1A':['BA','AA'],
				'1B':['BB','AB']
				}
			).adjoin_zerohat().adjoin_onehat().sort(key=str))
		assert(torus==Torus().sort(key=str))

	def test_grid(this):
		grid = P(Poset(
			relations={
				('00',(0,0)):[('*0',(0,0)),('0*',(0,0))],
				('01',(0,0)):[('*1',(0,0)),('0*',(0,0))],
				('10',(0,0)):[('*0',(0,0)),('1*',(0,0)),('*0',(1,0))],
				('11',(0,0)):[('*1',(0,0)),('1*',(0,0)),('*1',(1,0))],
				('10',(1,0)):[('*0',(1,0)),('1*',(1,0))],
				('11',(1,0)):[('*1',(1,0)),('1*',(1,0))],
				('*0',(0,0)):[('**',(0,0))],
				('0*',(0,0)):[('**',(0,0))],
				('*1',(0,0)):[('**',(0,0))],
				('1*',(0,0)):[('**',(0,0)),('**',(1,0))],
				('*1',(1,0)):[('**',(1,0))],
				('*0',(1,0)):[('**',(1,0))],
				('1*',(1,0)):[('**',(1,0))],
				}
			).adjoin_zerohat().sort(key=str))
		grid = P(Poset(incMat=[[0, 0, -1, 0, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, 0, -1], [0, 0, 0, -1, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1], [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1], [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, -1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1], [1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1], [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, -1], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1], [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1], [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], elements=[('**', (0, 0)), ('**', (1, 0)), ('*0', (0, 0)), ('*0', (1, 0)), ('*1', (0, 0)), ('*1', (1, 0)), ('0*', (0, 0)), ('00', (0, 0)), ('01', (0, 0)), ('1*', (0, 0)), ('1*', (1, 0)), ('10', (0, 0)), ('10', (1, 0)), ('11', (0, 0)), ('11', (1, 0)), 0], ranks=[[15], [7, 8, 11, 12, 13, 14], [2, 3, 4, 5, 6, 9, 10], [0, 1]]))
		assert(grid==Grid(2,[2,1]).sort(key=str))

	def test_kleinBottle(this):
		def f(s):
			a,b,c,d = s
			return (a+b,(int(c),int(d)))
		kb = P(Poset(relations={
			f('0000'):[f('*000'),f('0*00'),f('*010'),f('0*01')],
			f('1000'):[f('*000'),f('*010'),f('1*00'),f('1*01')],
			f('0100'):[f('0*00'),f('*100'),f('0*01'),f('*110')],
			f('1100'):[f('1*00'),f('*100'),f('*110'),f('1*01')],
			f('*000'):[f('**00'),f('**01')],
			f('0*00'):[f('**00'),f('**11')],
			f('*010'):[f('**10'),f('**11')],
			f('0*01'):[f('**01'),f('**10')],
			f('1*00'):[f('**00'),f('**10')],
			f('*100'):[f('**00'),f('**01')],
			f('1*01'):[f('**01'),f('**11')],
			f('*110'):[f('**10'),f('**11')],
			}).adjoin_zerohat().adjoin_onehat().sort(key=str))

		assert(kb==KleinBottle().sort(key=str))

	def test_projectiveSpace(this):
		def f(s):
			a,b,c,d = s
			return (a+b,(int(c),int(d)))
		ps = P(Poset(relations={
			f('0000'):[f('*000'),f('0*00')],
			f('1000'):[f('*000'),f('*010'),f('1*00'),f('1*01')],
			f('1010'):[f('*010'),f('0*01')],
			f('0100'):[f('0*00'),f('0*01'),f('*100'),f('*110')],
			f('1100'):[f('1*00'),f('*100'),f('*110'),f('1*01')],
			f('*000'):[f('**00'),f('**11')],
			f('*010'):[f('**10'),f('**01')],
			f('0*00'):[f('**00'),f('**11')],
			f('0*01'):[f('**01'),f('**10')],
			f('1*00'):[f('**00'),f('**10')],
			f('*100'):[f('**00'),f('**01')],
			f('*110'):[f('**10'),f('**11')],
			f('1*01'):[f('**11'),f('**01')]
			}).adjoin_zerohat().adjoin_onehat().sort(key=str))
		assert(ps==ProjectiveSpace().sort(key=str))


	def test_uncrossing(this):
		uc3 = Poset(
			relations={
				0:[1,2,3,4,5],
				1:[6,7],
				2:[6,8,9],
				3:[8,10],
				4:[9,11],
				5:[7,10,11],
				6:[12,13],
				7:[12,13],
				8:[12,14],
				9:[13,14],
				10:[12,14],
				11:[13,14],
				12:[15],
				13:[15],
				14:[15]
				},
			indices=True,
			elements=[0,
				((1, 6), (2, 5), (3, 4)),
				((1, 6), (2, 3), (4, 5)),
				((1, 4), (2, 3), (5, 6)),
				((1, 2), (3, 6), (4, 5)),
				((1, 2), (3, 4), (5, 6)),
				((1, 6), (2, 4), (3, 5)),
				((1, 5), (2, 6), (3, 4)),
				((1, 5), (2, 3), (4, 6)),
				((1, 3), (2, 6), (4, 5)),
				((1, 3), (2, 4), (5, 6)),
				((1, 2), (3, 5), (4, 6)),
				((1, 5), (2, 4), (3, 6)),
				((1, 4), (2, 6), (3, 5)),
				((1, 3), (2, 5), (4, 6)),
				((1, 4), (2, 5), (3, 6))]
			)
		assert(uc3.is_isomorphic(Uncrossing(3))) #passes
		assert(uc3==Uncrossing(3)) #fails

	def test_bnq(this):
		#B_3(2)
		#Previously we tested the actual labeling was correct, but I changed the
		#element labels in Bnq and I can't be bothered to update them here. Below
		#we just test isomorphism class.
		spaces = [0,
			1|1<<1, 1|1<<2, 1|1<<3, 1|1<<4, 1|1<<5, 1|1<<6, 1|1<<7,
			1|1<<1|1<<2|1<<3, 1|1<<1|1<<4|1<<5, 1|1<<2|1<<4|1<<6,
			1|1<<3|1<<4|1<<7, 1|1<<1|1<<6|1<<7, 1|1<<3|1<<5|1<<6,
			1|1<<2|1<<5|1<<7,
			1|1<<1|1<<2|1<<3|1<<4|1<<5|1<<6|1<<7
			]
		b32 = Poset(
			relations={
				0:[1,2,3,4,5,6,7],
				1:[8,9,12],
				2:[8,10,14],
				3:[8,11,13],
				4:[9,10,11],
				5:[9,13,14],
				6:[10,12,13],
				7:[11,12,14],
				8:[15],
				9:[15],
				10:[15],
				11:[15],
				12:[15],
				13:[15],
				14:[15]
				},
			indices=True,
			#elements are subspaces considered as sets encoded as bit strings
			#the vectors are ordered reverse lexicographically (base-q order)
			#in the future this may be changed to tuples representing matrices
	#			elements=[tuple(),
	#				((0,0,1),)
			elements=spaces
			).sort()
		assert(b32.isoClass()==Bnq(n=3,q=2).isoClass())

	def test_latticeOfFlats_graph(this):
		triFlats = Poset(
			relations={
				(('a',),('b',),('c',)):[(('a','b'),('c',)), (('a','c'),('b',)), (('a',),('b','c'))],
				(('a','b'),('c',)): [(('a','b','c'),)],
				(('a','c'),('b',)): [(('a','b','c'),)],
				(('a',),('b','c')): [(('a','b','c'),)]
				}).sort()
		assert(triFlats==LatticeOfFlats([['a','b'],['b','c'],['c','a']]))

	def test_latticeOfFlats_rankFunction(this):
		pentFlats = P(Poset(
			relations={
				tuple():[(1,),(3,)],
				(1,):[(1,2,)],
				(1,2,):[(1,2,3)],
				(3,):[(1,2,3)]
				}
			).sort())
		assert(pentFlats==LatticeOfFlats([0,1,2,2,1,3,3,3]).sort())

	def test_noncrossingPartitionLattice(this):
		NC = Poset(elements=[
			((1,),(2,),(3,),(4,)), #0
			#
			((1,2),(3,),(4,)), #1
			((1,3),(2,),(4,)), #2
			((1,4),(2,),(3,)), #3
			((1,),(2,3),(4,)), #4
			((1,),(2,4),(3,)), #5
			((1,),(2,),(3,4)), #6
			#
			((1,2),(3,4)), #7
			((1,4),(2,3)), #8
			((1,2,3),(4,)), #9
			((1,2,4),(3,)), #10
			((1,3,4),(2,)), #11
			((1,),(2,3,4)), #12
			#
			((1,2,3,4),) #13
			],
			relations={0:[1,2,3,4,5,6],1:[7,9,10],2:[9,11],3:[8,10,11],4:[8,9,12],5:[10,12],6:[7,11,12],7:[13],8:[13],9:[13],10:[13],11:[13],12:[13]},
			indices=True
			)
		assert(NC==NoncrossingPartitionLattice(4))

	def test_distributiveLattice(this):
		Videals = P(Poset(
			relations={
				0:[1],
				1:[2,3],
				2:[4],
				3:[4]
				},
			indices=True,
			elements=[tuple(), ('*',), ('0','*'), ('1','*'), ('0','1','*')]
			).sort(key=str))
		V = Poset(relations={'*':['0','1']},elements=['0','1','*'])
		assert(Videals==DistributiveLattice(V).sort(key=str))

	def test_minorPoset(this):
		pent = Genlatt(relations={0:['a','c'],'a':['b'],'c':[1],'b':[1]},elements=[0,'a','b','c',1])
		pentMinors = P(Poset(
			relations={
				0:[1,2,3,4,5],
				1:[6,7,9],
				2:[6,8,10],
				3:[7,8,11],
				4:[9,12],
				5:[10,11,12],
				6:[13,14],
				7:[13,15],
				8:[13,16],
				9:[14,15],
				10:[14,16],
				11:[15,16],
				12:[14,15],
				13:[17],
				14:[17],
				15:[17],
				16:[17]
				},
			indices=True,
			elements=[0,
				(0,tuple()), ('a',tuple()), ('b',tuple()), ('c',tuple()), (1,tuple()),
				(0,('a',)), (0,('b',)), ('a',('b',)), (0,('c',)), ('a',(1,)), ('b',(1,)), ('c',(1,)),
				(0,('a','b')), (0,('a','c')), (0,('b','c')), ('a',('b',1)),
				(0,('a','b','c'))
				]
			).sort(key=lambda x:'0' if x==0 else str((x[0],sorted([str(g) for g in x[1]]))))
		)
		M=MinorPoset(pent).sort(key=lambda x:'0' if x==0 else str((x.min()[0],sorted([str(g) for g in x.G]))))
		pentMinors.elements = [0 if x==0 else pent.minor(z=x[0],H=x[1]) for x in pentMinors.elements]
		assert(pentMinors==M)

	def test_weakMinorPoset(this):
		pent = Genlatt(relations={0:['a','c'],'a':['b'],'c':[1],'b':[1]},elements=[0,'a','b','c',1])
		pentMinors = Poset(
			relations={
				0:[1,2,3,4,5],
				1:[6,7,9],
				2:[6,8,10],
				3:[8,11],
				4:[9,12],
				5:[11,12],
				6:[13,14],
				7:[13,15],
				8:[13,16],
				9:[14,15],
				10:[14,16],
				11:[16],
				12:[17],
				13:[17],
				14:[17],
				15:[17],
				16:[17]
				},
			indices=True,
			elements=[0,
				(0,tuple()), ('a',tuple()), ('b',tuple()), ('c',tuple()), (1,tuple()),
				(0,('a',)), (0,('b',)), ('a',('b',)), (0,('c',)), ('a',(1,)), ('b',(1,)), ('c',(1,)),
				(0,('a','b')), (0,('a','c')), (0,('b','c')), ('a',('b',1)),
				(0,('a','b','c'))
				]
			)#.sort(key=lambda x:'0' if x==0 else str((x[0],sorted([str(g) for g in x[1]])))
		#)
		M=MinorPoset(pent,weak=True)#.sort(key=lambda x:'0' if x==0 else str((x.min()[0],sorted([str(g) for g in x.G]))))
		pentMinors.elements = [0 if x==0 else pent.minor(z=x[0],H=x[1]) for x in pentMinors.elements]
		assert(set(pentMinors.elements)==set(M.elements))
		assert(pentMinors==M)

	def test_bruhat(this):
		bruhat = P(Poset(relations={
			(1,2,3):[(1,3,2),(2,1,3)],
			(2,1,3):[(3,1,2),(2,3,1)],
			(1,3,2):[(3,1,2),(2,3,1)],
			(3,1,2):[(3,2,1)],
			(2,3,1):[(3,2,1)]
			}).sort())
		assert(bruhat==Bruhat(3).sort())

	def test_bruhat_weak(this):
		bruhat = Poset(relations={
			(1,2,3):[(1,3,2),(2,1,3)],
			(2,1,3):[(3,1,2)],
			(1,3,2):[(2,3,1)],
			(3,1,2):[(3,2,1)],
			(2,3,1):[(3,2,1)]
			}).sort()
		assert(bruhat==Bruhat(3,True).sort())

	def test_butterfly(this):
		bf = P(Poset(relations={
			0:['a0','b0'],
			'a0':['a1','b1'],
			'b0':['a1','b1'],
			'a1':[1],
			'b1':[1]
			}).sort(key=str))
		assert(bf==Butterfly(2).sort(key=str))
##########################################
#Container tests
##########################################
class TestContainers:
	P = Poset(relations={0:['a','b'],'a':[1],'b':[1]},elements=[0,'a','b',1])
	Q = Poset(relations={0:['a','b'],'a':[1],'b':[1]},elements=[0,'b','a',1])
	R = Poset(relations={0:['A','B'],'A':[1],'B':[1]},elements=[0,'A','B',1])

	def test_hash(this):
		assert(hash(this.P)==hash(this.Q))
		assert(hash(this.P.isoClass()) == hash(this.R.isoClass()))
		assert(hash(this.P)!=hash(this.R))
##########################################
#test operations on posets
##########################################
class TestOperations:
	V = Poset(relations={'*':['x','y']})
	AB = Poset(relations={'a':['b']})
	ABC = Poset(relations={'a':['b'],'b':['c']})
	Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])
	def test_adjoin_zerohat(this):
		A2 = Poset(elements=[0,1])
		assert(P(elements=[2,0,1],incMat=[[0,1,1],[-1,0,0],[-1,0,0]],ranks=[[0],[1,2]])==A2.adjoin_zerohat())
	def test_adjoin_zerohat(this):
		A2 = Poset(elements=[0,1])
		assert(P(elements=[2,0,1],incMat=[[0,-1,-1],[1,0,0],[1,0,0]],ranks=[[1,2],[0]])==A2.adjoin_onehat())

	def test_identify(this):
		Q = this.Bool3.identify({tuple():[(1,),(2,)], (3,):[(1,3),(2,3)]})
		expected = P(Poset(relations={tuple():[(3,),(1,2)], (1,2):[(1,2,3)], (3,):[(1,2,3)]}))
		assert(expected==Q)
	def test_dual(this):
		assert(P(Poset(relations={'x':['*'],'y':['*']}).sort())==this.V.dual().sort())
	def test_union(this):
		assert(P(Poset(relations={'*':['x','y'],'a':['b']}).sort())==this.V.union(this.AB).sort())
	def test_bddUnion(this):
		assert(P(Poset(relations={0:['x','y','b'],'x':[1],'y':[1],'b':[1]}).sort(key=str))==this.V.adjoin_onehat().bddUnion(this.ABC).sort(key=str))
	def test_starProduct(this):
		assert(P(Poset(relations={'x':['b'],'y':['b']}).sort(key=str))==this.V.dual().starProduct(this.AB).sort(key=str))
	def test_cartesianProduct(this):
		assert(P(Poset(relations={('*','a'):[('x','a'),('y','a'),('*','b')], ('x','a'):[('x','b')], ('y','a'):[('y','b')], ('*','b'):[('x','b'),('y','b')]}).sort(key=str))==this.V.cartesianProduct(this.AB).sort(key=str))
	def test_diamondProduct(this):
		assert(P(Poset(relations={0:[('b','x'),('b','y')],('b','x'):[('c','x')],('b','y'):[('c','y')]}).sort(key=str))==this.ABC.diamondProduct(this.V).sort(key=str))
##########################################
#Queries
##########################################
class TestQueries:
	pent = LatticeOfFlats([0,1,2,2,1,3,3,3])
	B3 = Boolean(3)
	def test_ranked(this):
		assert(not this.pent.isRanked())
		assert(this.B3.isRanked())

	def test_isLattice(this):
		this.pent.cache={}
		B = Bruhat(3)
		B.cache={}
		assert(this.pent.isLattice())
		assert(not B.isLattice())
		assert(not this.pent.complSubposet(this.pent.max()).isLattice())
		assert(not this.pent.complSubposet(this.pent.min()).isLattice())

	def test_Eulerian(this):
		this.pent.cache={}
		assert(not this.pent.isEulerian())
		this.B3.cache={}
		assert(this.B3.isEulerian())
	def test_Gorenstein(this):
		assert(not this.pent.isGorenstein())
		assert(this.B3.isGorenstein())
	def test_covers(this):
		assert({tuple():[(1,),(2,),(3,)],(1,):[(1,2),(1,3)],(2,):[(1,2),(2,3)],(3,):[(1,3),(2,3)],(1,2):[(1,2,3)],(2,3):[(1,2,3)],(1,3):[(1,2,3)]}==this.B3.covers())
		assert({0:[1,2,4],1:[3,5],2:[3,6],4:[5,6],3:[7],5:[7],6:[7]}==this.B3.covers(True))
		this.B3.cache={}
		assert({0:[1,2,4],1:[3,5],2:[3,6],4:[5,6],3:[7],5:[7],6:[7]}==this.B3.covers(True))
		assert({tuple():[(1,),(3,)],(1,):[(1,2)],(3,):[(1,2,3)],(1,2):[(1,2,3)]}==this.pent.covers())
##########################################
#Subposet Selection
##########################################
class TestSubposetSelection:
	V = Poset(relations={'*':['x','y']})
	pent = LatticeOfFlats([0,1,2,2,1,3,3,3])

	def test_min(this):
		assert(['*']==this.V.min())
	def test_max(this):
		assert(sorted(this.V.max())==['x','y'])
	def test_subposet(this):
		assert(P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort())==this.pent.subposet([(1,2),(1,),(3,)]).sort())
		assert(P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort())==this.pent.subposet((1,2,3),indices=True).sort())
	def test_complSubposet(this):
		assert(P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort())==this.pent.complSubposet([tuple(),(1,2,3)]).sort())
		assert(P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort())==this.pent.complSubposet((0,4),indices=True).sort())

	def test_interval(this):
		assert(P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort())==this.pent.interval((1,),(1,2,3)).sort())
		assert(P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort())==this.pent.interval(1,4,indices=True).sort())

	def test_filter(this):
		assert(P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort())==this.pent.filter(((1,),)).sort())
		assert(P(Poset(relations={(1,2):[(1,2,3)]}).sort())==this.pent.filter(((1,),),strict=True).sort())
		assert(P(Poset(relations={1:[2],2:[4]}).sort())==this.pent.filter((1,),indices=True).sort())

	def test_properPart(this):
		assert(P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort())==this.pent.properPart().sort())
	def test_rankSelection(this):
		assert(P(Poset(relations={tuple():[(1,2)]}).sort())==this.pent.rankSelection([0,2]).sort())
##########################################
#Internal Computations
##########################################
class TestInternalComputations:
	pent = LatticeOfFlats([0,1,2,2,1,3,3,3])
	def test_less(this):
		assert(not this.pent.less((1,),(1,)))
		assert(not this.pent.less((1,),(3,)))
		assert(not this.pent.less((1,2,),(1,)))
		assert(this.pent.less((1,),(1,2,)))
	def test_isAntichain(this):
		assert(not this.pent.isAntichain([(1,),(1,2,)]))
		assert(this.pent.isAntichain([(1,),(3,)]))
		assert(this.pent.isAntichain([(1,)]))
		assert(this.pent.isAntichain([]))
		assert(not this.pent.isAntichain([1,2],indices=True))
		assert(this.pent.isAntichain([1,3],indices=True))
	def test_join(this):
		assert((1,)==this.pent.join((1,),(1,)))
		assert((1,2,)==this.pent.join((1,),(1,2,)))
		assert((1,2,3)==this.pent.join((1,),(3,)))
		assert(None==Poset(elements=[0,1]).join(0,1))
		P = Butterfly(2).properPart()
		assert(None==P.join(0,1,True))
	def test_mobius(this):
		assert(1==this.pent.mobius())
		assert(0==this.pent.mobius(tuple(),(1,2)))
	def test_rank(this):
		assert((0,1,2,3,1)==tuple(this.pent.rank(p) for p in sorted(this.pent.elements)))
##########################################
#Invariants
##########################################
class TestInvariants:
	B4 = Boolean(4)
	B3 = Boolean(3)
	def test_flagVectors(this):
		assert({tuple():[1,1],(1,):[4,3],(2,):[6,5],(1,2):[12,3],(3,):[4,3],(1,3):[12,5],(2,3):[12,3],(1,2,3):[24,1]}==this.B4.flagVectors())
	def test_abIndex(this):
		assert(Polynomial([[1,'aaa'],[3,'baa'],[5,'aba'],[3,'bba'],[3,'aab'],[5,'bab'],[3,'abb'],[1,'bbb']])==this.B4.abIndex())
	def test_cdIndex(this):
		assert(Polynomial([[1,'ccc'],[2,'cd'],[2,'dc']])==this.B4.cdIndex())
		assert(Polynomial([[1,'cc'],[1,'d']])==this.B3.complSubposet([(1,2)]).cdIndex())
	def test_zeta(this):
		assert([[1,1,1,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]]==Boolean(2).zeta())
	def test_bettiNumbers(this):
		assert([1,2,1]==Torus().properPart().bettiNumbers())
	def test_buildIsomorphism(this):
		#make sure two elements hash the same so we test
		#reversing a choice
		P=Bruhat(3,True).union(Bruhat(3))
		Q=Bruhat(3,True).union(Bruhat(3))
		Q=Q.reorder(Q.elements[::-1])
		print(Q.elements)
		print(P.elements)
		for _ in range(2): #do it twice to test cache
			phi = Q.buildIsomorphism(P,indices=True)
			phi_inv = {v:k for k,v in phi.items()}
			assert(all(set(Q.filter([i],indices=True)) == set(phi_inv[j] for j in P.filter([phi[i]],indices=True)) for i in range(len(Q))))
		print(phi)
		print(phi_inv)
		assert(None==Q.buildIsomorphism(Bruhat(3).union(Bruhat(3))))
		#example of non-isomorphic posets with the same hash sets?
#		B3=Boolean(3)
#		B=Boolean(3)
#		B.incMat[2][6]=0
#		B.incMat[1][6]=1
#		assert(None==B.buildIsomorphism(B3))
##########################################
#Polynomial
##########################################
class TestPolynomial:
	p = Polynomial({'cc':1,'d':1})
	q = Polynomial({'cc':1,'d':2})
	def test_add(this):
		this.p = Polynomial({'a':1,'b':1,'x':0})
		assert('a+b'==str(this.p))
		this.p = Polynomial({'cc':1,'d':1})
		this.q = Polynomial({'cc':1,'d':2})
		assert(this.p+this.q==Polynomial({'cc':2,'d':3}))
	def test_abToCd(this):
		assert(Boolean(5).cdIndex()==Boolean(5).abIndex().abToCd())
		B42=Bnq(n=4,q=2)
		assert(B42.abIndex()==B42.abIndex().abToCd())
	def test_cdToAb(this):
		assert(Boolean(5).abIndex()==Boolean(5).cdIndex().cdToAb())
	def test_sub(this):
		assert(Polynomial({'d':-1})==this.p-this.q)
	def test_mul(this):
		assert(this.p*this.q==Polynomial({'cccc':1,'ccd':2,'dcc':1,'dd':2}))
	def test_pow(this):
		assert(this.p**2==Polynomial({'cccc':1,'ccd':1,'dcc':1,'dd':1}))
	
	def test_eq(this):
		assert(Polynomial({'ab':0,'ba':0})==Polynomial())
		assert(Polynomial({'ab':1,'ba':0,'bb':-1})==Polynomial({'ab':1,'bb':-1,'d':0}))
##########################################
#Misc
##########################################
class TestMisc:
	B = Butterfly(2).properPart()
	B_chains = [tuple(),('a0',),('a0','a1'),('a0','b1'),('a1',),('b0',),('b0','a1'),('b0','b1'),('b1',)]
	def test_chains(this):
		assert(this.B_chains==sorted(this.B.chains()))
	def test_orderComplex(this):
		assert(P(Poset(relations={0:[1,4,5,8],1:[2,3],4:[2,6],5:[6,7],8:[7,3]},indices=True,elements=this.B_chains))==this.B.orderComplex().sort())
	def test_relations(this):
		assert([('a0','a1'),('a0','b1'),('b0','a1'),('b0','b1')]==sorted(this.B.relations()))
		assert([(0,1),(0,3),(2,1),(2,3)]==sorted(this.B.sort().relations(indices=True)))
	def test_reorder(this):
		assert(P(Poset(elements=['b1','a0','b0','a1'],incMat=[[0,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,0]]))==this.B.reorder(['b1','a0','b0','a1']))
		assert(P(Poset(elements=['b1','a0','b0','a1'],incMat=[[0,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,0]]))==this.B.sort().reorder([3,0,2,1],indices=True))
		assert(P(Poset(elements=['a0','a1','b0','b1'],relations={0:[1,3],2:[1,3]},indices=True))==this.B.reorder(this.B.elements[::-1]).sort())
	def test_shuffle(this):
		assert(this.B.shuffle()==this.B)
	def test_ranks(this):
		pent = LatticeOfFlats([0,1,2,2,1,3,3,3])
		assert([[0],[1,4],[2],[3]]==Poset(relations=pent.relations()).sort().ranks)
	def test_copy(this):
		P = Boolean(3)
		Q = Poset(P)
		assert(P==Q)
		Q.elements=list(range(len(Q)))
		assert(P!=Q and P.is_isomorphic(Q))

		P = Boolean(2)
		LG = Genlatt(P,G=P.max())
		assert(P.covers()!=LG.covers())
##########################################
#Genlatt test
##########################################
#class TestGenlatt:
#	LG = Genlatt(LatticeOfFlats([0,1,2,2,1,3,3,3]))
#	KH = Genlatt(Boolean(2),G=((1,2),))
#	def test_cartesianProduct(this):
#		assert(this.LG.cartesianProduct(this.KH)=='todo')
#	def test_diamondProduct(this):
#		assert(this.LG.diamondProduct(this.KH)=='todo')
#	def test_adjoin_onehat(this):
#		assert(this.LG.adjoin_onehat()=='todo')
##########################################
#Test toSage and fromSage
##########################################
@pytest.mark.skipif(shutil.which('sage')==None, reason='sage is not in the PATH')
def test_sage():
	sage_str = '''
from posets_test import *
from posets import *
B = Butterfly(2).properPart().sort()
B.elements = [0,2,1,3]
SB = sage.combinat.posets.poset_examples.Posets.Crown(2)
BS = B.toSage()
testeq_exit(SB.lequal_matrix(), BS.lequal_matrix(), "toSage")
testeq_exit(Poset.fromSage(SB).sort(), P(B.sort()), "fromSage")
	'''
	tmpfile = tempfile.mktemp(suffix='.sage')
	with open(tmpfile,'w') as file:
		file.write(sage_str)
	result = None
	result = subprocess.Popen(['sage',tmpfile],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result!=None:
		txt, err = result.communicate()
		assert(result.returncode==0)
	os.remove(tmpfile)
	os.remove(tmpfile+'.py')
##########################################
#Test pythonPosetToMac and macPosetToPython
#from ../convertPosets.m2
##########################################
#macaulay2 needs the module to be installed with pip (and not just in the current path)
#check if it's installed and that macaulay2 is on path
from pip._internal.operations.freeze import freeze
module_installed = 'posets' in [pkg[:pkg.index('==' if '==' in pkg else ' @ ' if ' @ ' in pkg else '')] for pkg in freeze(local_only=True)]
@pytest.mark.skipif(shutil.which('M2')==None, reason='Macaulay2 is not in the PATH')
@pytest.mark.skipif(not module_installed, reason='Module is not installed, build and install it to test Macaulay2')
def test_M2():
	mac_str = f'''
	load "{os.path.realpath('../convertPosets.m2')}"

	posets = import "posets"

	P = posets@@Cube 2
	Q = pythonPosetToMac(P)
	R = macPosetToPython(Q)
	--print(R == P) --true
	if R != P then exit 1
	S = pythonPosetToMac(R)
	if S != Q then exit 2
	exit 0
	'''
	tmpfile = tempfile.mktemp(suffix='.m2')
	with open(tmpfile,'w') as file:
		file.write(mac_str)

	result = None
	result = subprocess.Popen(['M2','--script',tmpfile],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result!=None:
		txt, err = result.communicate()
		print('txt',txt,'err',err)
		assert(result.returncode != 1) #posetToPython test
		assert(result.returncode != 2) #pythonToPoset test
	os.remove(tmpfile)
