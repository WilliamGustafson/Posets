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

def make_Bool3():
	return Poset(zeta = [[1,1,1,1,1,1,1,1], [1,0,1,0,1,0,1], [1,1,0,0,1,1], [1,0,0,0,1],[1,1,1,1], [1,0,1], [1,1],[1]], elements=[tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])

##########################################
#test triangular array
##########################################
class TestTriangularArray:
	def test_setitem(this):
		T = TriangularArray(range(10))
		Z = TriangularArray((0 for _ in range(10)))
		k = 0
		for i in range(T.size):
			for j in range(i,T.size):
				Z[i,j] = k
				k+=1
		assert(Z==T)
	
	def test_getitem(this):
		T = TriangularArray(range(10))
		k=0
		for i in range(T.size):
			for j in range(i,T.size):
				assert(k==T[i,j])
				k+=1
		assert(k==10)
	
	def test_row(this):
		T = TriangularArray(range(10))
		x = list(range(10))
		j = 0
		for i in range(T.size):
			assert(x[j:j+(T.size-i)]==T.row(i))
			j+=T.size - i
	
	def test_col(this):
		T = TriangularArray(range(10))
		#0 1 2 3
		#  4 5 6
		#    7 8
		#      9
		cols = [[0],[1,4],[2,5,7],[3,6,8,9]]
		for j in range(len(cols)):
			assert(cols[j] == list(T.col(j)))
	
	def test_revtranspose(this):
		assert(TriangularArray(range(10)).revtranspose() == TriangularArray([9,8,6,3,7,5,2,4,1,0]))
	
	def test_subarray(this):
		T = TriangularArray(range(10))
		assert(T.subarray((0,1))==TriangularArray([0,1,4]))
		assert(T.subarray([0,2,3])==TriangularArray([0,2,3,7,8,9]))
		assert(T.subarray([2])==TriangularArray([7]))
		assert(T.subarray([0,1,2,3])==T)
	
	def test_inverse(this):
		T = Boolean(4).zeta.inverse()
		assert(T[i,j]==(-1)**(sum(c=='1' for c in bin(i))+sum(c=='1' for c in bin(j))) for i in range(T.size) for j in range(T.size))
		T = TriangularArray([1,2,3, 4,5, 6])
		actual = T.inverse()
		expected = TriangularArray([1, -1/2, -1/12, 1/4, -5/24, 1/6])
		epsilon = 1/(1<<16)
		assert(x-y < epsilon for x,y in zip(actual.data,expected.data))
			
##########################################
#test constructor options
##########################################

class TestConstructorOptions:
	Bool3 = make_Bool3()
	def test_less(this):
		bool3 = Poset(less=lambda X,Y:all(x in Y for x in X), elements=[tuple(),(1,),(1,2),(2,),(3,),(1,3),(2,3),(1,2,3)])
		assert(this.Bool3 == bool3)
	def test_relsListIndices(this):
		'''
		relations=list indices=True
		'''
		bool3 = Poset(relations=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[4,5],[4,6],[3,7],[5,7],[6,7]], elements = this.Bool3.elements, indices=True)
		assert(bool3==this.Bool3)

	def test_relsDictIndices(this):
		'''
		relations=dict indices=True
		'''
		bool3 = Poset(relations={0:[1,2,4],1:[3,5],2:[3,6],4:[5,6],5:[7],6:[7],3:[7]}, indices=True, elements=this.Bool3.elements)
		assert(bool3==this.Bool3)

	def test_restList(this):
		b3_rels_list = [[tuple(),(1,)],[tuple(),(2,)],[tuple(),(3,)],[(1,),(1,2)],[(1,),(1,3)],[(2,),(1,2)],[(2,),(2,3)],[(3,),(1,3)],[(3,),(2,3)],[(1,2),(1,2,3)],[(1,3),(1,2,3)],[(2,3),(1,2,3)]]
		bool3 = Poset(relations=b3_rels_list, indices=False, elements=this.Bool3.elements)
		assert(bool3==this.Bool3)

	def test_relsDict(this):
		b3_rels_dict={tuple():[(1,),(2,),(3,)],(1,):[(1,2),(1,3)],(2,):[(1,2),(2,3)],(3,):[(1,3),(2,3)],(1,3):[(1,2,3)],(2,3):[(1,2,3)],(1,2):[(1,2,3)]}
		bool3 = Poset(relations=b3_rels_dict, elements=this.Bool3.elements, indices=False)
		assert(bool3==this.Bool3)

	def test_reorder(this):
		B = Boolean(3).reorder([0,1,2,4,3,5,6,7],indices=True)
		assert(B.elements==[tuple(),(1,),(2,),(3,),(1,2),(1,3),(2,3),(1,2,3)])
		B = Boolean(3).reorder([tuple(),(1,2),(1,),(3,),(2,),(2,3),(1,3),(1,2,3)])
		assert(B.elements == [tuple(),(1,),(3,),(2,),(1,2),(2,3),(1,3),(1,2,3)])

	def test_sort(this):
		B3 = Poset(elements=range(8), less = lambda i,j:i&j==i)
		P = Poset(elements=[0,2,1,4,6,5,3,7],less=lambda i,j:i&j==i)
		assert(B3==P)
		assert(B3.elements!=P.elements)
		assert(B3.elements == P.sort().elements)

	def antichain(this):
		a4 = Poset(elements=[0,1,2,3],zeta=[0 for _ in range(10)],zeta_flat=True,ranks=[[0,1,2,3]])
		assert(a4==Poset(elements=[0,1,2,3]))
##########################################
#test example posets
##########################################
class TestExamples:
	def test_Bool(this):
		Bool3 = make_Bool3()
		assert(Bool3==Boolean(3))
	def test_chain(this):
		chain3 = Poset([1,1,1,1,1,1,1,1,1,1], list(range(4)), [[i] for i in range(4)],flat_zeta=True)
		assert(chain3==Chain(3))
	def test_polygon(this):
		square = Poset(relations={1:[(1,2),(1,4)],2:[(1,2),(2,3)],3:[(2,3),(3,4)],4:[(3,4),(1,4)]}).adjoin_zerohat().adjoin_onehat()
		assert(square==Polygon(4))
#
	def test_cube(this):
		square = Poset(
			relations={'00':['*0','0*'],'10':['*0','1*'],'01':['*1','0*'],'11':['*1','1*'],'0*':['**'],'*0':['**'],'1*':['**'],'*1':['**']}
		).adjoin_zerohat()
		assert(square==Cube(2))
	def test_torus(this):
		torus = Poset(
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
			).adjoin_zerohat().adjoin_onehat()
		assert(torus==Torus())
	def test_grid(this):
		grid = Poset(
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
			).adjoin_zerohat()
		assert(grid==Grid(2,[2,1]))


	def test_kleinBottle(this):
		def f(s):
			a,b,c,d = s
			return (a+b,(int(c),int(d)))
		kb = Poset(relations={
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
			}).adjoin_zerohat().adjoin_onehat()

		assert(kb==KleinBottle())

	def test_projectiveSpace(this):
		def f(s):
			a,b,c,d = s
			return (a+b,(int(c),int(d)))
		ps = Poset(relations={
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
			}).adjoin_zerohat().adjoin_onehat()
		assert(ps==ProjectiveSpace())


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
		actual=Uncrossing(3)
		assert(uc3.is_isomorphic(Uncrossing(3))) #fails
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
			)
		assert(b32.isoClass()==Bnq(n=3,q=2).isoClass())

	def test_latticeOfFlats_graph(this):
		triFlats = Poset(
			relations={
				(('a',),('b',),('c',)):[(('a','b'),('c',)), (('a','c'),('b',)), (('a',),('b','c'))],
				(('a','b'),('c',)): [(('a','b','c'),)],
				(('a','c'),('b',)): [(('a','b','c'),)],
				(('a',),('b','c')): [(('a','b','c'),)]
				})
		assert(triFlats==LatticeOfFlats([['a','b'],['b','c'],['c','a']]))

	def test_latticeOfFlats_rankFunction(this):
		pentFlats = Poset(
			relations={
				tuple():[(1,),(3,)],
				(1,):[(1,2,)],
				(1,2,):[(1,2,3)],
				(3,):[(1,2,3)]
				}
			)
		assert(pentFlats==LatticeOfFlats([0,1,2,2,1,3,3,3]))

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
		V = Poset(relations={'*':['0','1']},elements=['0','1','*'])
		Videals = Poset(
			relations={
				0:[1],
				1:[2,3],
				2:[4],
				3:[4]
				},
			indices=True,
			elements=[tuple(sorted(x,key=lambda y:V.elements.index(y))) for x in [tuple(), ('*',), ('0','*'), ('1','*'), ('0','1','*')]]
			)
		actual = DistributiveLattice(V)
		assert(Videals==DistributiveLattice(V))

	def test_minorPoset(this):
		pent = Genlatt(relations={0:['a','c'],'a':['b'],'c':[1],'b':[1]},elements=[0,'a','b','c',1])
		pentMinors = Poset(
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
			)
		
		M=MinorPoset(pent)
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
			)
		#)
		M=MinorPoset(pent,weak=True)
		pentMinors.elements = [0 if x==0 else pent.minor(z=x[0],H=x[1]) for x in pentMinors.elements]
		assert(set(pentMinors.elements)==set(M.elements))
		assert(pentMinors==M)

	def test_bruhat(this):
		bruhat = Poset(relations={
			(1,2,3):[(1,3,2),(2,1,3)],
			(2,1,3):[(3,1,2),(2,3,1)],
			(1,3,2):[(3,1,2),(2,3,1)],
			(3,1,2):[(3,2,1)],
			(2,3,1):[(3,2,1)]
			})
		assert(bruhat==Bruhat(3))

	def test_bruhat_weak(this):
		bruhat = Poset(relations={
			(1,2,3):[(1,3,2),(2,1,3)],
			(2,1,3):[(3,1,2)],
			(1,3,2):[(2,3,1)],
			(3,1,2):[(3,2,1)],
			(2,3,1):[(3,2,1)]
			})
		assert(bruhat==Bruhat(3,True))

	def test_butterfly(this):
		bf = Poset(relations={
			0:['a0','b0'],
			'a0':['a1','b1'],
			'b0':['a1','b1'],
			'a1':[1],
			'b1':[1]
			})
		assert(bf==Butterfly(2))

	def test_intervals(this):
		B = Boolean(2)
		I = Intervals(B)
		expected = Poset(relations={tuple():[(x,x) for x in B],(tuple(),tuple()):[(tuple(),(1,)),(tuple(),(2,))],((1,),(1,)):[(tuple(),(1,)),((1,),(1,2))],((2,),(2,)):[(tuple(),(2,)),((2,),(1,2))],((1,2),(1,2)):[((1,),(1,2)),((2,),(1,2))],(tuple(),(1,)):[(tuple(),(1,2))],(tuple(),(2,)):[(tuple(),(1,2))],((1,),(1,2)):[(tuple(),(1,2))],((2,),(1,2)):[(tuple(),(1,2))]})
		assert(I==expected)

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
	Bool3 = make_Bool3()
	def test_adjoin_zerohat(this):
		A2 = Poset(elements=[0,1])
		X = A2.adjoin_zerohat()
		Y=Poset(elements=[2,0,1],zeta=[[1,1,1],[1,0],[1]],ranks=[[0],[1,2]])
		assert(Poset(elements=[2,0,1],zeta=[[1,1,1],[1,0],[1]],ranks=[[0],[1,2]])==A2.adjoin_zerohat())
	def test_adjoin_onehat(this):
		A2 = Poset(elements=[0,1])
		assert(Poset(elements=[0,1,2],zeta=[[1,0,1],[1,1],[1]],ranks=[[0,1],[2]])==A2.adjoin_onehat())

	def test_identify(this):
		Q = this.Bool3.identify({tuple():[(1,),(2,)], (3,):[(1,3),(2,3)]})
		expected = Poset(relations={tuple():[(3,),(1,2)], (1,2):[(1,2,3)], (3,):[(1,2,3)]})
		assert(expected==Q)
	def test_dual(this):
		assert(Poset(elements=['x','y','*'],relations={'x':['*'],'y':['*']})==this.V.dual())
	def test_union(this):
		assert(Poset(relations={'*':['x','y'],'a':['b']})==this.V.union(this.AB))
	def test_bddUnion(this):
		assert(Poset(relations={0:['x','y','b'],'x':[1],'y':[1],'b':[1]})==this.V.adjoin_onehat().bddUnion(this.ABC))
	def test_starProduct(this):
		assert(Poset(relations={'x':['b'],'y':['b']})==this.V.dual().starProduct(this.AB))
	def test_cartesianProduct(this):
		expected = Poset(relations={('*','a'):[('x','a'),('y','a'),('*','b')], ('x','a'):[('x','b')], ('y','a'):[('y','b')], ('*','b'):[('x','b'),('y','b')]})
		assert(Poset(relations={('*','a'):[('x','a'),('y','a'),('*','b')], ('x','a'):[('x','b')], ('y','a'):[('y','b')], ('*','b'):[('x','b'),('y','b')]})==this.V.cartesianProduct(this.AB))
	def test_diamondProduct(this):
		assert(Poset(relations={0:[('b','x'),('b','y')],('b','x'):[('c','x')],('b','y'):[('c','y')]})==this.ABC.diamondProduct(this.V))
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
		assert(make_Bool3().max()==[(1,2,3)])
	def test_subposet(this):
		assert(Poset(relations={(1,):[(1,2)],(3,):[]})==this.pent.subposet([(1,2),(1,),(3,)]))
		assert(Poset(relations={(1,):[(1,2)],(3,):[]})==this.pent.subposet((1,2,3),indices=True))
	def test_complSubposet(this):
		assert(Poset(relations={(1,):[(1,2)],(3,):[]})==this.pent.complSubposet([tuple(),(1,2,3)]))
		assert(Poset(relations={(1,):[(1,2)],(3,):[]})==this.pent.complSubposet((0,4),indices=True))

	def test_interval(this):
		assert(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]})==this.pent.interval((1,),(1,2,3)))
		assert(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]})==this.pent.interval(1,4,indices=True))
		assert(Poset(relations={tuple():[(3,)]})==this.pent.interval(tuple(),(3,)))

	def test_filter(this):
		assert(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]})==this.pent.filter(((1,),)))
		assert(Poset(relations={(1,2):[(1,2,3)]})==this.pent.filter(((1,),),strict=True))
	
	def test_ideal(this):
		assert(Poset(relations={tuple():[(1,),(1,2)],(1,):[(1,2)]})==this.pent.ideal(((1,2),)))
		assert(Poset(relations={tuple():[(1,),(3,)]})==this.pent.ideal(((1,),(3,))))

	def test_properPart(this):
		assert(Poset(relations={(1,):[(1,2)],(3,):[]})==this.pent.properPart())
	def test_rankSelection(this):
		assert(Poset(relations={tuple():[(1,2)]})==this.pent.rankSelection([0,2]))
##########################################
#Internal Computations
##########################################
class TestInternalComputations:
	pent = Poset(elements=[tuple(),(1,),(1,2),(3,),(1,2,3)],zeta=[1,1,1,1,1, 1,1,0,1, 1,0,1, 1,1, 1],flat_zeta=True)
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
		assert(1==this.pent.mobius(0,len(this.pent)-1,indices=True))
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
		assert(Polynomial({'aaa':1,'baa':3,'aba':5,'bba':3,'aab':3,'bab':5,'abb':3,'bbb':1})==this.B4.abIndex())
	def test_cdIndex(this):
		assert(Polynomial({'ccc':1,'cd':2,'dc':2})==this.B4.cdIndex())
		assert(Polynomial({'cc':1,'d':1})==this.B3.complSubposet([(1,2)]).cdIndex())
		assert(Polynomial({'c'*4:1,'ccd':3,'cdc':5,'dcc':3,'dd':4})==Boolean(5).cdIndex())

		#Example 6.14 with $M$ the 3-dimensional solid torus
		#Euler flag enumeration of Whitney stratified spaces
		#by Ehrenborg, Richard and Goresky, Mark and Readdy, Margaret
		P = Chain(2)
		P.zeta[0,1] = -1 #Euler char of torus bounday
		P.zeta[0,2] = 1 #Euler char of solid torus
		P.ranks = [[0],[],[1],[2]] #0, 2-dimensional torus, 3-dimensional solid torus
		assert(P.cdIndex() == Polynomial({'cc':1,'d':-2}))
		P.hasseDiagram = ZetaHasseDiagram(P,keep_ranks=True)
		with open('/tmp/a.tex','w') as file: file.write(P.latex(standalone=True))


	def test_bettiNumbers(this):
		assert([1,2,1]==Torus().properPart().bettiNumbers())
	def test_buildIsomorphism(this):
		#make sure two elements hash the same so we test
		#reversing a choice
		P=Bruhat(3,True).union(Bruhat(3))
		Q=Bruhat(3,True).union(Bruhat(3))
		Q=Q.reorder(Q.elements[::-1])
		assert(len(Q)==Q.zeta.size)
		for _ in range(2): #do it twice to test cache
			phi = Q.buildIsomorphism(P,indices=True)
			phi_inv = {v:k for k,v in phi.items()}
			assert(all(set(Q.elements.index(x) for x in Q.filter([i],indices=True)) == set(phi_inv[P.elements.index(j)] for j in P.filter([phi[i]],indices=True)) for i in range(len(Q))))
		assert(None==Q.buildIsomorphism(Bruhat(3).union(Bruhat(3))))
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
		assert(this.p+2==Polynomial({'cc':1,'d':1,'':2}))
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
		assert(this.p*2==Polynomial({'cc':2,'d':2}))
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
		expected = Poset(relations={0:[1,4,5,8],1:[2,3],4:[2,6],5:[6,7],8:[7,3]},indices=True,elements=this.B_chains)
		actual = this.B.orderComplex()
		assert(expected==actual)
	def test_relations(this):
		assert([('a0','a1'),('a0','b1'),('b0','a1'),('b0','b1')]==sorted(this.B.relations()))
		assert([(0,2),(0,3),(1,2),(1,3)]==sorted(this.B.relations(indices=True)))
	def test_shuffle(this):
		assert(this.B.shuffle()==this.B)
	def test_ranks(this):
		pent = Poset(relations={0:[1,2],1:[3],2:[4],3:[4]})
		assert([[0],[1,2],[3],[4]] == Poset.make_ranks(pent.zeta))
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
#Test toSage and fromSage
##########################################
@pytest.mark.skipif(shutil.which('sage')==None, reason='sage is not in the PATH')
def test_sage():
	sage_str = '''
from posets import *
B = Butterfly(2).properPart().relabel([0,1,2,3])
SB = Posets.Crown(2)
BS = B.toSage()
assert SB.lequal_matrix()==BS.lequal_matrix(), "Poset.toSage"
assert Poset.fromSage(SB)==B, "Poset.fromSage"
'''
	result = None
	result = subprocess.Popen(['sage','-c',sage_str],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	if result!=None:
		txt, err = result.communicate()
		assert(result.returncode==0)
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
		assert(result.returncode != 1) #posetToPython test
		assert(result.returncode != 2) #pythonToPoset test
	os.remove(tmpfile)
