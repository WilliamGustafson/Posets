#!/usr/bin/env python
'''
Usage: posets_test.py [-l LOG_FILE] [-a] [-c] [-s] [-m]

This module when executed tests the posets module.

Arguments:
	-a Run all tests, this is the default behavior.

	-skip Reverses the effect of all test flags, specified tests are skipped instead of ran.

	-c Continue on failure, printing a message on failure or logging if -l is provided.
		The default is to raise an exception on any failure and exit.

	-l Instead of printing messages log failure messages to LOG_FILE. The default is to print messages.

	-m Test Macauly2 functions to convert between Macaulay2 and python posets. Requires installation
		of Macaulay2 along with the Python and Posets packages for Macaulay2.

	-s Test toSage and fromSage, requires sage to be installed and in the path. Default
		is not to test these functions.

	-co Run constructor options tests

	-ex Run examples tests

	-op Run operations tests

	-qu Run Queries tests

	-sp Run subposet selection tests

	-ic Run internal computations tests

	-in Run invariant computations tests

	-mi Run miscellaneous tests
'''
##########################################
#general utilities setup
##########################################
from posets import *
import sys
import subprocess
import os

success = True
test_flags = ['-'+f for f in ['s','m','co','ex','op','qu','sp','ic','in','mi']]
tests = [] #tests to run

if '-skip' in sys.argv:
	for tf in test_flags:
		if tf not in sys.argv: tests.append(tf)

if '-a' in sys.argv or not any(tf in sys.argv for tf in test_flags):
	tests = test_flags.copy()

if '-l' in sys.argv:
	log_file = open(sys.argv[sys.argv.index('-l')+1],'w')
	def out(*args,end='\n'):
		log_file.write(*args)
		log_file.write('\n')
else:
	def out(*args,end=''):
		print(*args,end)

if '-c' in sys.argv:
	def failure(s):
		global success
		success = False
		out(s)
else:
	def failure(s):
		global success
		success = False
		out(s)
		raise Exception(s)

#methods to check a condition and throw an exception if false
def test(bool, name="Unnamed"):
	if not bool:
		failure(name+" test failed")

def testeq(actual, expected, name="Unnamed"):
	if expected!=actual:
		failure(name+" equality test failed\nactual="+repr(actual)+'\n\nexpected='+repr(expected))

def testeq_exit(actual, expected, name="Unnamed"):
	try:
		testeq(actual, expected, name)
	except Exception as e:
		print(e)
		sys.exit(1)

def testneq(actual, expected, name="Unnamed"):
	if expected == actual:
		failure(name+" inequality test failed\nactual="+repr(actual)+'\n\nexpected='+repr(expected))

def finish():
	if success:
		out('All tests passed.')
	if '-l' in sys.argv:
		log_file.close()

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

if __name__ == '__main__':
	test_timer = Timer()
	all_timer = Timer()
	##########################################
	#test constructor options
	##########################################
	if '-co' in tests:
		test_timer.reset()
		Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])

		bool3 = P(Poset(relations=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[4,5],[4,6],[3,7],[5,7],[6,7]], elements = Bool3.elements, indices=True))
		testeq(Bool3, bool3, "B_3 constructed with relations=list indices=True")

		bool3 = P(Poset(relations={0:[1,2,4],1:[3,5],2:[3,6],4:[5,6],5:[7],6:[7],3:[7]}, indices=True, elements=bool3.elements))
		testeq(Bool3, bool3, "B_3 constructed with relations=dict indices=True")

		b3_rels_list = [[tuple(),(1,)],[tuple(),(2,)],[tuple(),(3,)],[(1,),(1,2)],[(1,),(1,3)],[(2,),(1,2)],[(2,),(2,3)],[(3,),(1,3)],[(3,),(2,3)],[(1,2),(1,2,3)],[(1,3),(1,2,3)],[(2,3),(1,2,3)]]
		bool3 = P(Poset(relations=b3_rels_list, indices=False, elements=bool3.elements))
		testeq(Bool3, bool3, "B_3 constructed with relations=list indices=False")

		b3_rels_dict={tuple():[(1,),(2,),(3,)],(1,):[(1,2),(1,3)],(2,):[(1,2),(2,3)],(3,):[(1,3),(2,3)],(1,3):[(1,2,3)],(2,3):[(1,2,3)],(1,2):[(1,2,3)]}
		bool3 = P(Poset(relations=b3_rels_dict, elements=bool3.elements, indices=False))
		testeq(Bool3, bool3, "B_3 constructed with relations=dict indices=True")

		bool3 = P(Bool3.reorder(Bool3.elements[::-1]))
		testneq(Bool3, bool3, "Reorder indices=False")

		bool3 = P(Bool3.reorder(list(range(len(Bool3)))[::-1], indices=True))
		testneq(Bool3, bool3, "Reorder indices=True")
		testeq(Bool3, Bool3.reorder(list(range(len(Bool3)))[::-1], indices=True).sort(), "Sort")

		bool3 = P(Poset(relations=b3_rels_list).sort())
		testeq(Bool3.sort(), bool3, "B_3 constructed with relations=list elements=None")

		bool3 = P(Poset(relations=b3_rels_dict).sort())
		testeq(Bool3.sort(), bool3, "B_3 constructed with relations=dict elements=None")

		a4 = P(elements=[0,1,2,3],incMat=[[0]*4 for i in range(4)],ranks=[[0,1,2,3]])
		testeq(Poset(elements=[0,1,2,3]), a4, "A_4")


		test_timer.stop();print("Finished constructor options tests",test_timer)
	##########################################
	#test example posets
	##########################################
	if '-ex' in tests:
		test_timer.reset()

		Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])
		chain3 = P([[0,1,1,1],[-1,0,1,1],[-1,-1,0,1],[-1,-1,-1,0]], list(range(4)), [[i] for i in range(4)])
		testeq(Chain(3), chain3, "Chain")

		testeq(Boolean(3), Bool3, "Boolean")

		square = P(Poset(relations={1:[(1,2),(1,4)],2:[(1,2),(2,3)],3:[(2,3),(3,4)],4:[(3,4),(1,4)]}).adjoin_zerohat().adjoin_onehat().sort(key=str))
		testeq(Polygon(4).sort(key=str), square, "Polygon")

		square = P(Poset(
			relations={'00':['*0','0*'],'10':['*0','1*'],'01':['*1','0*'],'11':['*1','1*'],'0*':['**'],'*0':['**'],'1*':['**'],'*1':['**']}
			).adjoin_zerohat().sort(key=str))
		testeq(Cube(2).sort(key=str), square, "Cube")

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
		testeq(Torus().sort(key=str), torus, "Torus")

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
		testeq(Grid(2,[2,1]).sort(key=str), grid, "Grid")

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

		testeq(KleinBottle().sort(key=str), kb, "Klein Bottle")

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
		testeq(ProjectiveSpace().sort(key=str), ps, "Projective Plane")



		uc3 = P(Poset(
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
				'(3,4)(2,5)(1,6)','(2,3)(4,5)(1,6)','(2,3)(1,4)(5,6)','(1,2)(4,5)(3,6)','(1,2)(3,4)(5,6)',
				'(2,4)(3,5)(1,6)','(3,4)(1,5)(2,6)','(2,3)(1,5)(4,6)','(1,3)(4,5)(2,6)','(1,3)(2,4)(5,6)','(1,2)(3,5)(4,6)',
				'(2,4)(1,5)(3,6)','(1,4)(3,5)(2,6)','(1,3)(2,5)(4,6)',
				'(1,4)(2,5)(3,6)'
				]
			).sort(key=str))
		testeq(Uncrossing(3).sort(key=str), uc3, "Uncrossing")

		#B_3(2)
		b32 = P(Poset(
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
			elements=[1,
				1|1<<1, 1|1<<2, 1|1<<3, 1|1<<4, 1|1<<5, 1|1<<6, 1|1<<7,
				1|1<<1|1<<2|1<<3, 1|1<<1|1<<4|1<<5, 1|1<<2|1<<4|1<<6,
				1|1<<3|1<<4|1<<7, 1|1<<1|1<<6|1<<7, 1|1<<3|1<<5|1<<6,
				1|1<<2|1<<5|1<<7,
				1|1<<1|1<<2|1<<3|1<<4|1<<5|1<<6|1<<7
				]
			).sort())
		testeq(Bnq(n=3,q=2).sort(), b32, "B_3(2)")

		triFlats = P(Poset(
			relations={
				(('a',),('b',),('c',)):[(('a','b'),('c',)), (('a','c'),('b',)), (('a',),('b','c'))],
				(('a','b'),('c',)): [(('a','b','c'),)],
				(('a','c'),('b',)): [(('a','b','c'),)],
				(('a',),('b','c')): [(('a','b','c'),)]
				}).sort())
		testeq(LatticeOfFlats([['a','b'],['b','c'],['c','a']]).sort(), triFlats, "Triangle lattice of flats")

		pentFlats = P(Poset(
			relations={
				tuple():[(1,),(3,)],
				(1,):[(1,2,)],
				(1,2,):[(1,2,3)],
				(3,):[(1,2,3)]
				}
			).sort())
		testeq(LatticeOfFlats([0,1,2,2,1,3,3,3]).sort(), pentFlats, "Pentagon lattice (via LatticeOfFlats)")

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
		testeq(DistributiveLattice(V).sort(key=str), Videals, "Lattice of ideals")

		pent = Poset(relations={0:['a','c'],'a':['b'],'c':[1],'b':[1]},elements=[0,'a','b','c',1])
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
			).sort(key=str))
		MinorPoset(pent).sort(key=str)
		testeq(MinorPoset(pent).sort(key=str), pentMinors, "Minor poset of pentagon")

		bruhat = P(Poset(relations={
			(1,2,3):[(1,3,2),(2,1,3)],
			(2,1,3):[(3,1,2),(2,3,1)],
			(1,3,2):[(3,1,2),(2,3,1)],
			(3,1,2):[(3,2,1)],
			(2,3,1):[(3,2,1)]
			}).sort())
		testeq(Bruhat(3).sort(), bruhat, "Bruhat order")

		bf = P(Poset(relations={
			0:['a0','b0'],
			'a0':['a1','b1'],
			'b0':['a1','b1'],
			'a1':[1],
			'b1':[1]
			}).sort(key=str))
		testeq(Butterfly(2).sort(key=str), bf, "Butterfly")

		test_timer.stop();print('Finished examples tests',test_timer)
	##########################################
	#test operations on posets
	##########################################
	if '-op' in tests:
		test_timer.reset()

		Bool3 = Poset(incMat = [[0,1,1,1,1,1,1,1],[-1,0,0,1,0,1,0,1],[-1,0,0,1,0,0,1,1],[-1,-1,-1,0,0,0,0,1],[-1,0,0,0,0,1,1,1],[-1,-1,0,0,-1,0,0,1],[-1,0,-1,0,-1,0,0,1],[-1,-1,-1,-1,-1,-1,-1,0]], elements = [tuple(),(1,),(2,),(1,2),(3,),(1,3),(2,3),(1,2,3)],ranks=[[0],[1,2,4],[3,5,6],[7]])
		A2 = Poset(elements=[0,1])
		testeq(A2.adjoin_zerohat(), P(elements=[2,0,1],incMat=[[0,1,1],[-1,0,0],[-1,0,0]],ranks=[[0],[1,2]]), "adjoin_zerohat test")
		testeq(A2.adjoin_onehat(), P(elements=[2,0,1],incMat=[[0,-1,-1],[1,0,0],[1,0,0]],ranks=[[1,2],[0]]), "adjoin_onehat test")

		Q = Bool3.identify({tuple():[(1,),(2,)], (3,):[(1,3),(2,3)]})
		expected = P(Poset(relations={tuple():[(3,),(1,2)], (1,2):[(1,2,3)], (3,):[(1,2,3)]}))
		testeq(Q, expected, "identify test")

		V = Poset(relations={'*':['x','y']})
		AB = Poset(relations={'a':['b']})
		ABC = Poset(relations={'a':['b'],'b':['c']})

		#dual
		testeq(V.dual().sort(), P(Poset(relations={'x':['*'],'y':['*']}).sort()),"dual")

		#union
		testeq(V.union(AB).sort(), P(Poset(relations={'*':['x','y'],'a':['b']}).sort()),"union")

		#bddUnion
		testeq(V.adjoin_onehat().bddUnion(ABC).sort(key=str), P(Poset(relations={0:['x','y','b'],'x':[1],'y':[1],'b':[1]}).sort(key=str)),"bddUnion")

		#starProduct
		testeq(V.dual().starProduct(AB).sort(key=str), P(Poset(relations={'x':['b'],'y':['b']}).sort(key=str)),'starProduct')

		#cartesianProduct
		testeq(V.cartesianProduct(AB).sort(key=str), P(Poset(relations={('*','a'):[('x','a'),('y','a'),('*','b')], ('x','a'):[('x','b')], ('y','a'):[('y','b')], ('*','b'):[('x','b'),('y','b')]}).sort(key=str)), "cartesianProduct")

		#diamondProduct
		testeq(ABC.diamondProduct(V).sort(key=str), P(Poset(relations={0:[('b','x'),('b','y')],('b','x'):[('c','x')], ('b','y'):[('c','y')]}).sort(key=str)), "diamondProduct")

		test_timer.stop();print('Finished operations tests',test_timer)
	##########################################
	#Queries
	##########################################
	if '-qu' in tests:
		test_timer.reset()

		pent = LatticeOfFlats([0,1,2,2,1,3,3,3])
		B3 = Boolean(3)

		test(not pent.isRanked(), "isRanked False")
		test(B3.isRanked(), "isRanked True")

		test(not pent.isEulerian(), "isEulerian False")
		test(B3.isEulerian(), "isEulerian True")

		test(not pent.isGorenstein(), 'isGorenstein False')
		test(B3.isGorenstein(), 'isGorenstein True')

		testeq(B3.covers(), {tuple():[(1,),(2,),(3,)], (1,):[(1,2),(1,3)], (2,):[(1,2),(2,3)], (3,):[(1,3),(2,3)], (1,2):[(1,2,3)], (2,3):[(1,2,3)], (1,3):[(1,2,3)]}, "covers")
		testeq(pent.covers(), {tuple():[(1,),(3,)], (1,):[(1,2)], (3,):[(1,2,3)], (1,2):[(1,2,3)]})

		test_timer.stop();print('Finished queries tests', test_timer)
	##########################################
	#Subposet Selection
	##########################################
	if '-sp' in tests:
		test_timer.reset()

		V = Poset(relations={'*':['x','y']})
		pent = LatticeOfFlats([0,1,2,2,1,3,3,3])

		testeq(V.min(),['*'], 'min')
		testeq(sorted(V.max()),['x','y'], 'max')

		testeq(pent.subposet([(1,2),(1,),(3,)]).sort(), P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort()), 'subposet')
		testeq(pent.complSubposet([tuple(),(1,2,3)]).sort(), P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort()), 'subposet')

		testeq(pent.subposet((1,2,3),indices=True).sort(), P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort()), 'subposet indices=True')
		testeq(pent.complSubposet((0,4),indices=True).sort(), P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort()), 'subposet indices=True')

		testeq(pent.interval((1,),(1,2,3)).sort(), P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort()), "interval")
		testeq(pent.interval(1,4,indices=True).sort(), P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort()), "interval indices=True")

		testeq(pent.filter(((1,),)).sort(), P(Poset(relations={(1,):[(1,2)],(1,2):[(1,2,3)]}).sort()), "filter")
		testeq(pent.filter(((1,),),strict=True).sort(), P(Poset(relations={(1,2):[(1,2,3)]}).sort()), "filter strict=True")
		testeq(pent.filter((1,),indices=True).sort(), P(Poset(relations={1:[2],2:[4]}).sort()), "filter indices=True")

		testeq(pent.properPart().sort(), P(Poset(relations={(1,):[(1,2)],(3,):[]}).sort()), 'propPart')

		testeq(pent.rankSelection([0,2]).sort(), P(Poset(relations={tuple():[(1,2)]}).sort()), 'rankSelection')

		test_timer.stop();print('Finished subposet selection tests',test_timer)
	##########################################
	#Internal Computations
	##########################################
	if '-ic' in tests:
		test_timer.reset()

		pent = LatticeOfFlats([0,1,2,2,1,3,3,3])

		test(not pent.less((1,),(1,)), 'less False (equality)')
		test(not pent.less((1,),(3,)), 'less False (incomprable)')
		test(not pent.less((1,2,),(1,)), 'less False (greater)')
		test(pent.less((1,),(1,2,)), 'less True')

		test(not pent.isAntichain([(1,),(1,2,)]), 'isAntichain False')
		test(pent.isAntichain([(1,),(3,)]), 'isAntichain True')
		test(pent.isAntichain([(1,)]), 'isAntichain singleton')
		test(pent.isAntichain([]), 'isAntichain empty set')
		test(not pent.isAntichain([1,2],indices=True), 'isAntichain False indices=True')
		test(pent.isAntichain([1,3],indices=True), 'isAntichain True indices=True')

		testeq(pent.join((1,),(1,)), (1,), 'join self')
		testeq(pent.join((1,),(1,2,)), (1,2,), 'join less')
		testeq(pent.join((1,),(3,)), (1,2,3), 'join')
		testeq(Poset(elements=[0,1]).join(0,1), None, 'join None')

		testeq(pent.mobius(), 1, 'mobius no arguments')
		testeq(pent.mobius(tuple(),(1,2)), 0, 'mobius')

		testeq(tuple(pent.rank(p) for p in sorted(pent.elements)), (0,1,2,3,1), 'rank')

		test_timer.stop();print('Finished internal computation tests',test_timer)
	##########################################
	#Invariants
	##########################################
	if '-in' in tests:
		test_timer.reset()

		B4 = Boolean(4)
		B3 = Boolean(3)
		testeq(B4.flagVectors(), [[[],1,1],[[1],4,3],[[2],6,5],[[1,2],12,3],[[3],4,3],[[1,3],12,5],[[2,3],12,3],[[1,2,3],24,1]], 'flagVectors')
		testeq(B4.abIndex(), Polynomial([[1,'aaa'],[3,'baa'],[5,'aba'],[3,'bba'],[3,'aab'],[5,'bab'],[3,'abb'],[1,'bbb']]), 'abIndex')
		testeq(B4.cdIndex(), Polynomial([[1,'ccc'],[2,'cd'],[2,'dc']]), 'cdIndex')
		testeq(B3.complSubposet([(1,2)]).cdIndex(), Polynomial([[1,'aa'],[1,'ab'],[2,'ba'],[0,'bb']]), 'cdIndex non-DS')
		testeq(B4.cdIndex_IA(), B4.cdIndex(), 'cdIndex_IA')
		testeq(B3.cd_coeff_mat('d').tolist(), [[-2,1,1,0,1,0,0,0]]+[[0]*8]*7, 'cd_coeff_mat')
		testeq(B3.cd_op('d',[[0],[1]]+[[0] for i in range(6)]), {tuple():1}, 'cd_op')

		testeq(Boolean(2).zeta(), [[1,1,1,1],[0,1,0,1],[0,0,1,1],[0,0,0,1]], 'zeta')

		testeq(Torus().properPart().bettiNumbers(), [1,2,1], 'bettiNumbers')

		B2 = Boolean(2)
		B = Boolean(2)
		B.reorder(B.elements[::-1])
		phi = B.buildIsomorphism(B2)
		phi_inv = {v:k for k,v in phi.items()}
		test(all(set(B.filter([i],indices=True)) == set(phi_inv[j] for j in B2.filter([phi[i]],indices=True)) for i in range(len(B))), "buildIsomorphism")

		test_timer.stop();print('Finished invariant tests',test_timer)

	##########################################
	#Misc
	##########################################
	if '-mi' in tests:
		test_timer.reset()

		B = Butterfly(2).properPart()
		B_chains = [tuple(),('a0',),('a0','a1'),('a0','b1'),('a1',),('b0',),('b0','a1'),('b0','b1'),('b1',)]
		testeq(sorted(B.chains()), B_chains, "chains")

		testeq(B.orderComplex().sort(), P(Poset(relations={0:[1,4,5,8],1:[2,3],4:[2,6],5:[6,7],8:[7,3]},indices=True,elements=B_chains)))

		testeq(sorted(B.relations()), [('a0','a1'),('a0','b1'),('b0','a1'),('b0','b1')], 'relations')
		testeq(sorted(B.sort().relations(indices=True)), [(0,1),(0,3),(2,1),(2,3)], 'relations indices=True')

		testeq(B.reorder(['b1','a0','b0','a1']), P(Poset(elements=['b1','a0','b0','a1'],incMat=[[0,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,0]])), 'reorder')

		testeq(B.sort().reorder([3,0,2,1],indices=True), P(Poset(elements=['b1','a0','b0','a1'],incMat=[[0,0,0,0],[1,0,0,1],[1,0,0,1],[0,0,0,0]])), 'reorder indices=True')

		testeq(B.reorder(B.elements[::-1]).sort(), P(Poset(elements=['a0','a1','b0','b1'],relations={0:[1,3],2:[1,3]},indices=True)), "sort")

		test(B.shuffle().is_isomorphic(B), "shuffle")

		pent = LatticeOfFlats([0,1,2,2,1,3,3,3])
		testeq(Poset(relations=pent.relations()).sort().ranks, [[0],[1,4],[2],[3]], "ranks")

		test_timer.stop();print('Finished miscellaneous tests',test_timer)
	##########################################
	#Test toSage and fromSage
	##########################################
	if '-s' in tests:
		test_timer.reset()
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
		with open(sys.argv[0]+'.sage','w') as file:
			file.write(sage_str)

		result = None
		try:
			result = subprocess.Popen(['sage',sys.argv[0]+'.sage'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		except:
			print('Couldn\'t test fromSage and toSage')
			print('Unexpected return code from sage process, perhaps you don\'t have sage installed.')
		if result!=None:
			txt, err = result.communicate()
			if result.returncode == 1:
				if b'fromSage' in err:
					failure(err.decode('ascii')+'\nfromSage test failed')
				else:
					failure(err.decode('ascii')+'\ntoSage test failed')
		os.remove(sys.argv[0]+'.sage')
		os.remove(sys.argv[0]+'.sage.py')

		test_timer.stop();print('Finished Sage tests',test_timer)

	##########################################
	#Test pythonPosetToMac and macPosetToPython
	#from convertPosets.m2
	##########################################
	if '-m' in tests:
		test_timer.reset()

		mac_str = '''
load "convertPosets.m2"

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
		with open(sys.argv[0]+'.m2','w') as file:
			file.write(mac_str)

		result = None
		try:
			result = subprocess.Popen(['M2','--script',sys.argv[0]+'.m2'],stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		except:
			print('Couldn\'t test pythonPosetToMac and macPoseToPython')
			print('Unexpected return code from M2 process, perhaps you don\'t have M2 installed.')
		if result!=None:
			txt, err = result.communicate()
			if result.returncode == 1:
				failure(err.decode('ascii')+'\nmacPosetToPython test failed')
			elif result.returncode == 2:
				failure(err.decode('ascii')+'\npythonPosetToMac test failed')
		os.remove(sys.argv[0]+'.m2')

		test_timer.stop();print('Finished Macaulay2 tests',test_timer)

	finish()
	all_timer.stop();print('Total time',all_timer)
