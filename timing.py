'''
Run this script to generate/append to a table timing.csv of times to compute the cd-index.

Usage: timing.py Family [-s start] [-e end]

Computes the cd-index of Family(n) for n in [start,end] using the flag vectors algorithm and
the new summation formula and records the times in timing.csv. Use timing.r afterwards
to generate plots of the timings.

Arguments:
	Family - Name of the function in posets.py used to create the posets to test.
		The function must take one integer paramter e.g. Boolean, Butterfly, Cube, Uncrossing, Bruhat, Torus.
		Default is 'Boolean'.

	-s, --start - The start paremeter, the default is 3.

	-e, --end - The end parameter, the default is 5.
'''
import posets
import hot
import sys
import time
import os

#ensure csv has the column names
if not os.path.isfile('timing.csv'):
	with open('timing.csv','w') as file:
		file.write('Family,n,Method,Seconds\n')

#convenience method to grab parameters from command line
def getarg(default,short,long=None):
	if '-'+short in sys.argv:
		return sys.argv[sys.argv.index('-'+short)+1]
	if long==None: return default
	if '--'+long in sys.argv:
		return sys.argv[sys.argv.index('--'+long)+1]
	return default

#get the name of the function to generate posets
if len(sys.argv)<2 or sys.argv[1][0]=='-':
	Pos = posets.Boolean
else:
	Pos = getattr(posets,sys.argv[1])

#get start and end parameters
start = int(getarg(3,'s','start'))
end = int(getarg(5,'e','end'))

for n in range(start,end+1):
	print('\nn =',n)

	P = Pos(n)

	P.cache = {}

	t = time.perf_counter()
	psi2 = hot.cdIndex2(P)
	t = time.perf_counter()-t
	print('new summation formula',t)
	print('')

	with open('timing.csv','a') as file:
		file.write(Pos.__name__)
		file.write(',')
		file.write(str(n))
		file.write(',')
		file.write('new,')
		file.write(str(t))
		file.write('\n')

	P.cache  = {}
	t = time.perf_counter()
	psi3 = hot.cdIndex3(P)
	t = time.perf_counter()-t
	print('flag vector algorithm new poly class', t)

	with open('timing.csv','a') as file:
		file.write(Pos.__name__)
		file.write(',')
		file.write(str(n))
		file.write(',')
		file.write('old,')
		file.write(str(t))
		file.write('\n')

	assert(str(psi2)==str(psi3))

