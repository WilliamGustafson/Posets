from posets import *
import os
import sys

if not os.path.isfile('snowman.csv'):
	with open('snowman.csv','w') as file:
		file.write('n,m,Psi\n')

if len(sys.argv)<5:
	print('Usage: snowman.py start_n start_m end_n end_m')
	sys.exit()

sn,sm,en,em = tuple(int(x) for x in sys.argv[1:5])

for n,m in itertools.product(range(sn,en+1),range(sm,em+1)):
	print('n =',n,'m =',m)
	psi = Snowman(n,m).cdIndex()
	with open('snowman.csv','a') as file:
		file.write(','.join((str(n), str(m), str(psi)))+'\n')
