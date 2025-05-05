#set up function make_fig inside of doc string executions
import sys
import os
sys.path.append('../src')
from posets import *

def make_fig(P,name,**kwargs):
	"""
	Write out the hasse diagram of P to a tex file in figures and compile if the pdf
	isn't already there.
	"""
	if not os.path.isdir('figures'): os.makedirs('figures')
	if os.path.isfile('figures/'+name+'.pdf'): return
	with open('figures/'+name+'.tex','w') as tex:
		tex.write(P.latex(standalone=True, **kwargs))
	os.system('pdflatex -output-directory=figures figures/'+name)
