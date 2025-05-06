r'''
@is_section@
@sections_order@Poset@PosetIsoClass@Genlatt@HasseDiagram@SubposetsHasseDiagram@Built in posets@Polynomial@@
\section{Introduction}

This module provides a class \verb|Poset| that encodes a finite
partially ordered set (poset). Most notably, this module can efficiently
compute flag vectors, the \av\bv-index and the \cv\dv-index.
Qausigraded posets, in the sense of \cite{ehrenborg-goresky-readdy-15}, can be encoded and the \av\bv-index and \cv\dv-index of quasigraded
psoets can be computed. Latex code
for Hasse diagrams can be produced with a very flexible interface.
There are
methods for common operations and constructions such as Cartesian products,
disjoint unions, interval lattices, lattice of ideals, etc. Various examples
of posets are provided such as Boolean algebras, the face lattice of the
$n$-dimensional cube, (noncrossing) partition lattices, the type $A_n$ Bruhat
and weak orders, uncrossing orders etc. General subposets can be
selected as well as particular ones of interest such as intervals and
rank selections. Posets from this
module can also be converted to and from posets from \href{https://www.sagemath.org}{sagemath} and \href{https://www.macaulay2.com/}{Macaulay2}.

\subsection{Installation}

Download the wheel file \href{https://www.github.com/WilliamGustafson/posets/releases}{here} and install it with pip via \verb|python -m pip posets-*-py3-none-any.whl|.

\subsection{Building}

Building requires \href{https://hatch.pypa.io}{hatch} to be installed.
After cloning the repository from the base directory of the repository
run \verb|hatch build| to build distribution files and then
run \begin{verbatim}python -m pip install dist/posets-*-py3-none-any.whl\end{verbatim}
to install the wheel file.

The documentation can be built in pdf form by running
\verb|make docs| from the base directory.
Compilation requires \LaTeX to be installed with the packages pgf/tikz,
graphicx, fancyvrb, amsmath, amsssymb, scrextend, mdframed and hyperref
as well as the python module \verb|pydox|.
\verb|pydox| can be obtained from \url{github.com/WilliamGustafson/pydox.git}.
If \verb|pydox| is not on
your path either make a symlink to it or call make with
\verb|make PYDOX=[path to pydox] docs|.

You can run the tests via \verb|make test|, this requires \verb|pytest|
to be installed. You can output an html coverage report, located
at \verb|tests/htmlcov/index.html|, with \verb|make coverage|.
Making the coverage report requires \href{https://pytest.org}{pytest}, \href{https://coverage.readthedocs.io}{coverage} and the \href{https://pytest-cov.readthedocs.io}{pytest-cov} plugin. Note, coverage on \verb|hasseDiagram.py| is very low because testing
drawing functions cannot be easily automated.

\subsection{Quick start}
Here we give a quick introduction to using the posets module.
In this subsection python commands and outputs are denoted
in typewriter font.

In the code snippets below we assume the module is imported via

\verb|from posets import *|

Constructing a poset:
\begin{verbatim}P = Poset(relations={'':['a','b'],'a':['ab'],'b':['ab']})
Q = Poset(relations=[['','a','b'],['a','ab'],['b','ab']])
R = Poset(elements=['ab','a','b',''], less=lambda x,y: return x in y)
S = Poset(incMat = [[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]], elements=['','a','b','ab'])
\end{verbatim}

Built in examples (see page~\pageref{Built in posets}):
\begin{verbatim}
Boolean(3) #Boolean algebra of rank 3
Cube(3) #Face lattice of the 3-dimensional cube
Bruhat(3) #Bruhat order on symmetric group of order 3!
Bnq(n=3,q=2) #Lattice of subspaces of F_2^3
DistributiveLattice(P) #lattice of ideals of P
Intervals(P) #lattice of intervals of P (including the empty interval)
\end{verbatim}
These examples come with default drawing methods, for example,
when making latex code by calling \verb|DistributiveLattice(P).latex()|
the resulting figure depicts elements of the lattice as
Hasse diagrams of $P$ with elements of the ideal highlighted
(again, see page~\pageref{Built in posets}). Note, you will have
to set the \verb|height|, \verb|width| and possibly \verb|nodescale|
parameters in order to get sensible output\footnote{A future update
to further automate this is planned.}.


Two posets compare equal when they have the same
set of elements and the same order relation on them:
\begin{verbatim}P == Q and Q == R and R == S #True
P == Poset(relations={'':['a','b']}) #False
P == Poset(relations={'':['ab'],'a':['ab'],'b':['ab']}) #False
\end{verbatim}

Use \verb|is_isomorphic| or \verb|PosetIsoClass| to check whether
posets are isomorphic:
\begin{verbatim}P.is_isomorphic(Boolean(2)) #True
P.isoClass()==Boolean(2).isoClass() #True
P.is_isomorphic(Poset(relations={'':['a','b']})) #False
\end{verbatim}

Viewing and creating Hasse diagrams:
\begin{verbatim}
P.show() #displays a Hasse diagram in a new window
P.latex() #returns latex code: \begin{tikzpicture}...
P.latex(standalone=True) #latex code for a standalone document: \documentclass{preview}...
display(P.img()) #Display a poset when in a Jupyter notebook
#this uses the output of latex()
\end{verbatim}

Computing invariants:
\begin{verbatim}
Cube(2).fVector() #{(): 1, (1,): 4, (2,): 4, (1, 2): 8}
Cube(2).hVector() #{(): 1, (1,): 3, (2,): 3, (1, 2): 1}
Boolean(5).sparseKVector() #{(3,): 8, (2,): 8, (1, 3): 4, (1,): 3, (): 1}
Boolean(5).cdIndex() #Polynomial({'ccd': 3, 'cdc': 5, 'dd': 4, 'dcc': 3, 'cccc': 1})
print(Boolean(5).cdIndex()) #c^{4}+3c^{2}d+5cdc+3dc^{2}+4d^{2}
\end{verbatim}

Polynomial operations:
\begin{verbatim}
#Create polynomials from dictionaries, keys are monomials, values are coefficients
p=Polynomial({'ab':1})
q=Polynomial({'a':1,'b':1})
#get and set coefficients like a dictionary
q['a'] #1
q['x'] #0
p['ba'] = 1
str(p) #ab+ba string method returns latex
p+q #ab+ba+a+b
#multiplication is non-commutative
p*q #aba+ab^{2}+ba^{2}+bab
q*p #a^{2}b+aba+bab+b^{2}a
2*p #2ab+2ba
p**2 #abab+ab^{2}a+ba^{2}b+baba non-negative integer exponentation only
p**(-1) #raises NotImplementedError
p**q #raises NotImplementedError
p.sub(q,'a') #ab+ba+2b^{2} substitute q for a in p
p.abToCd() #d rewrite a's and b's in terms of c=a+b and d=ab+ba when possible
Polynomial({'c':1,'d':1}).cdToAb() #a+b+ab+ba rewrite c's and d's in terms of a's and b's
\end{verbatim}

Converting posets to and from SageMath:
\begin{verbatim}
P.toSage() #Returns a SageMath class, must be run under sage
fromSage(Q) #Take a poset Q made with SageMath and return an instance of Poset
\end{verbatim}

Converting to and from Macaulay2:
\begin{verbatim}
-- In M2
load "convertPosets.m2" --Also loads Python and Posets packages
import "posets" --This module must be installed to system version of python
P = posets@@Boolean(3) --Calling python functions
pythonPosetToMac(P) --Returns an instance of the M2 class Posets
macPosetToPython(Q) --Take a poset made with M2 and return an
--instance of the python class Poset
\end{verbatim}
'''
from .poset import *
from .hasseDiagram import *
from .examples import *
from .polynomial import *

for x in ('poset','hasseDiagram','examples','polynomial'):
	if x in globals():
		del x
