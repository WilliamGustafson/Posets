r'''
\section{Introduction}

This module provides a class \verb|Poset| that encodes a finite
partially ordered set (poset). Most notably, this module can efficiently
compute flag vectors, the \av\bv-index and the \cv\dv-index.
Quasigraded posets, in the sense of \cite{ehrenborg-goresky-readdy-15}, can be encoded and the \av\bv-index and \cv\dv-index of quasigraded
posets can be computed. Latex code
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

Terminology and notation on posets generally follows \cite{stanley-12} and \cite{birkhoff-67}.

The full documentation for the current version can be \href{https://www.github.com/williamGustafson/posets/releases/latest/download/posets.pdf}{found here}.

\subsection{Installation}

Install with pip via \verb|python -m pip install posets|.
Alternatively, download the whl file \href{https://www.github.com/WilliamGustafson/posets/releases}{here} and install it with pip via \verb|python -m pip posets-*-py3-none-any.whl|.

\subsection{Usage}

Here we give an introduction to using the posets module.

In the code snippets below we assume the module is imported via

\verb|from posets import *|

Constructing a poset:
\begin{verbatim}P = Poset(relations={'':['a','b'],'a':['ab'],'b':['ab']})
Q = Poset(relations=[['','a','b'],['a','ab'],['b','ab']])
R = Poset(elements=['ab','a','b',''], less=lambda x,y: return x in y)
S = Poset(zeta = [[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]], elements=['','a','b','ab'])
\end{verbatim}

Built in examples (see page~\pageref{Built in posets}):
\begin{verbatim}
Boolean(3) #Boolean algebra of rank 3
Cube(3) #Face lattice of the 3-dimensional cube
Bruhat(3) #Bruhat order on symmetric group of order 3!
Bnq(n=3,q=2) #Lattice of subspaces of F_2^3
DistributiveLattice(P) #lattice of ideals of P
Intervals(P) #meet semilattice of intervals of P
\end{verbatim}
These examples come with default drawing methods, for example,
when making latex code by calling \verb|DistributiveLattice(P).latex()|
the resulting figure depicts elements of the lattice as
Hasse diagrams of $P$ with elements of the ideal highlighted
(again, see page~\pageref{Built in posets}). Note, you will have
to set the \verb|height|, \verb|width| and possibly \verb|nodescale|
parameters in order to get sensible output.


Two posets compare equal when they have the same
set of elements and the same zeta values (i.e. the same order relation with the same weights):
\begin{verbatim}P == Q and Q == R and R == S #True
P == Poset(relations={'':['a','b']}) #False
P == Poset(relations={'':['ab'],'a':['ab'],'b':['ab']}) #False
P == Poset(zeta=[[0,1,1,2],[0,0,0,3],[0,0,0,4],[0,0,0,0]],
	elements=['','a','b','ab']) #False
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
P.latex(standalone=True) #latex code for a
#standalone document: \documentclass{preview}...
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
#Create noncommutative polynomials from dictionaries,
#keys are monomials, values are coefficients
p=Polynomial({'ab':1})
q=Polynomial({'a':1,'b':1})

#get and set coefficients like a dictionary
q['a'] #1
q['x'] #0
p['ba'] = 1

#print latex
str(p) #ab+ba

#basic arithmetic, polynomials form an algebra
p+q #ab+ba+a+b
p*q #aba+ab^{2}+ba^{2}+bab
q*p #a^{2}b+aba+bab+b^{2}a
2*p #2ab+2ba
p**2 #abab+ab^{2}a+ba^{2}b+baba
p**(-1) #raises TypeError
p**q #raises TypeError

#substitutions and conversions
p.sub(q,'a') #ab+ba+2b^{2} substitute q for a in p
p.abToCd() #d rewrite a's and b's
#in terms of c=a+b and d=ab+ba when possible
Polynomial({'c':1,'d':1}).cdToAb() #a+b+ab+ba rewrite c's and d's
#in terms of a's and b's
\end{verbatim}

Converting posets to and from SageMath:
\begin{verbatim}
P.toSage() #Returns a SageMath class, must be run under sage
Poset.fromSage(Q) #Take a poset Q made with SageMath and return an instance of Poset
\end{verbatim}

Converting to and from Macaulay2:
\begin{verbatim}
-- In M2
load "convertPosets.m2" --Also loads Python and Posets packages
import "posets" --This module must be installed to system version of python
P = posets\@\@Boolean(3) --Calling python functions
pythonPosetToMac(P) --Returns an instance of the M2 class Posets
macPosetToPython(Q) --Take a poset made with M2 and return an
--instance of the python class Poset
\end{verbatim}

Quasigraded posets:
\begin{verbatim}
#Provide the zeta and rank functions explicitly
#To construct a 2-chain with top two elements rank 2 and 3
#and with zeta value -1 between minimum and the element covering it:
T = Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]])
\end{verbatim}
The poset \verb|T| above is from \cite[Example 6.14]{ehrenborg-goresky-readdy-15} with $M$ taken to
be the 3-dimensional solid torus.

You can calculate the flag vectors and the \cv\dv-index just as you would for a classical poset,
for example, \verb|T.cdIndex()| returns the polynomial $\cv^2-2\dv$.

When plotting a quasigraded poset by default only the underlying poset is shown with element heights
based on rank, the zeta values are not shown. If you wish to display the zeta values you can use
the class \verb|ZetaHasseDiagram| to draw a Hasse diagram of your poset with an element $p$ depicted as
the associated filter, namely the subposet $\{q:q\ge p\}$, and with elements of the filters labeled by the
corresponding zeta value. To do so, either construct the poset with \verb|hasse_class=ZetaHasseDiagram|
such as in \verb|Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]],hasse_class=ZetaHasseDiagram)| or
set the Hasse diagram attribute on the poset as below:
\begin{verbatim}
T = Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]])
T.hasseDiagram = ZetaHasseDiagram(T)
\end{verbatim}
You can also represent elements with ideals instead of filters by passing \verb|filters=False|.
See \verb|ZetaHasseDiagram| and \verb|SubposetsHasseDiagram| for a thorough explanation of the options.
@is_section@sections_order@Poset@PosetIsoClass@Genlatt@HasseDiagram@SubposetsHasseDiagram@Built in posets@Polynomial@@
'''
from .poset import *
from .hasseDiagram import *
from .examples import *
from .polynomial import *

for x in ('poset','hasseDiagram','examples','polynomial'):
	if x in globals():
		del x
