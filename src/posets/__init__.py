r'''
@is_section@exec@version='0.0.1'@
This module provides a class \verb|Poset| that encodes a finite
partially ordered set (poset). The class provides methods to construct
new posets via operations such as Cartesian products and disjoint unions,
select subposets and compute invariants such as flag vectors, the \av\bv-index
and the \cv\dv-index. There is also a class \verb|PosetIsoClass| that
encodes an isomorphism class of a poset.

\subsection{Installation}

After cloning the repository from the root directory
run \verb|hatch build| to build distribution files and then
\verb|python -m pip install dist/posets-@eval@version@-py3-none-any.whl|
to install the built wheel file.

\subsection{Overview}
Here we give a quick introduction to using the posets module by way of examples.

First import the module.

\verb|from posets import *|

You can construct a poset in several ways, by specifying the relations
either as a list or dictionary, by providing a function \verb|less|
that returns a Boolean value or by providing an incidence matrix.
You can also copy a poset by passing a \verb|Poset| object to the constructor.
The \verb|Poset| documentation contains a full explanation, below is
an example of constructing the same poset in four ways.
\begin{center}\begin{verbatim}P = Poset(relations={'a':'ab','b':'ab'})
P = Poset(relations=[['a','ab'],['b','ab']])
P = Poset(elements=['a','b','ab'], less=lambda x,y: return x in y and x!=y)
P = Poset(incMat = [[0,0,1],[0,0,1],[0,0,0]], elements=['a','b','ab'])
\end{verbatim}\end{center}

Printing a poset via \verb|print(P)| will list the elements, the zeta function (a matrix $\zeta$ with entries $\zeta_{i,j}=1$ if $i\le j$ and 0 otherwise as well as a list \verb|ranks|; \verb|ranks[i]| is a list of all indices \verb|j| such that \verb|elements[j]| is length \verb|i|
(the length if $p\in P$ is the length of the longest chain in $P$ with maximum $p$).

You can display the Hasse diagram of a poset in a new window with \verb|P.show()|
or generate tikz code with \verb|P.latex()|. Both methods take keyword arguments to control the output, e.g. \verb|height|, \verb|width|, \verb|nodescale|.
The \verb|latex| method allows for finer grain control of the output compared to \verb|show| but must be compiled before viewing. Note the aesthetics of larger posets are heavily impacted by the ordering of \verb|elements|.

This module contains various examples of posets, e.g.
\verb|Boolean|, \verb|Cube|, and \verb|Bruhat|. For example,
you can construct
the poset of all subsets of $\{1,2,3,4,5\}$ via \verb|B5 = Boolean(5)|.

Various operations are given as methods you call on a \verb|Poset| object
For example, to construct the face poset of a triangle crossed with a
square use the \verb|diamondProduct| method:
\verb|Q = Boolean(3).diamondProduct(Cube(2))|.
See the operations subsection for a complete list.

Two posets can be compared for equality with the same meaning as in
mathematics (the same underlying set with the same order). For
example
\begin{center}\begin{verbatim}P == B5 #False
P == Poset(elements=['ab','b','a'],incMat=[[0,0,0],[1,0,0],[1,0,0]]) #True
P == Poset(relations={'A':'AB','B':'AB'}) #False
\end{verbatim}\end{center}

To test whether two posets are isomorphic you can use the method
\verb|is_isomorphic| e.g. \verb|B5.is_isomorphic(Boolean(3).CartesianProducct(Boolean(2)))|
returns \verb|True|. You can also construct an instance of \verb|PosetIsoClass|,
which encodes the poset's isomorphism class, via either \verb|B5.isoClass()|
or \verb|PosetIsoClass(B5)|. The equality operator for \verb|PosetIsoClass|
returns \verb|True| when the two represented posets are isomorphic.
The class \verb|PosetIsoClass| inherits from the \verb|Poset| class and
thus has all the same methods, but they are wrapped so that any method
of \verb|Poset| that returns a \verb|Poset| object instead returns
a \verb|PosetIsoClass| object. So \verb|Boolean(3).isoClass().CartesianProduct(Boolean(2))|
gives the isomorphism class of $B_3\times B_2\cong B_5$ and will compare
equal to \verb|B5.isoClass()|.

The posets module can compute some invariants, most notably the
\cv\dv-index and \av\bv-index of a poset. For example,
\verb|B5.flagVectors()| returns a table containing the flag $f$-vector
and flag $h$-vector (as a list of lists \verb|[S,f_S,h_S]|).
\verb|B5.abIndex()| and \verb|B5.cdIndex()| return the \av\bv-index
and the \cv\dv-index respectively, encoded as an instance of the
\verb|Polynomial| class provided by this module. Calling \verb|cdIndex|
on a poset that does not have a \cv\dv-index will still return a \cv\dv-polynomial,
but the result may not really be meaningful. This computes the \cv\dv-index
of a semi-Eulerian poset correctly. Other invariants include
M\"obius function values, Betti numbers
and (the face poset of) the order complex.
'''
from .poset import *
from .hasseDiagram import *
from .examples import *
from .polynomial import *
