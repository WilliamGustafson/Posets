r'''@is_section@
This module provides a class \verb|Poset| that encodes a finite
partially ordered set (poset). The class provides methods to construct
new posets via operations such as Cartesian products and disjoint unions,
select subposets, compute invariants such as flag vectors, the \av\bv-index
and the \cv\dv-index. There is also a class \verb|PosetIsoClass| that
encodes an isomorphism class of a poset.

\subsection{Installation}

After cloaning the repository from the root directory
run \verb|hatch build| to build distribution files and then
\verb|python -m pip install dist/posets-[version]-py3-none-any.whl|
to install the built wheel file.

\subsection{Example session}

First import the module.
\begin{center}from posets import *\end{center}

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

The main data of a \verb|Poset| object are a list \verb|elements| that
specifies the names of the elements and a linear order, a matrix
(list of row lists) \verb|incMat| that specifies the order relation
and a list of lists \verb|ranks| that specifies the length of each element
that is, the maximum length of a chain ending at that element;
\verb|ranks[i]| is a list of indices into \verb|elements| of the length
$i$ elements.

You can display a poset in a new window with \verb|P.show()|
or generate tikz code with \verb|P.latex()|. The \verb|latex| method
allows for fine grained control of the output via the keyword arguments.
Setting a reasonable value for \verb|height| and \verb|width|, and maybe
\verb|nodescale| if element names are large, is usually enough to generate
a nice figure (though the aesthetics strongly depend on the ordering
of \verb|elements|). The elements are placed vertically in rows by their
rank and within a rank elements are sorted as they occur in \verb|elements|.

Printing a poset via \verb|print(P)| shows the elements, zeta matrix
(the matrix indexed by the elements with 1 when $p\le q$ and 0 otherwise)
and the ranks list. You can get the zeta matrix via \verb|P.zeta()|.

The module contains various examples of posets, e.g.
\verb|Boolean|, \verb|Cube|, and \verb|Bruhat|. For example,
you can construct
the poset of all subsets of $\{1,2,3,4\}$ via \verb|B5 = Boolean(5)|.

Various operations are given as methods you call on a \verb|Poset| object.
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
