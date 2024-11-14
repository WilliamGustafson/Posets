r'''
@is_section@exec@version='0.0.1'@
@sections_order@Poset@PosetIsoClass@Genlatt@HasseDiagram@SubposetsHasseDiagram@Built in posets@Polynomial@Utilities@@
@exec@
def eval_txt(s):
	ret = [r'\color{input}\begin{verbatim}']
	ret.append(s)
	ret.append(r'\end{verbatim}\color{output}\begin{verbatim}')
	ret.append(str(eval(s)))
	ret.append(r'\end{verbatim}\color{black}')
	return ''.join(ret)

def exec_txt(s):
	lines = s.split('\n')
	ret = [r'\color{input}\begin{verbatim}']
	ret.append(s)
	ret.append(r'\end{verbatim}\color{output}\begin{verbatim}')
	exec('\n'.join(lines[:-1]))
	ret.append(str(eval(lines[-1])))
	ret.append(r'\end{verbatim}\color{black}')
	return ''.join(ret)
@
This module provides a class \verb|Poset| that encodes a finite
partially ordered set (poset). The class provides methods to construct
new posets via operations such as Cartesian products and disjoint unions,
select subposets and compute invariants such as flag vectors, the \av\bv-index
and the \cv\dv-index. There is also a class \verb|PosetIsoClass| that
encodes an isomorphism class of a poset. The highlight of this module over
alternatives is the flexible plotting function that produces latex code
for Hasse diagrams that can be compiled into a pdf.

\subsection{Installation}

After cloning the repository from the root directory
run \verb|hatch build| to build distribution files and then
\verb|python -m pip install dist/posets-@eval@version@-py3-none-any.whl|
to install the built wheel file.

The documentation can be built as a pdf by running
\verb|pydox ../src/posets -i \*doc_funcs.py -c| from the docs directory.
\verb|pydox| can be obtained from \url{github.com/WilliamGustafson/pydox.git}.
Compilation requires \LaTeX to be installed with the packages pgf/tikz,
graphicx, fancyvrb, amsmath, amsssymb, scrextend mdframed and hyperref.

\subsection{Overview}
Here we give a quick introduction to using the posets module by way of examples.

First import the module.

\verb|from posets import *|

You can construct a poset in several ways, by specifying the relations
either as a list or dictionary, by providing a function \verb|less|
that returns a Boolean value or by providing an incidence matrix.
You can also copy a poset by passing a \verb|Poset| object to the constructor.
The \verb|Poset| documentation contains a full explanation, below is
an example of constructing the same poset (depicted in Figure~\ref{v-poset-fig})
in four ways.

\begin{center}\begin{verbatim}P = Poset(relations={'a':['ab'],'b':['ab']})
P = Poset(relations=[['a','ab'],['b','ab']])
P = Poset(elements=['a','b','ab'], less=lambda x,y: return x in y and x!=y)
P = Poset(incMat = [[0,0,1],[0,0,1],[0,0,0]], elements=['a','b','ab'])
\end{verbatim}\end{center}

\begin{figure}
\centering
\includegraphics{figures/v.pdf}
\caption{The Hasse diagram of an example poset where $a\le ab$ and $b\le ab$.}
\label{v-poset-fig}
\end{figure}
@exec@
import os
from posets import Poset
P = Poset(relations={'a':['ab'],'b':['ab']})
with open('figures/v.tex','w') as file:
	file.write(P.latex(height=3,width=2,standalone=True))
os.system('pdflatex --output-directory=figures figures/v.tex')
@

Printing a poset via \verb|print(P)| will list the elements, the zeta function \footnote{The zeta function is the matrix $\zeta$ with entries $\zeta_{i,j}=1$ if $i\le j$ and 0 otherwise} as well as a list \verb|ranks|; \verb|ranks[i]| is a list of all indices \verb|j| such that \verb|elements[j]| is length\footnote{the length of $p\in P$ is the length of the longest chain\footnotemark in $P$ with maximum $p$}\footnotetext{A chain of a poset is a subset $\{p_1,\dots,p_k\}$ that is totally ordered,
meaning $p_1<\dots<p_k$.} \verb|i|.
@eval@eval_txt('P')@

You can display the Hasse diagram of a poset in a new window with \verb|P.show()|
or generate tikz code with \verb|P.latex()|. Both methods take keyword arguments to control the output, e.g. \verb|height|, \verb|width|, \verb|nodescale|.
The \verb|latex| method allows for finer grain control of the output compared to \verb|show| but must be compiled before viewing. In the Hasse diagram elements are
ordered left to right along ranks by their order in \verb|P.elements|. As an
example
figure~\ref{v-poset-fig} was generated from the command below.
\begin{verbatim}P.latex(height=3,width=2,standalone=True)\end{verbatim}

This module contains various examples of posets, e.g. the functions
\verb|Boolean|, \verb|Cube|, and \verb|Bruhat|. For example,
you can construct
the poset of all subsets of $\{1,2,3,4,5\}$ via \verb|B5 = Boolean(5)|.

Various operations are given as methods you call on a \verb|Poset| object
For example, to construct the face poset of a triangle crossed with a
square use the \verb|diamondProduct| method:
\begin{verbatim}Q = Boolean(3).diamondProduct(Cube(2))\end{verbatim}
See the operations subsection for a complete list.

Two posets can be compared for equality with the same meaning as in
mathematics (the same underlying set with the same order). For
example
\begin{center}\begin{verbatim}P == B5 #False
P == Poset(elements=['ab','b','a'],incMat=[[0,0,0],[1,0,0],[1,0,0]]) #True
P == Poset(relations={'A':'AB','B':'AB'}) #False
\end{verbatim}\end{center}

To test whether two posets are isomorphic you can use the method
\verb|is_isomorphic| e.g.
\begin{verbatim}B5.is_isomorphic(Boolean(3).CartesianProduct(Boolean(2)))\end{verbatim}
returns \verb|True|. You can also construct an instance of \verb|PosetIsoClass|,
which encodes the poset's isomorphism class, via either \verb|B5.isoClass()|
or \verb|PosetIsoClass(B5)|. The equality operator for \verb|PosetIsoClass|
returns \verb|True| when the two represented posets are isomorphic.

The class \verb|PosetIsoClass| inherits from the \verb|Poset| class and
thus has all the same methods, but they are wrapped so that any method
of \verb|Poset| that returns a \verb|Poset| object instead returns
a \verb|PosetIsoClass| object. For example, the call
\begin{verbatim}Boolean(3).isoClass().CartesianProduct(Boolean(2))\end{verbatim}
returns the isomorphism class of $B_3\times B_2\cong B_5$ and will compare
equal to \verb|B5.isoClass()|.

The posets module can compute some invariants, most notably the
\cv\dv-index and \av\bv-index of a poset. For example,
\verb|B5.flagVectors()| returns a table containing the flag $f$-vector
and flag $h$-vector (as a list of lists \verb|[S,f_S,h_S]|).
\verb|B5.abIndex()| and \verb|B5.cdIndex()| return the \av\bv-index
and the \cv\dv-index respectively, encoded as an instance of the
\verb|Polynomial| class provided by this module. Calling \verb|cdIndex|
on a poset that does not have a \cv\dv-index will still return a \cv\dv-polynomial,
but the result may not really be meaningful. This method computes the \cv\dv-index
of a semi-Eulerian poset (as in \cite{juhnke-kubitzke-24}) correctly though.
Other invariants include
M\"obius function values, Betti numbers
and (the face poset of) the order complex.
'''
from .poset import *
from .hasseDiagram import *
from .examples import *
from .polynomial import *

#del poset
#del hasseDiagram
#del examples
#del polynomial
