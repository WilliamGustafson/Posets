---
author:
- William Gustafson
bibliography:
- bib.bib
date: 2024-11-30
title: Posets
---

# Introduction

This module provides a class [`Poset`](#Poset) that encodes a finite
partially ordered set (poset). Most notably, this module can efficiently
compute flag vectors, the [**a**]{.upright}[**b**]{.upright}-index and
the [**c**]{.upright}[**d**]{.upright}-index. Latex code for Hasse
diagrams can be produced with a very flexible interface. There are
methods for common operations and constructions such as Cartesian
products, disjoint unions, interval lattices, lattice of ideals, etc.
Various examples of posets are provided such as Boolean algebras, the
face lattice of the $n$-dimensional cube, (noncrossing) partition
lattices, the type $A_n$ Bruhat and weak orders, uncrossing orders etc.
General subposets can be selected as well as particular ones of interest
such as intervals and rank selections. Posets from this module can also
be converted to and from posets from
[sagemath](https://www.sagemath.org) and
[Macaulay2](https://www.macaulay2.com/).

## Installation

Download the wheel file
[here](https://www.github.com/WilliamGustafson/posets/releases) and
install it with pip via `python -m pip posets-*-py3-none-any.whl`.

## Building

Building requires [hatch](https://hatch.pypa.io) to be installed. After
cloning the repository from the base directory of the repository run
`hatch build` to build distribution files and then run

    python -m pip install dist/posets-*-py3-none-any.whl

to install the wheel file.

The documentation can be built in pdf form by running `make docs` from
the base directory. Compilation requires LaTeXto be installed with the
packages pgf/tikz, graphicx, fancyvrb, amsmath, amsssymb, scrextend,
mdframed and hyperref as well as the python module `pydox`. `pydox` can
be obtained from
[github.com/WilliamGustafson/pydox.git](github.com/WilliamGustafson/pydox.git){.uri}.
If `pydox` is not on your path either make a symlink to it or call make
as `make PYDOX=[path to pydox] docs`.

You can run the tests via `make test`, this requires `pytest` to be
installed. You can output an html coverage report, located at
`tests/htmlcov/index.html`, with `make coverage`. Making the coverage
report requires [pytest](https://pytest.org),
[coverage](https://coverage.readthedocs.io) and the
[pytest-cov](https://pytest-cov.readthedocs.io) plugin. Note, coverage
on `hasseDiagram.py` is very low because testing drawing functions
cannot be easily automated.

## Quick start

Here we give a quick introduction to using the posets module. In this
subsection python commands and outputs are denoted in typewriter font.

In the code snippets below we assume the module is imported via

`from posets import *`

Constructing a poset:

    P = Poset(relations={'':['a','b'],'a':['ab'],'b':['ab']})
    Q = Poset(relations=[['','a','b'],['a','ab'],['b','ab']])
    R = Poset(elements=['ab','a','b',''], less=lambda x,y: return x in y)
    S = Poset(incMat = [[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]], elements=['','a','b','ab'])

Built in examples (see page ):

    Boolean(3) #Boolean algebra of rank 3
    Cube(3) #Face lattice of the 3-dimensional cube
    Bruhat(3) #Bruhat order on symmetric group of order 3!
    Bnq(n=3,q=2) #Lattice of subspaces of F_2^3
    DistributiveLattice(P) #lattice of ideals of P
    Intervals(P) #lattice of intervals of P (including the empty interval)

These examples come with default drawing methods, for example, when
making latex code by calling `DistributiveLattice(P).latex()` the
resulting figure depicts elements of the lattice as Hasse diagrams of
$P$ with elements of the ideal highlighted (again, see page ). Note, you
will have to set the `height`, `width` and possibly `nodescale`
parameters in order to get sensible output[^1].

Two posets compare equal when they have the same set of elements and the
same order relation on them:

    P == Q and Q == R and R == S #True
    P == Poset(relations={'':['a','b']}) #False
    P == Poset(relations={'':['ab'],'a':['ab'],'b':['ab']}) #False

Use [`is_isomorphic`](#Poset.is_isomorphic) or
[`PosetIsoClass`](#PosetIsoClass) to check whether posets are
isomorphic:

    P.is_isomorphic(Boolean(2)) #True
    P.isoClass()==Boolean(2).isoClass() #True
    P.is_isomorphic(Poset(relations={'':['a','b']})) #False

Viewing and creating Hasse diagrams:

    P.show() #displays a Hasse diagram in a new window
    P.latex() #returns latex code: \begin{tikzpicture}...
    P.latex(standalone=True) #latex code for a standalone document: \documentclass{preview}...
    display(P.img()) #Display a poset when in a Jupyter notebook
    #this uses the output of latex()

Computing invariants:

    Cube(2).fVector() #{(): 1, (1,): 4, (2,): 4, (1, 2): 8}
    Cube(2).hVector() #{(): 1, (1,): 3, (2,): 3, (1, 2): 1}
    Boolean(5).sparseKVector() #{(3,): 8, (2,): 8, (1, 3): 4, (1,): 3, (): 1}
    Boolean(5).cdIndex() #Polynomial({'ccd': 3, 'cdc': 5, 'dd': 4, 'dcc': 3, 'cccc': 1})
    print(Boolean(5).cdIndex()) #c^{4}+3c^{2}d+5cdc+3dc^{2}+4d^{2}

Polynomial operations:

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

Converting posets to and from SageMath:

    P.toSage() #Returns a SageMath class, must be run under sage
    fromSage(Q) #Take a poset Q made with SageMath and return an instance of Poset

Converting to and from Macaulay2:

    -- In M2
    load "convertPosets.m2" --Also loads Python and Posets packages
    import "posets" --This module must be installed to system version of python
    P = posetsBoolean(3) --Calling python functions
    pythonPosetToMac(P) --Returns an instance of the M2 class Posets
    macPosetToPython(Q) --Take a poset made with M2 and return an
    --instance of the python class Poset

# Poset {#Poset}

**[`class Poset`]{#Poset}**

A class representing a finite partially ordered set.

Posets are encoded by a list `elements`, an incidence matrix `incMat`
describing the relations and a list `ranks` that specifies the length of
each element. This last attribute is not strictly needed to encode a
poset but many calculations use the length of elements so it is computed
on construction. Instances of [`Poset`](#Poset) also have an attribute
`hasseDiagram` which is an instance of the
[`HasseDiagram`](#HasseDiagram) class used for plotting the poset.

To construct a poset you must pass at least either an incident matrix
`incMat`, a function [`less`](#Poset.less) or a list/dictionary
[`relations`](#Poset.relations) to describe the relations. Additionally
you may wish to specify the elements as a list called `elements` or by
using the [`relations`](#Poset.relations) argument. The full list of
constructor arguments are listed below.

-    `incMat` -- A matrix whose $i,j$ entry is 1 if the $i$th element is
    strictly less than the $j$th element.

-   `elements` -- A list specifying the elements of the poset.

    The default value is `[0,...,len(incMat)-1]`

-    `ranks` -- A list of lists. The $i$th list is a list of indices of
    element of length $i$. This argument is inessential, if not provided
    it will be computed by the constructor. If constructing a large
    poset with an easily computed rank function you may wish to compute
    and pass the rank function to the constructor.

-    [`relations`](#Poset.relations) -- Either a list of pairs $(x,y)$
    such that $x<y$ or a dictionary whose values are lists of elements
    greater than the associated key. This is used to construct incMat if
    it is not provided.

-    [`less`](#Poset.less) -- A function that given two elements $p,q$
    returns `True` when $p < q$. This is used to construct incMat if
    neither `incMat` nor [`relations`](#Poset.relations) are provided.

-    `indices` -- A boolean indicating indices instead of elements are
    used in [`relations`](#Poset.relations) and [`less`](#Poset.less).
    The default is `False`.

-    `name` -- An optional identifier. If not provided no name attribute
    is set.

-   `hasse_class` -- An instance of `hasse_class` is constructed with
    arguments being this poset plus all keyword arguments passed to
    [`Poset`](#Poset), i.e.:

                    this.hasseDiagram = hasse_class(this, **kwargs)

    If you subclass [`HasseDiagram`](#HasseDiagram) to change default
    drawing behavior pass your subclass when constructing a poset.

    The default value is [`HasseDiagram`](#HasseDiagram).

-    `trans_close` -- If `True` the transitive closure of `incMat` is
    computed, this should be `False` only if the provided matrix
    satisfies $$\verb|incMat[i][j] ==|\begin{cases}
                        1 & \text{when } i<j\\
                        -1 & \text{when } i>j\\
                        0 & \text{otherwise}.
                    \end{cases}$$

The keyword arguments are passed to [`HasseDiagram`](#HasseDiagram) (or
`hasse_class` if specified).

Function calls to several of the more costly computations are cached.
Generally, functions in this class do not change the poset but instead
return a new poset. [`Poset`](#Poset) objects may be considered
immutable (this is not enforced in any way), or if you alter a poset you
should clear the cache via: `this.cache = {}`.

::: child
## Operations {#Operations}

**[`def adjoin_onehat(this, label=None)`]{#Poset.adjoin_onehat}**

Returns a new poset with a new maximum adjoined.

The label default is the same as `Poset.adjoin_zerohat()`

**[`def adjoin_zerohat(this, label=None)`]{#Poset.adjoin_zerohat}**

Returns a new poset with a new minimum adjoined.

By default the label is 0 and if 0 is already an element the default
label is the first positive integer that is not an element.

**[`def bddProduct(this, that)`]{#Poset.bddProduct}**

Computes the Cartesian product of two posets with maximum and minimum
adjoined, that is, the poset
$$\big((P\setminus\{\max P,\min P\})\times(Q\setminus\{\max Q,\min Q\})\big)\cup\{\widehat{0},\widehat{1}\}.$$

**[`def bddUnion(this, that)`]{#Poset.bddUnion}**

Computes the disjoint union of two posets with maximum and minimums
adjoined, that is, the poset
$$\big( (P\setminus\{\max P,\min P\})\sqcup(Q\setminus\{\max Q,\min Q\})\big).$$

The labels in the returned poset are the same as in
[`element_union`](#Poset.element_union).

**[`def cartesianProduct(this, that)`]{#Poset.cartesianProduct}**

Computes the cartesian product.

**[`def diamondProduct(this, that)`]{#Poset.diamondProduct}**

Computes the diamond product which is the Cartesian product of the two
posets with their minimums removed and then adjoined with a new minimum.

**[`def dual(this)`]{#Poset.dual}**

Returns the dual poset which has the same elements and relation $p\le q$
when $q\le p$ in the original poset.

**[`def identify(this, X, indices=False)`]{#Poset.identify}**

Returns a new poset after making identifications indicated by `X`.

The new relation is $p\le q$ when there exists any representatives $p'$
and $q'$ such that $p'\le q'$. The result may not truly be a poset as it
may not satisfy the antisymmetry axiom ($p<q<p$ implies $p = q$).

`X` should either be a dictionary where keys are the representatives and
the value is a list of elements to identify with the key, or a list of
lists where the first element of each list is the representative.
Trivial equivalence classes need not be specified.

**[`def prism(this)`]{#Poset.prism}**

Computes the prism of a poset, that is, the diamond product with
`Cube(1)`.

**[`def pyr(this)`]{#Poset.pyr}**

Computes the pyramid of a poset, that is, the Cartesian product with a
length 1 chain.

**[`def starProduct(this, that)`]{#Poset.starProduct}**

Computes the star product of two posets.

This is the union of `this` with the maximum removed and `that` with the
minimum removed and all relations $p < q$ for $p$ in `this` and $q$ in
`that`.

**[`def union(this, that)`]{#Poset.union}**

Computes the disjoint union of two posets.

The labels in the returned poset are determined by
[`element_union`](#Poset.element_union).

## Subposet Selection {#Subposet Selection}

**[`def complSubposet(this, S, indices=False)`]{#Poset.complSubposet}**

Returns the subposet of elements not contained in `S`.

**[`def filter(this, x, indices=False, strict=False)`]{#Poset.filter}**

Returns the subposet of elements greater than or equal to any element of
`x`.

If `strict` is `True` then `x` is not included in the returned poset and
if it is `False` `x` is included.

**[`def ideal(this, x, indices=False, strict=False)`]{#Poset.ideal}**

Returns the subposet of elements less than or equal to any element of
`x`.

Wrapper for [`this.dual().filter`](#Poset.filter).

**[`def interval(this, i, j, indices=False)`]{#Poset.interval}**

Returns the closed interval $[i,j]$.

**[`def max(this, indices=False)`]{#Poset.max}**

Returns a list of the maximal elements of the poset.

**[`def min(this, indices=False)`]{#Poset.min}**

Returns a list of the minimal elements of the poset.

**[`def properPart(this)`]{#Poset.properPart}**

Returns the subposet of all elements that are neither maximal nor
minimal.

**[`def rankSelection(this, S)`]{#Poset.rankSelection}**

Returns the subposet of elements whose rank is contained in `S`.

Does not automatically include the minimum or maximum.

**[`def subposet(this, S, indices=False, keep_hasseDiagram=True)`]{#Poset.subposet}**

Returns the subposet of elements in `S`.

## Internal Computations {#Internal Computations}

**[`def isAntichain(this, A, indices=False)`]{#Poset.isAntichain}**

Returns whether the given set is an antichain ($i\not<j$ for all $i$ and
$j$).

**[`def join(this, i, j, indices=False)`]{#Poset.join}**

Computes the join of $i$ and $j$, if it does not exist returns `None`.

**[`def less(this, i, j, indices=False)`]{#Poset.less}**

Returns whether $i$ is strictly less than $j$.

**[`def lesseq(this, i, j, indices=False)`]{#Poset.lesseq}**

Returns whether $i$ is less than or equal to $j$.

**[`def mobius(this, i=None, j=None, indices=False)`]{#Poset.mobius}**

Computes the value of the Möbius function from $i$ to $j$.

If $i$ or $j$ is not provided computes the mobius from the minimum to
the maximum and throws an exception of type `ValueError` if there is no
minimum or maximum.

**[`def rank(this, i, indices=False)`]{#Poset.rank}**

Returns the length of $i$ (the length of the longest chain ending at
$i$).

Returns `None` if `i` is not an element (or `i` is not a valid index if
`indices` is `True`).

## Queries {#Queries}

**[`def covers(this, indices=False)`]{#Poset.covers}**

Returns the list of covers of the poset.

We say $q$ covers $p$ when $p<q$ and $p<r<q$ implies $r=p$ or $r=q$.

**[`def isEulerian(this)`]{#Poset.isEulerian}**

Checks whether the given poset is Eulerian (every interval with at least
2 elements has an equal number of odd and even rank elements).

**[`def isGorenstein(this)`]{#Poset.isGorenstein}**

Checks if the poset is Gorenstein\*.

A poset is Gorenstein\* if the proper part of all intervals with more
than two elements have sphere homology. In other words, this function
checks that

::: center
`this.interval(p,q).properPart().bettiNumbers()`
:::

is either `[2]` or `[1,0,...,0,1]` for all $p<=q$ such that
`this.rank(q) - this.rank(p) >= 2`.

**[`def isLattice(this)`]{#Poset.isLattice}**

Checks if the poset is a lattice.

Returns `True` if `this` is an empty poset.

**[`def isRanked(this)`]{#Poset.isRanked}**

Checks whether the poset is ranked.

## Invariants {#Invariants}

**[`def abIndex(this)`]{#Poset.abIndex}**

Returns the -index of the poset.

If the poset has a unique minimum and maximum but isn't ranked this
computes the -index considering the poset to be quasigraded (in the
sense of [@ehrenborg-goresky-readdy-15]) with $\overline{\zeta}=\zeta$
and $\rho$ the length function.

For more information on the -index see [@bayer-21]

**[`def bettiNumbers(this)`]{#Poset.bettiNumbers}**

Computes the Betti numbers of the poset, that is, the ranks of the
homology of the order complex (the simplicial complex of all chains).

**[`def cdIndex(this)`]{#Poset.cdIndex}**

Returns the **cd**-index.

The **cd**-index is encoded as an instance of the class
[`Polynomial`](#Polynomial). If the given poset does not have a
**cd**-index then a **cd**-polynomial is still returned, but this is not
meaningful. If you wish to check whether a poset has a **cd**-index then
check the Boolean below:

::: center
`this.cdIndex().cdToAb() == this.abIndex()`
:::

If the given poset is semi-Eulerian then the **cd**-index as defined in
[@juhnke-kubitzke-24] is computed.

For computation we use the sparse $k$-vector formula see
[@billera-ehrenborg-00 Proposition 7.1]. For more info on the
**cd**-index see [@bayer-21].

**[`def fVector(this)`]{#Poset.fVector}**

Returns flag $f$-vector as a dictionary with keys $S\subseteq[n]$.

This method is intended for use with a poset that has a unique minimum
and maximum. On a general poset this counts chains that contain the
first minimal element, `this[this.ranks[0][0]]` and ignores the final
rank.

**[`def flagVectors(this)`]{#Poset.flagVectors}**

Returns the table of flag $f$- and $h$-vectors as a dictionary with keys
$S\subseteq[n]$ encoded as tuples and with elements `(f_S, h_S)`.

**[`def flagVectorsLatex(this,standalone=False)`]{#Poset.flagVectorsLatex}**

Returns a string of latex code representing the table of flag vectors of
the poset.

Requires the package longtable to compile.

**[`def hVector(this)`]{#Poset.hVector}**

Returns the flag $h$-vector of the poset.

**[`def orderComplex(this, indices=False)`]{#Poset.orderComplex}**

Returns the poset of all chains ordered by inclusion.

**[`def sparseKVector(this)`]{#Poset.sparseKVector}**

Returns the sparse $k$-vector
$k_S = \sum_{T\subseteq S}(-1)^{\lvert{S\setminus T}\rvert}h_T$.

The sparse $k$-vector only has entries $k_S$ for sparse sets $S$, that
is, sets $S\subseteq[\text{rk}(P)-1]$ such that if $i\in S$ then
$i+1\not\in S$. The sparse $k$-vector is returned as a dictionary whose
keys are tuples.

**[`def zeta(this)`]{#Poset.zeta}**

Returns the zeta matrix, the matrix whose $i,j$ entry is $1$ if the
Boolean

    this.lesseq(i,j,indices=True)

is true and $0$ otherwise.

## Maps {#Maps}

**[`def buildIsomorphism(this, that, indices=False)`]{#Poset.buildIsomorphism}**

Returns an isomorphism from `this` to `that` as a dictionary or `None`
if the posets are not isomorphic.

If `indices` is `True` then the dictionary keys and values are indices,
otherwise they are elements.

**[`def is_isomorphic(this, that)`]{#Poset.is_isomorphic}**

Returns `True` if the posets are isomorphic and `False` otherwise.

## Miscellaneous {#Miscellaneous}

**[`def __contains__(this, p)`]{#Poset.__contains__}**

Wrapper for [`this.elements.__contains__`](#Poset.__contains__).

**[`def __eq__(this,that)`]{#Poset.__eq__}**

Returns `True` when `that` is a [`Poset`](#Poset) representing the same
poset as `this` and `False` otherwise.

**[`def __getitem__(this, i)`]{#Poset.__getitem__}**

Wrapper for [`this.elements.__getitem__`](#Poset.__getitem__).

**[`def __hash__(this)`]{#Poset.__hash__}**

Hashes the poset, dependent on `this.elements` and relations between
ranks.

**[`def __init__(this, incMat=None, elements=None, ranks=None, less=None, name=”,`]{#Poset.__init__}**\
**`hasse_class=None, trans_close=True, relations=None, indices=False, that=None,**kwargs)`**\

See [`Poset`](#Poset).

**[`def __iter__(this)`]{#Poset.__iter__}**

Wrapper for `this.elements__iter__`.

**[`def __len__(this)`]{#Poset.__len__}**

Wrapper for [`this.elements.__len__`](#Poset.__len__).

**[`def __repr__(this)`]{#Poset.__repr__}**

Gives a string that can be evaluated to recreate the poset.

To `eval` the returned string [`Poset`](#Poset) must be in the namespace
and `repr(this.elements)` must return a suitable string for evaluation.

**[`def __str__(this)`]{#Poset.__str__}**

Returns a nicely formatted string listing the zeta matrix, the ranks
list and the elements of the poset.

**[`def chains(this, indices=False)`]{#Poset.chains}**

Returns a list of all nonempty chains of the poset (subsets
$p_1<\dots<p_r$).

**[`def copy(this)`]{#Poset.copy}**

Returns a shallow copy of the poset.

Making a shallow copy via the copy module i.e. `Q = copy.copy(P)`
doesn't update the self reference in `Q.hasseDiagram` (in this example
`Q.hasseDiagram.P` is `P`). This doesn't matter if you treat posets as
immutable, but otherwise could cause issues when displaying or
generating hasse diagrams. The returned poset has the self reference
updated.

**[`def fromSage(P)`]{#Poset.fromSage}**

Convert an instance of `sage.combinat.posets.poset.FinitePoset` to an
instance of [`Poset`](#Poset).

**[`def img(this, tmpfile=’a.tex’, tmpdir=None, **kwargs)`]{#Poset.img}**

Produces latex code (via calling [`latex`](#Poset.latex)) compiles it
with pdflatex and returns a `wand.image.Image` object constructed from
the pdf.

In a Jupyter notebook calling `display` on the return value will show
the Hasse diagram in the output cell. By default `tmpdir` is
`tempfile.gettempdir()`.

This function converts the compiled pdf to an image using imagemagick,
this may fail due to imagemagick's default security policies. For more
info and how to fix the issue see [@askubuntu-imagemagick].

Keyword arguments are passed to `latex()` but `standalone` is alwasy set
to `True` (otherwise the pdf would not compile). Note this function will
hang if `pdflatex` fails to compile.

**[`def isoClass(this)`]{#Poset.isoClass}**

Returns an instance of [`PosetIsoClass`](#PosetIsoClass) representing
the isomorphism class of the poset.

**[`def latex(this, **kwargs)`]{#Poset.latex}**

Returns a string of tikz code to draw the Hasse diagram of the poset for
use in a LaTeX document.

This is a wrapper for [`HasseDiagram.latex`](#Poset.latex).

For a full list of keyword arguments see
[`HasseDiagram`](#HasseDiagram). The most common arguments are:

-   `height` -- The height in tikz units of the diagram.

    The default value is 10.

-   `width` -- The width in tikz units of the diagram.

    The default value is 8.

-   `labels` -- If `False` elements are represented by filled circles.
    If `True` by default elements are labeled by the result of casting
    the poset element to a string.

    The default value is `True`.

-   `ptsize` -- When labels is `False` this is the size of the circles
    used to represent elements. This has no effect if labels is `True`.

    The default value is `'2pt'`.

-   `nodescale` -- Each node is wrapped in
    `'\\scalebox{'+nodescale+'}'`.

    The default value is `'1'`.

-   `standalone` -- When `True` a preamble is added to the beginning and
    `'\\end{document}'` is added to the end so that the returned string
    is a full LaTeX document that can be compiled. Compiling requires
    the LaTeX packages tikz (pgf) and preview. The resulting figure can
    be incorporated into another LaTeX document with `\includegraphics`.

    When `False` only the code for the figure is returned; the return
    value begins with `\begin{tikzpicture}` and ends with
    `\end{tikzpicture}`.

    The default is `False`.

-   [`nodeLabel`](#HasseDiagram.nodeLabel) -- A function that takes the
    [`HasseDiagram`](#HasseDiagram) object and an index and returns the
    label for the indicated element as a string. For example, the
    default implementation
    [`HasseDiagram.nodeLabel`](#HasseDiagram.nodeLabel) returns the
    element cast to a string and is defined as below:

    ::: center
                    def nodeLabel(H, i):
                        return str(H.P[i])
    :::

    Note `H.P` is `this`.

**[`def make_ranks(incMat)`]{#Poset.make_ranks}**

Used by the constructor to compute the ranks list for a poset when it
isn't provided.

**[`def relabel(this, elements=None)`]{#Poset.relabel}**

Returns a new [`Poset`](#Poset) object with the `elements` attribute as
given.

If `elements` is `None` then the returned poset has `elements` attribute
set to `list(range(len(this)))`.

**[`def relations(this, indices=False)`]{#Poset.relations}**

Returns a list of all pairs $(e,f)$ where $e\le f$.

**[`def reorder(this, perm, indices=False)`]{#Poset.reorder}**

Returns a new [`Poset`](#Poset) object (representing the same poset)
with the elements reordered.

`perm` should be a list of elements if `indices` is `False` or a list of
indices if `True`. The returned poset has elements in the given order,
i.e. `perm[i]` is the $i$th element.

**[`def show(this, **kwargs)`]{#Poset.show}**

Opens a window displaying the Hasse diagram of the poset.

This is a wrapper for [`HasseDiagram.tkinter`](#HasseDiagram.tkinter).

For a full list of keyword arguments see
[`HasseDiagram`](#HasseDiagram). The most common arguments are:

-   `height` -- The height of the diagram.

    The default value is 10.

-   `width` -- The width of the diagram.

    The default width is 8.

-   `labels` -- If `False` elements are represented as filled circles.
    If `True` by default elements are labeled by the result of casting
    the poset element to a string.

    The default value is `True`.

-   `ptsize` -- When labels is `False` controls the size of the circles
    representing elements. This can be an integer or a string, if the
    value is a string the last two characters are ignored.

    The default value is `'2pt'`.

-   `scale` -- Scale of the diagram.

    The default value is 1.

-   `padding` -- A border of this width is added around all sides of the
    diagram.

    The default value is 1.

-   [`nodeLabel`](#HasseDiagram.nodeLabel) -- A function that takes the
    [`HasseDiagram`](#HasseDiagram) object and an index and returns the
    label for the indicated element as a string. For example, the
    default implementation
    [`HasseDiagram.nodeLabel`](#HasseDiagram.nodeLabel) returns the
    element cast to a string and is defined as below:

    ::: center
                    def nodeLabel(H, i):
                        return str(H.P[i])
    :::

    Note `H.P` is `this`.

**[`def shuffle(this)`]{#Poset.shuffle}**

Returns a new [`Poset`](#Poset) object (representing the same poset)
with the elements in a random order.

**[`def sort(this, key = None, indices=False)`]{#Poset.sort}**

Returns a new [`Poset`](#Poset) object (representing the same poset)
with the elements sorted.

**[`def toSage(this)`]{#Poset.toSage}**

Converts this to an instance of
`sage.combinat.posets.posets.FinitePoset`.

**[`def transClose(M)`]{#Poset.transClose}**

Given a matrix with entries $1,-1,0$ encoding a relation computes the
transitive closure.
:::

# PosetIsoClass {#PosetIsoClass}

**[`class PosetIsoClass(Poset)`]{#PosetIsoClass}**

This class encodes the isomorphism type of a poset.

Internally, this class inherits from [`Poset`](#Poset) and thus
instances of [`PosetIsoClass`](#PosetIsoClass) are also instances of
[`Poset`](#Poset). The major differences between this class and
[`Poset`](#Poset) are that `PosetIsoClass.__eq__` returns `True` when
the two posets are isomorphic and all methods in [`Poset`](#Poset) that
return a [`Poset`](#Poset) object in [`PosetIsoClass`](#PosetIsoClass)
instead return an instance of [`PosetIsoClass`](#PosetIsoClass).

Construct an instance of [`PosetIsoClass`](#PosetIsoClass) the same way
as you would an instance of [`Poset`](#Poset) or given a poset `P` use
`P.isoClass()`.

# Genlatt {#Genlatt}

**[`class Genlatt(Poset)`]{#Genlatt}**

A class to encode a "generator-enriched lattice" which is a lattice $L$
along with a set $G\subseteq L\setminus\{\widehat{0}\}$ that generates
$L$ under the join operation.

This class is mainly provided for the
[`minorPoset`](#Genlatt.minorPoset) method.

Constructor arguments are the same as for [`Poset`](#Poset) except that
this constructor accepts two additional keyword only arguments:

-   `G` - An iterable specifying the generating set, can either contain
    elements of the lattice or indices into the lattice. The join
    irreducibles are automatically added and may be omitted. If `G` is
    not provided the generating set will consist of the join
    irreducibles.

-   `G_indices` - If `True` then the provided argument `G` should
    consist of indices otherwise `G` should consist of elements.

Note, a lattice $L$ enriched with a generating set $G$ is denoted as the
pair $(L,G)$. See [@gustafson-23] for more on generator-enriched
lattices[^2].

::: child
**[`def Del(this, K)`]{#Genlatt.Del}**

Return the deletion set of the minor `K`.

The argument `K` should be an instance of [`Genlatt`](#Genlatt).

The deletion set of a minor $(K,H)$ of $(L,G)$ is the set
$$\text{Del}(K,H)=\{g\in G:g\vee\widehat{0}_K\not\in H\cup\{\widehat{0}_K\}\}$$
This is the minimal set of generators that must be deleted to form
$(K,H)$ from $(L,G)$.

**[`def __init__(this, *args, G=None, G_indices=False, **kwargs)`]{#Genlatt.__init__}**

See [`Genlatt`](#Genlatt).

**[`def _minors(this, minors, rels, i, weak, L)`]{#Genlatt._minors}**

Recursion backend to [`minors`](#Genlatt.minors).

**[`def contract(this, g, weak=False, L=None)`]{#Genlatt.contract}**

Return the [`Genlatt`](#Genlatt) obtained by contracting the generator
`g`, if `weak` is `True` performs the weak contraction with respect to
`L` (default value for `L` is `this`).

**[`def delete(this, g)`]{#Genlatt.delete}**

Return the [`Genlatt`](#Genlatt) obtained by deleting the generator `g`.

**[`def minor(this, H, z)`]{#Genlatt.minor}**

Given an iterable `H` of generators and an element `z` returns the
[`Genlatt`](#Genlatt) with minimum `z` and generating set `H` and with
the same order as `this`.

**[`def minorPoset(this, weak=False, **kwargs)`]{#Genlatt.minorPoset}**

Returns the minor poset of the given [`Genlatt`](#Genlatt) instance.

When generating a Hasse diagram with `latex()` use the prefix `L_` to
control options for the node diagrams.

**[`def minors(this, weak=False)`]{#Genlatt.minors}**

Returns a list of minors of the given [`Genlatt`](#Genlatt) instance and
an incomplete dictionary of relations.

The relations when transitively closed yield the relations for the minor
poset.
:::

# HasseDiagram {#HasseDiagram}

**[`class HasseDiagram`]{#HasseDiagram}**

A class that can produce latex/tikz code for the Hasse diagram of a
poset or display the diagram in a window using tkinter.

**Overview**

An instance of this class is attached to each instance of
[`Poset`](#Poset). This class is used to produce latex code for a poset
when `Poset.latex()` is called or to display a poset in a new window
when `Poset.show()` is called. These functions are wrappers for
`HasseDiagram.latex()` and `HasseDiagram.tkinter()`.

The constructor for this class takes keyword arguments that control how
the Hasse diagram is drawn. These keyword arguments set the default
options for that given instance of [`HasseDiagram`](#HasseDiagram). When
calling `latex()` to produce latex code or `tkinter()` to draw the
diagram in a tkinter window the same keyword arguments can be passed to
control how the diagram is drawn during that particular operation.

There are two types of options: constant values such as `height`,`width`
or `scale` and function values such as [`loc_x`](#HasseDiagram.loc_x),
[`loc_y`](#HasseDiagram.loc_y) or
[`nodeLabel`](#HasseDiagram.nodeLabel).

**Keyword arguments**

Options that affect both `latex()` and `tkinter()`:

-   `width` -- Width of the diagram. When calling `latex()` this is the
    width in tikz units (by default centimeters), for `tkinter()` the
    units are $\frac{1}{30}$th of tkinter's units.

    The default value is 8.

-   `height` -- Height of the diagram, uses the same units as width.

    The default value is 10.

-   `labels` -- If this is `True` display labels, obtained from
    `nodeLabels`, for elements; if this is `False` display filled
    circles for elements. The default value is `True`.

-   `ptsize` -- No effect when `labels` is `True`, when `labels` is
    `False` this is the size of the circles shown for elements. When
    calling `tkinter()` this can be either a number or a string. For
    compatibility if `ptsize` is a string the last two characters are
    ignored. When calling `latex()` this should be a string and include
    units.

    The default value is '2pt'.

-   `indices_for_nodes` -- If `True` then
    [`this.nodeLabel`](#HasseDiagram.nodeLabel) is not called and the
    node text is the index of the element in the poset. If `labels` is
    `False` this argument has no effect.

    The default value is `False`.

-   [`nodeLabel`](#HasseDiagram.nodeLabel) -- A function that given
    `this` and an index returns a string to label the corresponding
    element by.

    The default value is
    [`HasseDiagram.nodeLabel`](#HasseDiagram.nodeLabel).

-   [`loc_x`](#HasseDiagram.loc_x) -- A function that given `this` and
    an index returns the $x$-coordinate of the element in the diagram as
    a string. Positive values extend rightward and negative leftward.

    The default value is [`HasseDiagram.loc_x`](#HasseDiagram.loc_x).

-   [`loc_y`](#HasseDiagram.loc_y) -- A function that given `this` and
    an index returns the $y$-coordinate of the element in the diagram as
    a string. Positive values extend upward and negative values
    downward.

    The default value is [`HasseDiagram.loc_y`](#HasseDiagram.loc_y).

-   `jiggle` `jiggle_x` `jiggle_y` -- Coordinates of all elements are
    perturbed by a random vector in the rectangle

    ::: center
    $-\verb|jiggle|-\verb|jiggle_x| \le x \le \verb|jiggle|+\verb|jiggle_x|$\
    $-\verb|jiggle|-\verb|jiggle_y| \le y \le \verb|jiggle|+\verb|jiggle_y|$
    :::

    This can be useful if you want to prevent cover lines from
    successive ranks aligning to form the illusion of a line crossing
    between two ranks; or when drawing unranked posets if a line happens
    to cross over an element. The perturbation occurs in
    [`loc_x`](#HasseDiagram.loc_x) and [`loc_y`](#HasseDiagram.loc_y) so
    if these are overwritten and you want to preserve this behaviour add
    a line to the end of your implementation of
    [`loc_x`](#HasseDiagram.loc_x) such as

    ::: center
    `x = x+random.uniform(-jiggle-jiggle_x,jiggle+jiggle_x)`
    :::

    The default values are 0.

-   `scale` -- In `latex()` this is the scale parameter for the tikz
    environment, i.e. the tikz environment containing the figure begins

    ::: center
    `'\\begin{tikzpicture}[scale='+tikzscale+']'`
    :::

    In `tkinter()` all coordinates are scaled by $\verb|scale|$.

    The default value is '1', this parameter may be a string or a
    numeric type.

Options that affect only `latex()`:

-   `preamble` -- A string that when calling `latex()` is placed in the
    preamble. It should be used to include any extra packages or define
    commands needed to produce node labels. This has no effect when
    standalone is `False`.

    The default value is `''`.

-   `nodescale` -- Each node is wrapped in
    `'\\scalebox{'+nodescale+'}'`.

    The default value is `'1'`.

-   [`line_options`](#HasseDiagram.line_options) -- Tikz options to be
    included on lines drawn, i.e. lines will be written as

        '\\draw['+line_options+'](...'

    The value for [`line_options`](#HasseDiagram.line_options) can be
    either a string or a function; when it is a string the same options
    are placed on every line and when the value is a function it is
    passed `this`, the [`HasseDiagram`](#HasseDiagram) object, `i`, the
    index to the element at the bottom of the cover and `j`, the index
    to the element at the top of the cover.

    The default value is `''`.

-   [`node_options`](#HasseDiagram.node_options) -- Tikz options to be
    included on nodes drawn, i.e. nodes will be written as

        '\\node['+node_options_'](...'

    Just as with [`line_options`](#HasseDiagram.line_options) the value
    for [`node_options`](#HasseDiagram.node_options) can be either a
    string or a function; if it is a function it is passed `this`, the
    [`HasseDiagram`](#HasseDiagram) object, and `i`, the index to the
    element being drawn.

-   `northsouth` -- If `True` lines are not drawn between nodes directly
    but from `node.north` to `node.south` which makes lines come
    together just beneath and above nodes. When `False` lines are drawn
    directly to nodes which makes lines directed towards the center of
    nodes.

    The default is `True`.

-   `lowsuffix` -- When this is nonempty lines will be drawn to
    `node.lowsuffix` instead of directly to nodes for the higher node in
    each cover. If `northsouth` is `True` this has no effect and
    `'.south'` is used for the low suffix.

    The default is `''`.

-   `highsuffix` -- This is the suffix for the bottom node in each
    cover. If `northsouth` is `True` this has no effect and `'.north'`
    is used for the high suffix.

    The default is `''`.

-   [`nodeName`](#HasseDiagram.nodeName) -- A function that takes `this`
    and an index `i` representing an element whose node is to be drawn
    and returns the name of the node in tikz. This does not affect the
    image but is useful if you intend to edit the latex code and want
    the node names to be human readable.

    The default value [`HasseDiagram.nodeName`](#HasseDiagram.nodeName)
    returns `str(i)`.

-   `standalone` -- When `True` a preamble is added to the beginning and
    `'\\end{document}'` is added to the end so that the returned string
    is a full latex document that can be compiled. Compiling requires
    the latex packages tikz (pgf) and preview. The resulting figure can
    be incorporated into another latex document with `\includegraphics`.

    When `False` only the code for the figure is returned; the return
    value begins with `\begin{tikzpicture}` and ends with
    `\end{tikzpicture}`.

    The default is `False`.

Options that affect only `tkinter()`:

-   `padding` -- A border of this width is added around all sides of the
    diagram. This is affected by `scale`.

    The default value is 3.

-   `offset` -- Cover lines start above the bottom element and end below
    the top element, this controls the separation.

    The default value is 0.5.

-   [`nodeDraw`](#HasseDiagram.nodeDraw) -- When labels is `False` this
    function is called instead of placing anything for the node. The
    function is passed `this` and an index to the element to be drawn.
    [`nodeDraw`](#HasseDiagram.nodeDraw) should use the `tkinter.Canvas`
    object `this.canvas` to draw. The center of your diagram should be
    at the point with coordinates given below.

    ::: center
        x = float(this.loc_x(this,i)) * float(this.scale) + float(this.scale) * \
        float(this.width)/2 + float(this.padding)

        y = 2 * float(this.padding) + float(this.height) * \
        float(this.scale) - (float(this.loc_y(this,i)) * float(this.scale) + \
        float(this.padding))
    :::

    For larger diagrams make sure to increase `height` and `width` as
    well as `offset`.

    The default value is
    [`HasseDiagram.nodeDraw`](#HasseDiagram.nodeDraw).

**Overriding function parameters**

Function parameters can be overriden in two ways. The first option is to
make a function with the same signature as the default function and to
pass that function as a keyword argument to the constructor or
`latex()`/`tkinter()` when called.

For example:

::: center
    def nodeLabel(this, i):
            return str(this.P.mobius(0, this.P[i]))

        #P is a Poset already constructed that has a minimum 0
        P.hasseDiagram.tkinter(nodeLabel = nodeLabel)
:::

The code above will show a Hasse Diagram of `P` with the elements
labeled by the Möbius values $\mu(0,p)$.

When overriding function parameters the first argument is always the
[`HasseDiagram`](#HasseDiagram) instance. The class
[`HasseDiagram`](#HasseDiagram) has an attribute for each of the options
described above as well as the following attributes:

-   `P` -- The poset to be drawn.

-   `in_tkinter` -- Boolean indicating whether `tkinter()` is being
    executed.

-   `in_latex` -- Boolean indicating whether `latex()` is being
    executed.

-   `canvas` -- While `tkinter()` is being executed this is the
    `tkinter.Canvas` object being drawn to.

Note that any function parameters, such as
[`nodeLabel`](#HasseDiagram.nodeLabel), are set via
$$\verb|this.nodeLabel = #provided function|$$ so if you intend to call
these functions you must pass `this` as an argument via
$$\verb|this.nodeLabel(this, i)|$$ The class methods remain unchanged of
course, for example [`HasseDiagram.nodeLabel`](#HasseDiagram.nodeLabel)
always refers to the default implementation.

**Subclassing**

The second way to override a function parameter is via subclassing. This
is more convenient if overriding several function parameters at once or
if the computations are more involved. It is also useful for adding
extra parameters. Any variables initialized in the constructor are saved
at the beginning of `latex()` or `tkinter()`, overriden during execution
of the function by any provided keyword arguments, and restored at the
end of execution. The Möbius example above can be accomplished by
subclassing as follows:

::: center
    class MobiusHasseDiagram(HasseDiagram):

        def nodeLabel(this, i):
            zerohat = this.P.min()[0]
            return str(this.P.mobius(zerohat, this.P[i]))

    P.hasseDiagram = MobiusHasseDiagram(P)
    P.hasseDiagram.tkinter()
:::

To provide an option that changes what element the Möbius value is
computed from just set the value in the constructor.

    class MobiusHasseDiagram(HasseDiagram):

            def __init__(this, P, z = None, **kwargs):
                super().__init__(P, **kwargs)

                if z == None:
                    this.z = this.P.min()[0] #z defaults to first minimal element
                else:
                    this.z = z

            def nodeLabel(this, i):
                return str(this.P.mobius(this.z, this.P[i]))

        #P is a Poset with minimum 0
        P.hasseDiagram = MobiusHasseDiagram(P)
        P.hasseDiagram.tkinter() #labels are $\mu(0, x)$
        P.hasseDiagram.tkinter(z = P[0]) #labels are $\mu(P_0, x)$
        P.hasseDiagram.tkinter() #labels are $\mu(0, x)$

Note you can pass a class to the [`Poset`](#Poset) constructor to
construct a poset with a `hasseDiagram` of that class.

::: child
**[`def __init__(this, P=None, that=None,**kwargs)`]{#HasseDiagram.__init__}**

See HasseDiagram.

**[`def latex(this, **kwargs)`]{#HasseDiagram.latex}**

Returns a string to depict the Hasse diagram in LaTeX.

The keyword arguments are described in [`HasseDiagram`](#HasseDiagram).

**[`def line_options(this,i,j)`]{#HasseDiagram.line_options}**

This is the default implementation of
[`line_options`](#HasseDiagram.line_options), it returns an empty
string.

**[`def loc_x(this, i)`]{#HasseDiagram.loc_x}**

This is the default implementation of [`loc_x`](#HasseDiagram.loc_x).

This spaces elements along each rank evenly. The length of a rank is the
ratio of the natural logarithms of the size of the rank to the size of
the largest rank.

The return value is a string.

**[`def loc_y(this,i)`]{#HasseDiagram.loc_y}**

This is the default value of [`loc_y`](#HasseDiagram.loc_y).

This evenly spaces ranks.

The return value is a string.

**[`def nodeDraw(this, i)`]{#HasseDiagram.nodeDraw}**

This is the default implementation of
[`nodeDraw`](#HasseDiagram.nodeDraw).

This draws a filled black circle of radius $\verb|ptsize|/2$.

**[`def nodeLabel(this,i)`]{#HasseDiagram.nodeLabel}**

This is the default implementation of
[`nodeLabel`](#HasseDiagram.nodeLabel).

The $i$th element is returned cast to a string.

**[`def nodeName(this,i)`]{#HasseDiagram.nodeName}**

This is the default implementation of
[`nodeName`](#HasseDiagram.nodeName).

$i$ is returned cast to a string.

**[`def nodeTikz(this,i)`]{#HasseDiagram.nodeTikz}**

This is the default implementation of nodeTikz used to draw nodes when
`labels` is `False`.

**[`def node_options(this,i)`]{#HasseDiagram.node_options}**

This is the default implementation of
[`node_options`](#HasseDiagram.node_options), it returns an empty
string.

**[`def tkinter(this, **kwargs)`]{#HasseDiagram.tkinter}**

Opens a window using tkinter and draws the Hasse diagram.

The keyword arguments are described in [`HasseDiagram`](#HasseDiagram).

**[`def validate(this)`]{#HasseDiagram.validate}**

Validates and corrects any variables on `this` that may need
preprocessing before drawing.
:::

# SubposetsHasseDiagram {#SubposetsHasseDiagram}

**[`class SubposetsHasseDiagram(HasseDiagram)`]{#SubposetsHasseDiagram}**

This is a class used to draw posets whose elements are themselves
subposets of some global poset, such as interval posets or lattices of
ideals.

The elements of the poset $P$ to be drawn are subposets of a poset $Q$.
The nodes in the Hasse diagram of $P$ are represented as posets. The
entire poset $Q$ is drawn for each element of $P$, the elements and
edges contained in the given subposet are drawn in black and elements
and edges not contained in the subposet are drawn in gray.

Options can be passed to this class in order to control the drawing of
the diagram in the same way as for the class
[`HasseDiagram`](#HasseDiagram). For example, calling `latex(width=5)`
on an instance of [`SubposetsHasseDiagram`](#SubposetsHasseDiagram) sets
the width of the entire diagram (that of $P$) to 5. To control options
for the subposets a prefix, by default `'Q'`, is used. For example,
`latex(Q_width=5,width=40)` would set the width of each subposet to 5
and the width of the entire diagram to 40.

::: child
**[`def Q_nodeName(this, i)`]{#SubposetsHasseDiagram.Q_nodeName}**

Returns a node name for an element of $P$.

To ensure the node names of the larger figure and of the subdiagrams do
not clash all node names are prefixed with `this.prefix`.

**[`def __init__(this, P, Q, is_in=lambda x,X:x in X, prefix=’Q’, draw_min=True, **kwargs)`]{#SubposetsHasseDiagram.__init__}**

Constructor arguments:

-   `prefix` -- String to prefix options to be passed to the instances
    of [`HasseDiagram`](#HasseDiagram) that draw the subdiagrams.

    The argument `prefix` should be a valid tikz node name. It is
    recommended that `prefix` is also a valid python variable name.

-   `is_in` -- A function used by the constructor to test whether
    elements of `Q` are elements of a subposet. The function `is_in`
    takes two arguments, an element `x` of the poset $Q$ and the
    subposet object `X` to test containment with. The default value
    returns `x in X`.

-   `draw_min` -- If `True` all elements of `P` are represented by a
    Hasse diagram. If `False` minimal elements are not drawn but instead
    labeled by the return value of
    [`this.minNodeLabel`](#SubposetsHasseDiagram.minNodeLabel).

All keyword arguments not beginning with the string `this.prefix+'_'`
are handled the same as in the class [`HasseDiagram`](#HasseDiagram).
Keyword arguments that begin with the string `this.prefix+'_'` are saved
as attributes and passed to the instances of
[`HasseDiagram`](#HasseDiagram) drawing the subposets when `latex()` is
called.

**[`def make_line_options(q)`]{#SubposetsHasseDiagram.make_line_options}**

Returns a function to be supplied as
[`line_options`](#HasseDiagram.line_options) in the `latex()` call to
draw a diagram for `q`.

**[`def make_node_options(q)`]{#SubposetsHasseDiagram.make_node_options}**

Returns a function to be supplied as
[`node_options`](#HasseDiagram.node_options) in the `latex()` call to
draw a diagram for `q`.

**[`def minNodeLabel(this)`]{#SubposetsHasseDiagram.minNodeLabel}**

Returns `r'$\emptyset$'`.

This function is called by [`nodeLabel`](#HasseDiagram.nodeLabel) to get
a node label for minimal elements if `draw_min` is `False`. To change
the label for minimal elements provide your own version of
[`minNodeLabel`](#SubposetsHasseDiagram.minNodeLabel).
:::

# Built in posets {#Built in posets}

**[`def Antichain(n)`]{#Antichain}**

Returns the poset on $1,\dots,n$ with no relations.

::: center
![image](figures/antichain_3.pdf)

The poset `Antichain(3)`.
:::

**[`def Bnq(n=2, q=2)`]{#Bnq}**

Returns the poset of subspaces of the vector space $\mathbb{F}_q^n$
where $\mathbb{F}_q$ is the field with q elements.

Currently only implemented for `q` a prime. Raises an instance of
`NotImplementedError` if `q` is not prime.

::: center
![image](figures/Bnq.pdf)

The poset `Bnq(3,2)`.
:::

**[`def Boolean(n)`]{#Boolean}**

Returns the poset of subsets of a set, ordered by inclusion.

The parameter $n$ may be an integer, in which case the poset of subsets
of $\{1,\dots,n\}$ is returned, or an iterable in which case the poset
of subsets of $n$ is returned.

::: center
![image](figures/Boolean_3.pdf)

The poset `Boolean(3)`.
:::

**[`def Bruhat(n,weak=False)`]{#Bruhat}**

Returns the type $A_{n-1}$ Bruhat order (the symmetric group $S_n$) or
the type $A_{n-1}$ left weak order.

::: center
::: center
![image](figures/Bruhat_3.pdf)

The poset `Bruhat(3)`
:::

::: center
![image](figures/Weak_3.pdf)

The poset `Bruhat(3,True)`
:::
:::

**[`def Butterfly(n)`]{#Butterfly}**

Returns the rank $n+1$ bounded poset where ranks $1,\dots,n$ have two
elements and all comparisons between ranks.

::: center
![image](figures/Butterfly_3.pdf)

The poset `Butterfly(3)`.
:::

**[`def Chain(n)`]{#Chain}**

Returns the poset on $0,\dots,n$ ordered linearly (i.e. by usual
ordering of integers).

::: center
![image](figures/chain_3.pdf)

The poset `Chain(3)`.
:::

**[`def Cube(n)`]{#Cube}**

Returns the face lattice of the $n$-dimensional cube.

::: center
![image](figures/cube_2.pdf)

The poset `Cube(2)`.
:::

**[`def DistributiveLattice(P, indices=False)`]{#DistributiveLattice}**

Returns the lattice of ideals of a given poset.

::: center
![image](figures/DL.pdf)

The poset `DistributiveLattice(Root(3))`.
:::

When generating a Hasse diagram with `latex()` use the prefix `irr_` to
control options for the node diagrams.

**[`def Empty()`]{#Empty}**

Returns an empty poset.

**[`def GluedCube(orientations = None)`]{#GluedCube}**

Returns the face poset of the cubical complex obtained from a
$2\times\dots\times2$ grid of cubes of dimension `len(orientations)` via
a series of gluings as indicated by the parameter `orientations`.

If `orientations` is `[1,...,1]` a torus is constructed and if
`orientations` is `[-1,...,-1]` the projective space of dimension $n$ is
constructed.

If `orientations[i] == 1` the two ends of the large cube are glued so
that points with the same image under projecting out the $i$th
coordinate are identified.

If `orientations[i] == -1` points on the two ends of the large cube are
identified with their antipodes.

If `orientations[i]` is any other value no gluing is performed for that
component.

::: center
![image](figures/gluedcube.pdf)

The poset `GluedCube([-1,1])`.
:::

**[`def Grid(n=2,d=None)`]{#Grid}**

Returns the face poset of the cubical complex consisting of a
$\verb|d[0]|\times\dots\times\verb|d[-1]|$ grid of $n$-cubes.

::: center
![image](figures/grid.pdf)

The poset `Grid(2,[1,1])`.
:::

**[`def Intervals(P)`]{#Intervals}**

Returns the lattice of intervals of a given poset (including the empty
interval).

::: center
![image](figures/interval.pdf)

The poset `Intervals(Boolean(2))`.
:::

When generating a Hasse diagram with `latex()` use the prefix `int_` to
control options for the node diagrams.

**[`def KleinBottle()`]{#KleinBottle}**

Returns the face poset of a cubical complex homeomorphic to the Klein
Bottle.

Pseudonym for `GluedCube([-1,1])`.

**[`def LatticeOfFlats(data,as_genlatt=False)`]{#LatticeOfFlats}**

Returns the lattice of flats given either a list of edges of a graph or
the rank function of a (poly)matroid.

When the input represents a graph it should be in the format
`[[i_1,j_1],...,[i_n,j_n]]` where the pair `[i_k,j_k]` represents an
edge between `i_k` and `j_k` in the graph.

When the input represents a (poly)matroid the input should be a list of
the ranks of sets ordered reverse lexicographically (i.e. binary order).
For example, if f is the rank function of a (poly)matroid with ground
set size 3 the input should be
$$\verb|[f({}),f({1}),f({2}),f({1,2}),f({3}),f({1,3}),f({2,3}),f({1,2,3})]|.$$

When `as_genlatt` is `True` the return value is an instance of
[`Genlatt`](#Genlatt) with generating set the closures of singletons.

This function may return a poset that isn't a lattice if the input
function isn't submodular or a preorder that isn't a poset if the input
is not order-preserving.

::: center
![image](figures/lof_triangle.pdf)

The poset `LatticeOfFlats([[1,2],[2,3],[3,1]])`.
:::

::: center
![image](figures/lof_poly.pdf)

The poset `LatticeOfFlats([0,1,2,2,1,3,3,3])`.
:::

**[`def MinorPoset(L,genL=None, weak=False)`]{#MinorPoset}**

Returns the minor poset given a lattice `L` and a list of generators
`genL`, or a list of edges specifying a graph.

The join irreducibles are automatically added to `genL`. If `genL` is
not provided the generating set will be only the join irreducibles.

If `L` is an instance of the [`Poset`](#Poset) class then it is assumed
to be a lattice, an instance of [`Genlatt`](#Genlatt) is created from
`L` and `genL` and the minor poset of the encoded generator-enriched
lattice is returned. In this case the returned poset when plotted with
[`Poset.latex`](#HasseDiagram.latex) has elements represented as
generator-enriched lattices.

If `L` is not an instance of the [`Poset`](#Poset) class it should be an
iterable of length 2 iterables that specify edges of a graph. For
example, `L=[[1,2],[2,3],[3,1]]` specifies the 3-cycle graph. The minor
poset of the graph is returned. In this case when plotting the returned
poset with [`Poset.latex`](#HasseDiagram.latex) the elements are
represented as graphs. Furthermore, there are a few additional options
you can use to control the presentation of the graphs in the Hasse
diagram:

-   `G_scale` -- Scale of the graph, default is 1.

-   `G_pt_size` -- size in points to use for the vertices, default is 2.

-   `G_node_options` -- Options to place on nodes in the graph, default
    is `''`.

-   `G_node_sep` -- String used to separate names of vertices in the
    vertex names for minors, default is `'/'`.

-   `G_label_dist` -- Distance of vertex to its label, default is `1/4`.

-   `G_label_scale` -- Scale factor for the vertex labels, default is 1.

If `weak` is `True` then the weak minor poset is returned. Briefly, this
poset does not have relations $(K,H)\le(M,I)$ when some generator $g$
was deleted to form $(M,I)$ and $g\le\widehat{0}_K$.

For more info on minor posets see [@gustafson-23].

::: center
![image](figures/M_triangle.pdf){width="40%"}

The poset `MinorPoset([[1,2],[2,3],[3,1]])`.
:::

::: center
![image](figures/M_lof_triangle.pdf){width="40%"}
![image](figures/M_lof_triangle_weak.pdf){width="40%"}

On the left the poset `MinorPoset(LatticeOfFlats([[1,2],[2,3],[3,1]]))`
and on the right the poset
`MinorPoset(LatticeOfFlats([[1,2],[2,3],[3,1]]),weak=True)`.
:::

::: center
![image](figures/M_lof_poly.pdf){width="40%"}
![image](figures/M_lof_poly_weak.pdf){width="40%"}

On the left the poset `MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3]))`
and on the right the poset
`MinorPoset(LatticeOfFlats([0,1,2,2,1,3,3,3]), weak=True)`.
:::

::: center
![image](figures/M_B_2.pdf){width="40%"}
![image](figures/M_B_2_weak.pdf){width="40%"}

On the right the poset
`MinorPoset(LatticeOfFlats(Boolean(2),Boolean(2)[1:4]))` and on the left
the poset
`MinorPoset(LatticeOfFlats(Boolean(2),Boolean(2)[1:4]),weak=True)`.
:::

**[`def NoncrossingPartitionLattice(n=3)`]{#NoncrossingPartitionLattice}**

Returns the lattice of noncrossing partitions of $1,\dots,n$ ordered by
refinement.

::: center
![image](figures/NC.pdf)

The noncrossing partition lattice $\text{NC}_4$.
:::

**[`def PartitionLattice(n=3)`]{#PartitionLattice}**

Returns the lattice of partitions of a $1,\dots,n$ ordered by
refinement.

::: center
![image](figures/Pi.pdf)

The partition lattice $\Pi_4$.
:::

**[`def Polygon(n)`]{#Polygon}**

Returns the face lattice of the $n$-gon.

::: center
![image](figures/polygon_4.pdf)

The poset `Polygon(4)`.
:::

**[`def ProjectiveSpace(n=2)`]{#ProjectiveSpace}**

Returns the face poset of a Cubical complex homeomorphic to projective
space of dimension $n$.

Pseudonym for `GluedCube([-1,...,-1])`.

::: center
![image](figures/projectiveSpace.pdf)

The poset `ProjectiveSpace(2)`.
:::

**[`def Root(n=3)`]{#Root}**

Returns the type $A_{n+1}$ root poset.

::: center
![image](figures/root_3.pdf)

The poset `Root(3)`.
:::

**[`def Simplex(n)`]{#Simplex}**

Returns Boolean(n+1) the face lattice of the $n$-dimensional simplex.

**[`def Torus(n=2, m=2)`]{#Torus}**

Returns the face poset of a cubical complex homeomorphic to the
$n$-dimensional Torus.

This poset is isomorphic to the Cartesian product of $n$ copies of $P_m$
with minimum and maximum adjoined where $P_m$ is the face lattice of an
$m$-gon with its minimum and maximum removed.

Let $\ell_m$ be the $m$th letter of the alphabet. When $m\le 26$ the set
is $\{0,1,\dots,m-1,A,B,\dots,\ell_m\}^n$ and otherwise is
$\{0,\dots,m-1,*0,\dots*[m-1]\}^n$. The order relation is componentwise
where $0<A,\ell_m\ 1<A,B\ \dots\ m-1<\ell_{m-1},\ell_m$ for $m\le26$,
and $0<*1,*2\ \dots\  m-1<*[m-1],*0$ for $m>26$.

::: center
![image](figures/torus.pdf)

The poset `Torus(2,2)`.
:::

**[`def Uncrossing(t, upper=False, weak=False, E_only=False, zerohat=True)`]{#Uncrossing}**

Returns either a lower interval $[\widehat{0},t]$ or the upper interval
$[t,\widehat{1}]$ in the uncrossing poset.

The parameter `t` should be either a pairing encoded as a list
`[s_1,t_1,...,s_n,t_n]` where `s_i` is paired to `t_i` or `t` can be an
integer greater than 1. If t is an integer the entire uncrossing poset
of order `t` is returned.

Covers in the uncrossing poset are of the form $\sigma<\tau$ where
$\sigma$ is obtained from $\tau$ by swapping points $i$ and $j$ to
remove a crossing. If `weak` is `True` then the weak subposet is
returned that has cover relations $\sigma<\tau$ when $\sigma$ is
obtained from $\tau$ by removing a single crossing via swapping two
adjacent points. If `E_only` is `True` only swaps $(i,j)$ such that the
pairing $\tau$ satisfies $\tau(i)<i$ and $\tau(j)<j$ are used. These two
flags are provided because this function acts as a backend to
[`Bruhat`](#Bruhat). Calling `Uncrossing(n,E_only=True)` constructs the
Bruhat order on $\mathfrak{S_n}$ and passing `weak=True` constructs the
weak order on $\mathfrak{S}_n$.

If `zerohat` is `False` then no minimum is adjoined.

Raises a `ValueError` when `t` is an integer less than 2.

For more info on the uncrossing poset see [@lam-15].

::: center
![image](figures/uc.pdf)

The poset `Uncrossing(3)==Uncrossing([1,4,2,5,3,6])`.
:::

**[`def UniformMatroid(n=3,r=3,q=1)`]{#UniformMatroid}**

Returns the lattice of flats of the uniform ($q$-)matroid of rank $r$ on
$n$ elements.

Currently only implemented for `q=1` or a prime. Raises an instance of
`NotImplementedError` if `q` is neither 1 nor prime.

::: center
![image](figures/unif.pdf)

The poset `UniformMatroid(4,3)`.
:::

::: center
![image](figures/qunif.pdf)

The poset `UniformMatroid(4,3,2)`.
:::

# Polynomial {#Polynomial}

**[`class Polynomial`]{#Polynomial}**

A barebones class encoding polynomials in noncommutative variables (used
by the [`Poset`](#Poset) class to compute the
[**c**]{.upright}[**d**]{.upright}-index).

This class is basically a wrapper around a dictionary representation for
polynomials (e.g.
$3\textup{\textbf{a}}\textup{\textbf{b}}+2\textup{\textbf{b}}\textup{\textbf{b}}$
is encoded as `{'ab':3, 'bb':2}`). The class provides methods for basic
arithmetic with polynomials, to substitute a polynomial for a variable
in another polynomial and to convert
[**a**]{.upright}[**b**]{.upright}-polynomials to
[**c**]{.upright}[**d**]{.upright}-polynomials (when possible) and vice
versa. You can also get and set coefficients as if a polynomial were a
dictionary.

::: child
**[`def __add__(*args)`]{#Polynomial.__add__}**

Polynomial addition.

**[`def __bool__(this)`]{#Polynomial.__bool__}**

**[`def __eq__(this,that)`]{#Polynomial.__eq__}**

**[`def __ge__(this, that)`]{#Polynomial.__ge__}**

Returns `True` if `this` is coefficientwise greater than or equal to
`that`.

**[`def __getitem__(this,i)`]{#Polynomial.__getitem__}**

**[`def __gt__(this, that)`]{#Polynomial.__gt__}**

Returns `True` if `this` is greater than or equal to `that`
coefficientwise and `this` is not equal to `that`.

**[`def __init__(this, data=None)`]{#Polynomial.__init__}**

Returns a [`Polynomial`](#Polynomial) given either a dictionary or a
list of pairs.

If `data` is a dictionary then the keys are the monomials and the values
are the coefficients. If `data` is a list then all elements should be of
the form `[c,m]` where `c` is a coefficient and `m` is a string
representing a monomial. If `data` is `None` then the zero polynomial is
returned.

**[`def __iter__(this)`]{#Polynomial.__iter__}**

**[`def __le__(this, that)`]{#Polynomial.__le__}**

Returns `True` is `this` is coefficientwise less or equal to `that`.

**[`def __len__(this)`]{#Polynomial.__len__}**

Returns the number of coefficients.

**[`def __lt__(this, that)`]{#Polynomial.__lt__}**

Returns `True` if `this` is coefficientwise less or equal to `that` and
`this` and `that` or not equal.

**[`def __mul__(*args)`]{#Polynomial.__mul__}**

Noncommutative polynomial multiplication.

**[`def __neg__(this)`]{#Polynomial.__neg__}**

**[`def __pow__(this,x)`]{#Polynomial.__pow__}**

Polynomial exponentiation by non-negative integers.

Raises `NotImplementedError` if either `x` is not an integer or `x<0`.

**[`def __add__(*args)`]{#Polynomial.__add__}**

Polynomial addition.

**[`def __repr__(this)`]{#Polynomial.__repr__}**

**[`def __mul__(*args)`]{#Polynomial.__mul__}**

Noncommutative polynomial multiplication.

**[`def __setitem__(this,i,value)`]{#Polynomial.__setitem__}**

**[`def __str__(this)`]{#Polynomial.__str__}**

**[`def __sub__(this, that)`]{#Polynomial.__sub__}**

Polynomial subtraction

**[`def _poly_add_prepoly(p, q)`]{#Polynomial._poly_add_prepoly}**

Internal backend for `__add__`.

**[`def _prepoly_mul_poly(q, p)`]{#Polynomial._prepoly_mul_poly}**

Internal backedn for `__mul__`.

**[`def _to_poly(*args)`]{#Polynomial._to_poly}**

Iterator that converts elements of a list to instances of
[`Polynomial`](#Polynomial).

Internal method used by `__mul__` and `__add__`.

**[`def abToCd(this)`]{#Polynomial.abToCd}**

Given an [**a**]{.upright}[**b**]{.upright}-polynomial returns the
corresponding [**c**]{.upright}[**d**]{.upright}-polynomial if possible
and the given polynomial if not.

**[`def cdToAb(this)`]{#Polynomial.cdToAb}**

Given a [**c**]{.upright}[**d**]{.upright}-polynomial returns the
corresponding [**a**]{.upright}[**b**]{.upright}-polynomial.

**[`def strip(this)`]{#Polynomial.strip}**

Removes any zero terms from a polynomial in place and returns it.

**[`def sub(this, poly, monom)`]{#Polynomial.sub}**

Returns the polynomial obtained by substituting the polynomial `poly`
for the monomial `m` (given as a string) in `this`.
:::

::: child
:::

[^1]: A future update to further automate this is planned.

[^2]: perhaps too much more
