
This module provides a class `Poset` that encodes a finite partially
ordered set (poset). Most notably, this module can efficiently compute
flag vectors, the **ab**-index and the **cd**-index. Quasigraded posets,
in the sense of [\[2\]](#references), can be encoded and the **ab**-index and
**cd**-index of quasigraded posets can be computed. Latex code for Hasse
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

Terminology and notation on posets generally follows [\[3\]](#references) and [\[1\]](#references).

The full documentation for the current version can be [found
here](https://www.github.com/williamGustafson/posets/releases/latest/download/posets.pdf).

## Installation

Install with pip via `python -m pip install posets`. Alternatively,
download the whl file
[here](https://www.github.com/WilliamGustafson/posets/releases) and
install it with pip via `python -m pip posets-*-py3-none-any.whl`.

## Usage

Here we give an introduction to using the posets module.

In the code snippets below we assume the module is imported via

`from posets import *`

Constructing a poset:

    P = Poset(relations={'':['a','b'],'a':['ab'],'b':['ab']})
    Q = Poset(relations=[['','a','b'],['a','ab'],['b','ab']])
    R = Poset(elements=['ab','a','b',''], less=lambda x,y: return x in y)
    S = Poset(zeta = [[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,0,0]], elements=['','a','b','ab'])

Built in examples (see page ):

    Boolean(3) #Boolean algebra of rank 3
    Cube(3) #Face lattice of the 3-dimensional cube
    Bruhat(3) #Bruhat order on symmetric group of order 3!
    Bnq(n=3,q=2) #Lattice of subspaces of F_2^3
    DistributiveLattice(P) #lattice of ideals of P
    Intervals(P) #meet semilattice of intervals of P

These examples come with default drawing methods, for example, when
making latex code by calling `DistributiveLattice(P).latex()` the
resulting figure depicts elements of the lattice as Hasse diagrams of
$P$ with elements of the ideal highlighted (again, see page ). Note, you
will have to set the `height`, `width` and possibly `nodescale`
parameters in order to get sensible output.

Two posets compare equal when they have the same set of elements and the
same zeta values (i.e. the same order relation with the same weights):

    P == Q and Q == R and R == S #True
    P == Poset(relations={'':['a','b']}) #False
    P == Poset(relations={'':['ab'],'a':['ab'],'b':['ab']}) #False
    P == Poset(zeta=[[0,1,1,2],[0,0,0,3],[0,0,0,4],[0,0,0,0]],
        elements=['','a','b','ab']) #False

Use `is_isomorphic` or `PosetIsoClass` to check whether posets are
isomorphic:

    P.is_isomorphic(Boolean(2)) #True
    P.isoClass()==Boolean(2).isoClass() #True
    P.is_isomorphic(Poset(relations={'':['a','b']})) #False

Viewing and creating Hasse diagrams:

    P.show() #displays a Hasse diagram in a new window
    P.latex() #returns latex code: \begin{tikzpicture}...
    P.latex(standalone=True) #latex code for a
    #standalone document: \documentclass{preview}...
    display(P.img()) #Display a poset when in a Jupyter notebook
    #this uses the output of latex()

Computing invariants:

    Cube(2).fVector() #{(): 1, (1,): 4, (2,): 4, (1, 2): 8}
    Cube(2).hVector() #{(): 1, (1,): 3, (2,): 3, (1, 2): 1}
    Boolean(5).sparseKVector() #{(3,): 8, (2,): 8, (1, 3): 4, (1,): 3, (): 1}
    Boolean(5).cdIndex() #Polynomial({'ccd': 3, 'cdc': 5, 'dd': 4, 'dcc': 3, 'cccc': 1})
    print(Boolean(5).cdIndex()) #c^{4}+3c^{2}d+5cdc+3dc^{2}+4d^{2}

Polynomial operations:

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

Converting posets to and from SageMath:

    P.toSage() #Returns a SageMath class, must be run under sage
    Poset.fromSage(Q) #Take a poset Q made with SageMath and return an instance of Poset

Converting to and from Macaulay2:

    -- In M2
    load "convertPosets.m2" --Also loads Python and Posets packages
    import "posets" --This module must be installed to system version of python
    P = posets@@Boolean(3) --Calling python functions
    pythonPosetToMac(P) --Returns an instance of the M2 class Posets
    macPosetToPython(Q) --Take a poset made with M2 and return an
    --instance of the python class Poset

Quasigraded posets:

    #Provide the zeta and rank functions explicitly
    #To construct a 2-chain with top two elements rank 2 and 3
    #and with zeta value -1 between minimum and the element covering it:
    T = Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]])

The poset `T` above is from \[2, Example 6.14\] with $M$ taken to be the
3-dimensional solid torus.

You can calculate the flag vectors and the **cd**-index just as you
would for a classical poset, for example, `T.cdIndex()` returns the
polynomial $\textbf{c}^2-2\textbf{d}$.

When plotting a quasigraded poset by default only the underlying poset
is shown with element heights based on rank, the zeta values are not
shown. If you wish to display the zeta values you can use the class
`ZetaHasseDiagram` to draw a Hasse diagram of your poset with an element
$p$ depicted as the associated filter, namely the subposet
$\\{q:q\ge p\\}$, and with elements of the filters labeled by the
corresponding zeta value. To do so, either construct the poset with
`hasse_class=ZetaHasseDiagram` such as in
`Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]],hasse_class=ZetaHasseDiagram)`
or set the Hasse diagram attribute on the poset as below:

    T = Poset([[1,-1,1],[1,1],[1]], ranks=[[0],[],[1],[2]])
    T.hasseDiagram = ZetaHasseDiagram(T)

You can also represent elements with ideals instead of filters by
passing `filters=False`. See `ZetaHasseDiagram` and
`SubposetsHasseDiagram` for a thorough explanation of the options.

# References
<div id="refs" class="references csl-bib-body">

<div id="ref-birkhoff-67" class="csl-entry">

<span class="csl-left-margin">1.
</span><span class="csl-right-inline">Garrett Birkhoff. 1967. *Lattice
theory*. American Mathematical Society, Providence, R.I.</span>

</div>

<div id="ref-ehrenborg-goresky-readdy-15" class="csl-entry">

<span class="csl-left-margin">2.
</span><span class="csl-right-inline">Richard Ehrenborg, Mark Goresky,
and Margaret Readdy. 2015. Euler flag enumeration of whitney stratified
spaces. *Adv. Math. (N. Y.)* 268: 85–128.</span>

</div>

<div id="ref-stanley-12" class="csl-entry">

<span class="csl-left-margin">3.
</span><span class="csl-right-inline">Richard P Stanley. 2012.
*Enumerative combinatorics. Volume 1*. Cambridge University Press,
Cambridge.</span>

</div>

</div>
