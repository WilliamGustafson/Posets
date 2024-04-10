# Overview
This is a python package to work with finite posets.
With this package you can

- Construct Boolean algebras, face lattices of cubes, type A Bruhat order and others out of the box.
- Perform operations between posets such as Cartesian products, star producs, disjoint unions, etc.
- Calculate invariants such as flag f and h vectors, ab-index, cd-index and Betti numbers.
- Display the Hasse diagram of a poset or produce latex code for it.
- Convert to and from posets in sage math and Macaulay2.

# Installation

In this directory, run
`python -m pip install .`

# Documentation

See the doc strings in the source code via the help function from a python shell
or produce an html file via `python -m pydoc -w posets`.

# Examples

```python
import * from posets #import the module

B = Boolean(3) #poset of subsets of {1,2,3}
print(B) #shows the zeta matrix, the elements and the ranks list (rank of an element is the length of the longest chain ending at said element)
B.show() #shows the Hasse diagram
B.latex() #gives latex code using tikz for the Hasse diagram, see HasseDiagram.__doc__ for documentation of all the optional arguments

B_ = Poset(relations={0:[1,2,3],1:[12,13],2:[12,23],3:[13,23],12:[123],13:[123],23:[123]}) #create a new poset with the specified relations (you must specify the cover relations, implied relations are automatically added)
B.is_isomorphic(B_) #True
```


Import the module.

`import * from posets`

Construct the rank 3 Boolean algebra.

`B = Boolean(3)`

You can display the Hasse diagram,

`B.show()`

or get latex code for it.

`B.latex()`

See the doc string on `HasseDiagram` for more information.
We can construct a length 1 chain and take Cartesian products.

```python
C = Chain(1)
C3 = C.CartesianProduct(C).CartesianProduct(C)
```

Test whether two posets are isomorphic.

```python
C.is_isomorphic(B) #False
C3.is_isomorphic(B) #True
```

Use `buildIsomorphism` to get an isomorphism as a dictionary, returns `None` when
there is no isomorphism.

Compute the ab-index,
`B.abIndex() #[['aa',1],['ba',2],['ab',2],['bb',1]]`

and the cd-index.

`B.cdIndex() #[['cc',1],['d',1]]`

Compute the Betti numbers,

`B.bettiNumbers() #[1,0,0,0]`
