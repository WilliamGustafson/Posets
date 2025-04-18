'''
@no_doc@
'''
import decorator
import time
import itertools

def subsets(S):
	r'''
	@no_doc@
	Iterator for subsets of an iterable.
	'''
	for T in itertools.chain(*(itertools.combinations(S,k) for k in range(len(S)+1))):
		yield T

@decorator.decorator
def cached_method(f, *args, **kwargs):
	r'''
	@no_doc@
	This is an attribute for class functions that will cache the return result in a cache property on the object.

	When the attribute is placed on a class function e.g.
	\begin{verbatim}
		\@cached_method
		def f(this,...):
	\end{verbatim}
	calls to the function will first check the dictionary this.cache for the function call
	and if it is stored instead of calling the function the cache value is returned.
	If the value is not present the function is called then the return value is cached
	and then that value is returned.

	The key in this.cache for a function call
	\begin{center}
		\verb|f(this,'a',7,[1,2,3],b= ['1'],a = 3)|
	\end{center}
	is
	\begin{center}
		\verb|"f('a', 7, [1,2,3], a=3, b=['1'])"|
	\end{center}
	The key for a function call \verb|f(this)| is \verb|'f()'|.

	This attribute should not be placed on a function whose return value is not solely
	dependent on its arguments (including the argument this).
	'''
	key = f.__name__ + str(args[1:])[:-1] + ((', ' + ', '.join([str(k)+'='+repr(v) for (k,v) in sorted(kwargs.items(),key = lambda x:x[1])])) if len(kwargs)>0 else '')+')'
	try:
		if key in args[0].cache:
			return args[0].cache[key]
	except:
		pass
	ret = f(*args, **kwargs)
	try:
		args[0].cache[key] = ret
	except:
		pass
	return ret

def triangle_num(n):
	r'''@no_doc@'''
	return n*(n-1) // 2

class TriangleRange:
	r'''@no_doc@
	Iterator for a triangle of numbers $n,n-1,\dots,1,n-1,\dots,1,n-2,\dots,1,\dots,2,1,1$.
	'''
	def __init__(this, n):
		this.n = n-1
		this.row = 0
		this.col = -1
	
	def __iter__(this):
		return this
	
	def __next__(this):
		if this.col == this.n-this.row:
			if this.row == this.n: raise StopIteration
			this.row+= 1
			this.col = 0
		else:
			this.col+=1
		return this.col
##############
#TriangularArray class
##############
class TriangularArray:
	r'''
	@is_section@
	A class encoding an element of the incidence algebra of a poset.

	This class is mainly provided for the zeta element of a poset, but
	is essentially just a triangular array.

	Constructor arguments:
	\begin{itemize}
		\item[]{\verb|data| -- An iterable specifying the entries in the upper diagonal; may be either a flat list or an iterable of iterables.}
		\item[]{\verb|rows| -- Number of rows of the data, must be provided if the data is in flat form.}
		\item[]{\verb|flat| -- Whether the data is in flat form or not.}
		\end{itemize}
	'''
	def __init__(this, data, size=0, flat=True):
		if flat:
			this.size = size
			this.data = [x for x in data]
		else:
			this.size = len(data) if size==0 else size
			this.data = list(itertools.chain(*data))
		assert(triangle_num(this.size+1) == len(this.data))
	def __getitem__(this, x):
		r'''
		Zero based indexing \verb|(i,j)| gives the element in row $i$ and column $j$.
		'''
		if isinstance(x,int): return this.data[this.size*x - triangle_num(x) : this.size*(x+1) - x - triangle_num(x)]
		if isinstance(x,tuple): return this.data[x[1] - x[0] - 1 + this.size*x[0] - triangle_num(x[0])]
		#if isinstance(x,tuple): return this.data[this.size*(x[0]-1)-triangle_num(x[0]+1)+x[1]-1]

	def __str__(this):
		if this.size==0: return ''
		space_len = max(len(str(entry)) for entry in this)
		ret = []
		for i in range(this.size):
			row = this[i]
			ret.append(' '*i*(space_len+1)+ ' '.join(('{x:'+str(space_len)+'}').format(x=x) for x in row))
		return '\n'.join(ret)
	def __iter__(this):
		return iter(this.data)
	
	def __repr__(this):
		return 'TriangularArray(('+', '.join(repr(x) for x in this)+'), flat=True,size='+str(this.size)+')'

	def revtranspose(this):
		r'''
		Returns a new instance of \verb|TriangularArray| by reversing columns and then transposing.
		'''
		return TriangularArray([this.data[len(this.data) - i - triangle_num(j) - j - 1] for i in range(this.size) for j in range(i,this.size)],size=this.size)

	def subarray(this, S):
		'''
		Returns a sub-triangular array with rows indexed by \verb|S[:-1]| and columns by \verb|S[1:]+1|.
		'''
		S = sorted(S)
		return TriangularArray([this.data[s + S[i]*(this.size-1)-triangle_num(S[i])-1] for i in range(len(S)-1) for s in S[i+1:]], size=len(S)-1)
##############
#End TriangularArray class
##############
class MockSet:
	def add(this,x):
		pass
