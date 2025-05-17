'''
@no_doc@no_children@
'''
import decorator
import time
import itertools

def subsets(S):
	r'''
	Iterator for subsets of an iterable.

	@no_doc@
	'''
	for T in itertools.chain(*(itertools.combinations(S,k) for k in range(len(S)+1))):
		yield T

@decorator.decorator
def cached_method(f, *args, **kwargs):
	r'''
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

	@no_doc@
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
	r'''
	Iterator for a triangle of numbers $n,n-1,\dots,1,n-1,\dots,1,n-2,\dots,1,\dots,2,1,1$.

	@no_doc@no_children@
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
	A class encoding a triangular array.

	This class is used to encode the zeta function of a poset.

	Constructor arguments:
	\begin{itemize}
		\item[]{\verb|data| -- An iterable specifying the entries in the upper diagonal; may be either a flat list or an iterable of iterables.}
		\item[]{\verb|flat| -- Whether the data is in flat form or not.}
		\end{itemize}
	Constructor raises \verb|ValueError| if the number of entries of \verb|data| is not a triangle number.

	@is_section@
	'''
	def __init__(this, data, flat=True):
		if flat:
			this.data = [x for x in data]
		else:
			this.data = list(itertools.chain(*data))
		size = int(0.5 + (2*len(this.data)+0.25)**0.5)-1
		assert(triangle_num(size+1) == len(this.data))
		this.size = size
	def __setitem__(this, idx, x):
		this.data[idx[1] - idx[0] + this.size*idx[0] - triangle_num(idx[0])] = x
	def __eq__(this,that):
		return isinstance(that,TriangularArray) and this.data == that.data
	def row(this, i):
		r'''
		Returns the $i$th row as a list.
		'''
		return this.data[this.size*i - triangle_num(i) : this.size*(i+1) - i - triangle_num(i)]
	def col(this, j):
		r'''
		Generator for the $i$th column.
		'''
		data = this.data
		n = this.size
		idx = j
		i = 0
		while i<=j:
			yield data[idx]
			i+=1
			idx += n-i

	def __getitem__(this, x):
		r'''
		Zero based indexing $(i,j)$  gives the element in row $i$ and column $j$.

		The argument \verb|x| must be a tuple of integers such that $0\le x_0\le x_1< n$ where $n$ is the size of the triangular array.
		'''
		return this.data[x[1] - x[0] + this.size*x[0] - triangle_num(x[0])]
		#if isinstance(x,tuple): return this.data[this.size*(x[0]-1)-triangle_num(x[0]+1)+x[1]-1]

	def __str__(this):
		if this.size==0: return ''
		space_len = max(len(str(entry)) for entry in this.data)
		ret = []
		for i in range(this.size):
			row = this.row(i)
			ret.append(' '*i*(space_len+1)+' '.join(' '*(space_len-len(str(x)))+str(x) for x in row))
#			ret.append(' '*i*(space_len+1)+ ' '.join(('{x:'+str(space_len)+'}').format(x=x) for x in row))
		return '\n'.join(ret)
	
	def __repr__(this):
		return 'TriangularArray(('+', '.join(repr(x) for x in this.data)+'), flat=True)'

	def revtranspose(this):
		r'''
		Returns a new instance of \verb|TriangularArray| obtained by transposing and reversing all columns and rows.
		'''
		return TriangularArray([this.data[len(this.data) - i - triangle_num(j) - j - 1] for i in range(this.size) for j in range(i,this.size)])

	def subarray(this, S):
		r'''
		Returns a sub-triangular array by selecting the rows and columns indexed by \verb|S|.
		'''
		S = sorted(S)
		return TriangularArray([this.data[t+s*(this.size)-triangle_num(s+1)] for i,s in enumerate(S) for t in S[i:]])
	
	def inverse(this):
		r'''
		Returns the inverse of the triangular array considered as an upper triangular matrix.

		Raises \verb|ZeroDivisionError| if a diagonal entry of the array is zero.
		'''
		#\alpha(x,y) = -1/\beta(y,y)\sum_{x< z\le y}\alpha(x,z)\beta(z,y)
		data = []
		idx = 0
		for i in range(this.size):
			data.append(1/this[i,i])
			idx+=1
			for j in range(i+1,this.size):
				col = list(this.col(j))
				data.append(-1/this[j,j] * sum(data[idx-k]*col[j-k] for k in range(1,j-i+1)))
				idx+=1
		return TriangularArray(data)
				
##############
#End TriangularArray class
##############
class MockSet:
	'''
	@no_doc@no_children@
	'''
	def add(this,x):
		pass
