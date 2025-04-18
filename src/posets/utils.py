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

