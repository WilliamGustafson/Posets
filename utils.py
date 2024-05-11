import decorator
import time
class Timer:
	def __init__(this, running=True):
		this.elapsed_time = 0
		this.running = False
		if running: this.start()

	def start(this):
		this.running = True
		this.start_time = time.time()

	def stop(this):
		this.stop_time = time.time()
		if this.running: this.elapsed_time += this.stop_time - this.start_time
		this.running = False

	def pause(this):
		the_time = time.time()
		this.elapsed_time += the_time - this.start_time
		this.running = False

	def reset(this, *args):
		this.stop()
		this.elapsed_time = 0
		this.start()

	def __str__(this):
		return str(this.elapsed_time)

@decorator.decorator
def cached_method(f, *args, **kwargs):
	'''
	This is an attribute for class functions that will cache the return result in a cache property on the object.

	When the attribute is placed on a class function e.g.

		@cached_method
		def f(this,...):

	calls to the function will first check the dictionary this.cache for the function call
	and if it is stored instead of calling the function the cache value is returned.
	If the value is not present the function is called then the return value is cached
	and then that value is returned.

	The key in this.cache for a function call f(this,'a',7,[1,2,3],b= ['1'],a = 3) is
		"f('a', 7, [1,2,3], a=3, b=['1'])"
	The key for a function call f(this) is 'f()'.

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

def requires(mod):
	@decorator.decorator
	def f(g, *args, **kwargs):
		if type(mod)==str:
			def ret(*args, **kwargs):
				raise NotImplementedError("You must install "+mod+" to use "+g.__name__)
		else:
			def ret(*args, **kwargs):
				return g(*args, **kwargs)
		return ret(*args, **kwargs)
	return f

def log_func(f):
	def g(*args, **kwargs):
		print(f.__name__,*['\t'+str(a) for a in args],*['\t'+k+'='+str(v) for (k,v) in kwargs.items()],sep='\n')
		ret = f(*args,**kwargs)
		print('\treturn value:',ret)
		return ret
	return g
