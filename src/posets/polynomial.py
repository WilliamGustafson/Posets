##############
#Polynomial class
##############
import math
import itertools

def iter_join(i, x):
	'''
	Concatenates two iterators.
	'''
	try:
		yield next(i)
		yield x
	except StopIteration:
		pass

class Polynomial:
	r'''@is_section@
	A barebones class encoding polynomials in noncommutative variables (used by the \verb|Poset| class to compute the cd-index).

	This is basically a wrapper around a dictionary representation for polynomials (e.g. $3ab+2bb$ is encoded as \verb|{'ab':3, 'bb':2}|)
	and provides methods to add, multiple, subtract polynomials, to substitute a polynomial
	for a variable in another polynomial and to convert ab-polynomials to cd-polynomials (when possible) and vice versa. You can also get and set
	coefficients as if this were a dictionary.
	'''
	def __init__(this, data):
		r'''
		Returns a \verb|Polynomial| given a list of pairs \verb|[c,m]| with \verb|c| a coefficient and \verb|m| a string representing a monomial.
		'''
		this.data = data if type(data)==dict else {d[1]:d[0] for d in data}

	def __mul__(*args):
		'''
		Noncommutative polynomial multiplication.
		'''
		return Polynomial.__add__(
			*(
			Polynomial(
				{''.join(map(lambda y:y[0],x)) :
				math.prod(map(lambda y:y[1],x))}
				)
			for x in itertools.product(*map(lambda y:y.data.items(),args))
			)
			)

	def __add__(*args):
		'''
		Polynomial addition.
		'''
		ret = {}
		for p in args:
			for m,c in p.data.items():
				if m in ret: ret[m]+=c
				else: ret[m]=c
		return Polynomial(ret)

	def __neg__(this):
		return Polynomial({m:-c for m,c in this.data.items()})

	def __sub__(this, that):
		return this+(-that)

	def __ge__(this, that):
		if isinstance(that, int):
			return all(v >= that for v in this.data.values())
		if isinstance(that, Polynomial):
			return all((m in this and this[m]>=that[m]) or (that[m]<=0) for m in that)

	def __gt__(this, that):
		if isinstance(that, int):
			return all(v > that for v in this.data.values())
		if isinstance(that, Polynomial):
			return this!=that and all((m in this and this[m]>=that[m]) or (that[m]<=0) for m in that)

	def __le__(this, that):
		if isinstance(that, int):
			return all(v <= that for v in this.data.values())
		if isinstance(that, Polynomial):
			return all((m in this and this[m]<=that[m]) or (that[m]>=0) for m in that)

	def __lt__(this, that):
		if isinstance(that, int):
			return all(v < that for v in this.data.values())
		if isinstance(that, Polynomial):
			return this!=that and all((m in this and this[m]<=that[m]) or (that[m]>=0) for m in that)


	def sub(this, poly, monom):
		r'''
		Returns the polynomial obtained by substituting the \verb|Polynomial| \verb|poly| for the monomial \verb|m| (given as a string) in \verb|this|.
		'''
		ret = Polynomial({}) #initialize to zero
		for m,c in this.data.items():
			r = [['',c]] #term to add to ret, starts as coefficient
			monom_iter = iter(monom)
			curr_char = next(monom_iter)
			last_char_ind = 0 #start of char block being read
			curr_ind = 0 #current index
			for v in m: #loop through chars looking for monom
				curr_ind += 1
				if v==curr_char:
					try:
						curr_char = next(monom_iter)
					except StopIteration:
						r = Polynomial._prepoly_mul_poly(r,poly)
						monom_iter = iter(monom)
						curr_char = next(monom_iter)
						last_char_ind = curr_ind
				else:
					s = m[last_char_ind:curr_ind]
					for x in r: x[0]+=s #multiply r by the variables in the buffer
					last_char_ind = curr_ind
			#if m ended in a partial match need to dump buffer
			if last_char_ind != curr_ind:
				s = m[last_char_ind:]
				for x in r: x[0]+=s
			Polynomial._poly_add_prepoly(ret,r) #add the poly to ret
		ret.data = {k:v for k,v in ret.data.items() if v!=0}
		return ret

	def _poly_add_prepoly(p, q):
		for m,c in q:
			p[m] = c + (p[m] if m in p else 0)
	def _prepoly_mul_poly(q, p):
		return [[x[0][0]+x[1][0], x[0][1]*x[1][1]] for x in itertools.product(q,p.data.items())]

	def abToCd(this):
		'''
		Given an ab-polynomial returns the corresponding cd-polynomial if possible and the given polynomial if not.
		'''
		if len(this.data)==0: return this
		#substitue a->c+e and b->c-e
		#where e=a-b
		#this scales by a factor of 2^deg
		ce = this.sub(Polynomial({'c':1,'e':1}),'a').sub(Polynomial({'c':1,'e':-1}),'b')

		cd = ce.sub(Polynomial({'cc':1,'d':-2}),'ee')
		#check if any e's are still present
		for m in cd.data:
			if 'e' in m:
				return this
		#divide coefficients by 2^n
		for monom in cd.data.keys(): break #grab a monomial
		power=sum(2 if v=='d' else 1 for v in monom)
		return Polynomial({k:v>>power for k,v in cd.data.items()})

	def cdToAb(this):
		'''
		Given a cd-polynomial returns the corresponding ab-polynomial.
		'''
		return this.sub(Polynomial({'a':1,'b':1}, 'c')).sub(Polynomial({'ab':1,'ba':1}))

	def __len__(this):
		return len(this.data)

	def __iter__(this):
		return iter(this.data)

	def __getitem__(this,i):
		return this.data[i]

	def __setitem__(this,i,value):
		this.data[i] = value

	def __str__(this):
		data = list(this.data.items())
		data.sort(key=lambda x:x[0])
		s = ""
		for i in range(0,len(data)):
			if data[i][1] == 0: continue
			if data[i][1] == -1: s+= '-'
			elif data[i][1] != 1: s += str(data[i][1])
			current = ''
			power = 0
			for c in data[i][0]:
				if current == '':
					current = c
					power = 1
					continue
				if c == current:
					power += 1
					continue
				s += current
				if power != 1: s += '^{' + str(power) + '}'
				current = c
				power = 1
			s += current
			if power != 1 and power != 0: s += '^{' + str(power) + '}'
			if power == 0 and current == "": s += '1'

			if i != len(data)-1:
				if data[i+1][1] >= 0: s += "+"
		if s == '': return '0'
		return s

	def __repr__(this):
		return 'Polynomial('+repr(this.data)+')'

	def __eq__(this,that):
		return this.data == that.data



##############
#End Polynomial class
##############
