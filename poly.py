##############
#Polynomial class
##############
import math
import itertools
class Polynomial:
	'''
	A barebones class encoding polynomials in noncommutative variables (used by Poset class to compute the cd-index).

	This is basically a wrapper around a list representation for polynomials (e.g. 3ab+2bb <--> [[3,'ab'],[2,'bb']]
	and provides methods to add, multiple, subtract polynomials, to substitute a polynomial
	for a variable in another polynomial and to convert ab-polynomials into cd-polynomials (when possible).
	'''
	def __init__(this, data):
		'''
		Returns a Polynomial given a list of pairs [c,m] with c a coefficient and m a string representing a monomial.
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


	def sub(this, p, m, filler_char = ' '):
		'''
		Returns the polynomial obtained by substituting the Polynomial p for the monomial m (given as a string) in this.

		this, p and m should not have any variable containing the filler character filler_char
		'''
		X=[[y[0],y[1].replace(m,'*')] for y in this]
		ret=Polynomial([]) #0
		for y in X:
			q=Polynomial([[y[0],'']])
			for i in range(0,len(y[1])):
				if y[1][i]=='*':
					q = q*p
				else: #mult by the monomial
					for j in range(0,len(q)):
						q[j][1]+=y[1][i]
			ret += q
		return Polynomial(ret)

	def __len__(this):
		return len(this.data)

	def __iter__(this):
		return iter(this.data)

	def __getitem__(this,i):
		return this.data[i]

	def __setitem__(this,i,value):
		this.data[i] = value

	def abToCd(this):
		'''
		Given an ab-polynomial return the corresponding cd-polynomial if possible and the given polynomial if not.
		'''
		if len(this)==0: return this
		#substitue a->c+e and b->c-e
		#where e=a-b
		#this scales by a factor of 2^deg
		ce = this.sub(Polynomial([[1,'c'],[1,'e']]),'a').sub(Polynomial([[1,'c'],[-1,'e']]),'b')

		cd = ce.sub(Polynomial([[1,'cc'],[-2,'d']]),'ee')
		#check if any e's are still present
		for m in cd:
			if 'e' in m[1]:
				return this
		#divide coefficients by 2^n
		power=sum([2 if cd[0][1][i]=='d' else 1 for i in range(len(cd[0][1]))])
		return Polynomial([[x[0]>>power,x[1]] for x in cd])

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
