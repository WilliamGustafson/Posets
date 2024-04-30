from posets import Polynomial
import random
import time
import poly

vars = ['a','b','c','d']

def randMonom(d=3):
	return ''.join(vars[random.randrange(len(vars))] for i in range(d))

def randPoly(d=3,length=5,coeffMax=10):
	ret = []
	while(len(ret)<length):
		coeff = 0
		while coeff==0: coeff = random.randrange(-coeffMax,coeffMax)
		monom = randMonom()
		while any(r[1]==monom for r in ret): monom = randMonom()
		ret.append([coeff,monom])
	return Polynomial(ret)

def add1(p,q):
	p.data.sort(key = lambda x:x[1])
	q.data.sort(key = lambda x:x[1])
	r = []

	piter = iter(p)
	qiter = iter(q)

	pi = next(piter)
	qi = next(qiter)

	try:
		while True:
			if pi[1] < qi[1]:
				r.append(pi)
				pi = next(piter)

			elif qi[1] < pi[1]:
				r.append(qi)
				qi = next(qiter)

			else: #pi==qi
				r.append([pi[0]+qi[0],pi[1]])
				pi = next(piter)
				qi = next(qiter)
	except StopIteration:
		if pi == r[-1]: #1st case stopped on pi add q's tail
			try:
				while True:
					r.append(qi)
					qi = next(qiter)
			except: pass
		elif qi == r[-1]: #2nd case stopped on qi add p's tail
			try:
				while True:
					r.append(pi)
					pi = next(piter)
			except: pass
		else:
			if pi[1] == r[-1][1]: #3rd case stopped on pi add q's tail
				try:
					while True:
						r.append(qi)
						qi = next(qiter)
				except: pass
			else: #3rd case stopped on qi add p's tail
				try:
					while True:
						r.append(pi)
						pi = next(piter)
				except: pass
	return Polynomial(r)

def add2(p,q):
	r = {}
	for pi in p:
		if pi[1] in r: r[pi[1]]+=pi[0]
		else: r[pi[1]]=pi[0]
	for qi in q:
		if qi[1] in r: r[qi[1]]+=qi[0]
		else: r[qi[1]]=qi[0]
	return Polynomial([x[::-1] for x in r.items()])

def add3(p,q):
	r = {}
	for pi in p:
		try:
			r[pi[1]]+=pi[0]
		except:
			r[pi[1]] = pi[0]
	for qi in q:
		try:
			r[qi[1]]+=qi[0]
		except:
			r[qi[1]] = qi[0]
	return Polynomial([x[::-1] for x in r.items()])

def add4(p,q):
	p.data.sort()
	q.data.sort()
	r = []

	piter = iter(p)
	qiter = iter(q)

	pi = next(piter)
	qi = next(qiter)

	try:
		if pi < qi:
			r.append(pi)
			pi = next(piter)

		elif qi < pi:
			r.append(qi)
			qi = next(qiter)

		else: #pi==qi
			r.append([pi[0]+qi[0],pi[1]])
	except StopIteration:
		pass
	return Polynomial(r)

n = 10000
#n = 10
P = [randPoly() for i in range(n)]
Q = [randPoly() for i in range(n)]
Z = zip(P,Q)
t = time.perf_counter()
#for p,q in Z: _ = p*q
#for p,q in Z:
#	print('__add__',(p+q).data)
#	print('add1',add1(p,q))
#	print('add2',add2(p,q).data)
#	print('add3',add3(p,q))
#print('__add__',time.perf_counter()-t)

#Polynomial.__add__ = add1
#Z = zip(P,Q)
#t = time.perf_counter()
#for p,q in Z: _ = p+q
#print('add1',time.perf_counter()-t)

#
#Polynomial.__add__ = add2
#Z = zip(P,Q)
#t = time.perf_counter()
#for p,q in Z: _ = p+q
#print('add2',time.perf_counter()-t)


#Polynomial.__add__ = add3
#Z = zip(P,Q)
#t = time.perf_counter()
#for p,q in Z: _ = p+q
#print('add3',time.perf_counter()-t)

P_ = [poly.Polynomial(p.data) for p in P]
Q_ = [poly.Polynomial(q.data) for q in Q]
#Z = zip(P_,Q_)
#t = time.perf_counter()
#for p,q in Z: _ = p+q
#print('poly.Polynomial.__add__',time.perf_counter()-t)

#Z = zip(P,Q)
#t = time.perf_counter()
#for p,q in Z: _ = p*q
#print('add2 multiplication',time.perf_counter()-t)
#
#Z = zip(P_,Q_)
#t = time.perf_counter()
#for p,q in Z: _ = p*q
#print('poly.Polynomial.__mul__',time.perf_counter()-t)

c=Polynomial([[1,'a'],[1,'b']])
c_=poly.Polynomial({'a':1, 'b':1})
t = time.perf_counter()
for p in P:
	_ = p.sub(c, 'c')
print('old sub:', time.perf_counter()-t)

t = time.perf_counter()
for p in P_:
	_ = p.sub(c_, 'c')
print('new sub:', time.perf_counter()-t)
exit()

for p,q in zip(P,Q):
	r = p.sub(c,'c')
	r_ = poly.Polynomial(p.data).sub(c_,'c')
	try:
		assert(poly.Polynomial(r.data)==r_)
	except:
		print('p',p)
		print('old',r)
		print('new',r_)
		print('')
