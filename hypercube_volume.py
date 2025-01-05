import numpy as np
from itertools import permutations
from math import log2, floor, pow, factorial, ceil
import matplotlib.pyplot as plt

# returns the ith digit of a binary integer
def ithBinaryDigit(n, i):
	return n // 2**i % 2

def sAsVector(n, l):
	s = np.array([0]*l)
	if n == 0:
		return s
	digits = floor(log2(n))
	index = 0
	for i in range(digits, -1, -1):
		s[l - i - 1] = ithBinaryDigit(n, i)
	return s

# returns the sum of the binary values of a number
def binaryValueSum(n):
	if n == 0:
		return 0
	digits = floor(log2(n))
	s = 0
	for i in range(digits + 1):
		s += ithBinaryDigit(n, i)
	return s

def varSumDensity(a, w):
	n = len(a)
	term1 = 1/((pow(2,n))*np.prod(a))
	term2 = 0
	aSum = sum(a)
	for binNum in range(int(pow(2,n))):
		s = sAsVector(binNum, n)
		aTerm = 2*np.dot(a, s) - aSum - w
		if aTerm <= 0:
			sign = pow(-1, sum(s)) # sign can be optimized by checking simply even or odd rather than calculating summation
			term2 += sign*(pow(-aTerm, n-1)/factorial(n-1))
	return term1 * term2

def varSumCDF(a, w):
	n = len(a)
	term1 = 1/(factorial(n) * np.prod(a))
	term2 = 0
	aSum = sum(a)
	for binNum in range(int(pow(2, n))):
		s = sAsVector(binNum, n)
		sign = pow(-1, sum(s))
		omega = np.dot(a, s)
		prob = (w + aSum)/2 - omega
		if prob > 0:
			sign = pow(-1, sum(s))
			term2 += sign*pow(prob, n)
	return term1 * term2
			
# testing binaryValueSum

# for each array in aColl it generates a new CDF/PDF corresponding to the sum of variables with limit a_i, -a_i for i in the second dimensional array
aColl = [np.array([6, 3, 18])]
for a in aColl:
	x = []
	y1 = []
	y2 = []
	for i in range(-floor(sum(a)) * 100, ceil(sum(a)) * 100 + 1):
		w = i/100
		x.append(w)
		y1.append(varSumCDF(a, w))
		y2.append(varSumDensity(a, w))
	plt.title("CDF; A = %s"%str(a))
	plt.plot(x, y1)
	plt.savefig("randomVarsCDF"+'-'.join(map(lambda x: str(x), a)))
	plt.close()
	plt.title("PDF; A = %s"%str(a))
	plt.plot(x, y2)
	plt.savefig("randomVarsDensity"+'-'.join(map(lambda x: str(x), a)))
	plt.show()
	plt.close()