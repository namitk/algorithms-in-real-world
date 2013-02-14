######################################################################################
# FINITEFIELD.PY
# A finite field of prime power order
# Author: Robert Campbell, <campbell@math.umbc.edu>
# Date: 8 April, 2011
# Version 0.4
# License: Simplified BSD (see details at bottom)
######################################################################################
"""Finite fields.
   Functions implemented are:
          +, *, **, 
   Usage: from finitefield import *     # names do not need path
          GF9 = FiniteField(3,[2,1])    # Define GF(3^2), polys w/ GF3 coeffs, mod x^2+x+2
          a = FiniteFieldElt(GF9,[1,2]) # Define 2x+1 in GF(9)
          a**12                         # Compute (2x+1)**12 in GF(9)"""
   
   
Version = 'FINITEFIELD.PY, version 0.4, 8 April, 2011, by Robert Campbell, <campbell@math.umbc.edu>'

import numbthy  # Use factors
import types # Use IntType, LongType
from operator import add,mul,mod # To allow reduce(add,list) construct for sum

class FiniteField(object):
	"""Finite fields of prime power order.
	Driving polynomial must be monic and top coeff (i.e. 1) is implicit.
	Usage: from finitefield import *     # names do not need path
               GF9 = FiniteField(3,[2,1])    # Define GF(3^2), polys w/ GF3 coeffs, mod x^2+x+2
               a = FiniteFieldElt(GF9,[1,2]) # Define 2x+1 in GF(9)
               a**12                         # Compute (2x+1)**12 in GF(9)"""
	def __init__(self, prime, poly, var='x'):
		self.char = prime
		self.degree = len(poly)
		self.modpoly = poly
		self.var = var
		self.facts_order_gpunits = numbthy.factors(self.char**self.degree - 1)
		self.reduc_table = [[0 for j in range(self.degree)] for i in range(2*self.degree-1)]
		for i in range(self.degree): self.reduc_table[i][i] = 1
		self.reduc_table[self.degree] = [(-self.modpoly[j])%self.char for j in range(self.degree)]
		for i in range(self.degree+1,2*self.degree-1):
			for j in range(self.degree):
				self.reduc_table[i][j] = sum(map(lambda k: (-self.modpoly[k]*self.reduc_table[i-self.degree+k][j]), range(self.degree))) % self.char
	def verbstr(self): # Requires feature from python 2.5.2 or better
		return "Z_"+str(self.char)+"["+self.var+"]/<"+''.join([(((str(self.modpoly[i]) if ((self.modpoly[i]!=1) and (i!=0)) else "")+self.var+"^"+str(i)+"+") if (self.modpoly[i]!=0) else "") for i in range(len(self.modpoly))])+self.var+"^"+str(self.degree)+">"
	def __str__(self):  # Over-ride string conversion used by print
		return "GF("+str(self.char)+"^"+str(self.degree)+")"
	def __repr__(self):  # Over-ride string conversion used by print
		return "FiniteField("+str(self.char)+", "+str(self.modpoly)+")"

class FiniteFieldElt(object):
	"""An element of a prime power order finite fields"
	Driving polynomial must be monic and top coeff (i.e. 1) is implicit.
	Usage: from finitefield import *     # names do not need path, e.g. add(x,y)
               GF9 = FiniteField(3,[2,1])    # Define GF(3^2), polys w/ GF3 coeffs, mod x^2+x+2
               a = FiniteFieldElt(GF9,[1,2]) # Define 2x+1 in GF(9)
               a**12                         # Compute (2x+1)**12 in GF(9)"""
	def __init__(self, field, elts=0):
		self.field = field
		if (type(elts) == type(0)): # Allow simplified form
			self.coeffs = [elts] + [0 for i in range(self.field.degree-1)]
		else:
			self.coeffs = elts + [0 for i in range(self.field.degree - len(elts))]
	def verbstr(self): # Requires feature from python 2.5.2 or better
		return "("+''.join([(((str(self.coeffs[i]) if ((self.coeffs[i]!=1) and (i!=0)) else "")+self.field.var+"^"+str(i)+"+") if (self.coeffs[i]!=0) else "") for i in range(len(self.coeffs))])[:-1]+")"
	def __str__(self):
		"""over-ride string conversion used by print"""
		return str(self.coeffs)
	def __repr__(self):
		"""over-ride string conversion used by print"""
		return "FiniteFieldElt("+self.field.__repr__()+","+str(self.coeffs)+")"
	def __cmp__(self,other):
		"""compare two elements for equality and allow sorting
		overloaded to allow comparisons to integers and lists of integers"""
		if((type(other) == types.IntType) or (type(other) == types.LongType)):
			return cmp(self.coeffs,[other]+[0 for i in range(self.field.degree-1)])
		elif(type(other) == types.ListType):
			return cmp(self.coeffs,other+[0 for i in range(self.field.degree-len(other))])
		elif(self.field != other.field):
			return -1
		else:
			return cmp(self.coeffs,other.coeffs)
	def add(self,summand):
		"""add elements of finite fields (overloaded to allow adding integers and lists of integers)"""
		return FiniteFieldElt(self.field, map(lambda x,y: (x+y)%self.field.char, self.coeffs, summand.coeffs))
	def __add__(self,summand):   # Overload the "+" operator
		if ((type(summand) == types.IntType) or (type(summand) == types.LongType)):
			# Coerce if adding integer and FiniteFieldElt
			return self.add(FiniteFieldElt(self.field,[summand]))
		elif(type(summand) == types.ListType):
			return self.add(FiniteFieldElt(self.field,summand))
		else:
			return self.add(summand)
	def __radd__(self,summand):  # Overload the "+" operator when first addend can be coerced to finfld
		if ((type(summand) == types.IntType) or (type(summand) == types.LongType)): # Coerce if adding int and finfld
			return self.add(FiniteFieldElt(self.field,[summand]))
		elif(type(summand) == types.ListType): # Coerce if adding list and finfld
			return self.add(FiniteFieldElt(self.field,summand))
		else:
			return self.add(summand)
	def __iadd__(self,summand): # Overload the "+=" operator
		self = self + summand
		return self
	def __neg__(self):  # Overload the "-" unary operator 
		return FiniteFieldElt(self.field, map(lambda x: self.field.char-x, self.coeffs))
	def __sub__(self,summand):  # Overload the "-" binary operator 
		return self.__add__(-summand)
	def __isub__(self,summand): # Overload the "-=" operator
		self = self - summand
		return self
# 	def mult(self,multand):  # Elementary multiplication in finite fields
# 		thelist = [0 for i in range(self.field.degree)]
# 		for d in range(2*self.field.degree-2):
# 			thelist = map(add, thelist, [sum(self.coeffs[j]*multand.coeffs[d-j] for j in range(max(0,d-self.field.degree),min(d,self.field.degree-1)+1))*i for i in self.field.reduc_table[d]])
# 		return FiniteFieldElt(self.field,map(lambda x: x%self.field.char, thelist))
	def mult(self,multand):  # Elementary multiplication in finite fields
		"""multiply elements of finite fields (overloaded to allow integers and lists of integers)"""
		thelist = [0 for i in range(self.field.degree)]
		for d in range(2*self.field.degree-1):
			for j in range(max(0,d-(self.field.degree-1)),min(d+1,self.field.degree)):
				list2 = [(self.coeffs[j]*multand.coeffs[d-j])*i for i in self.field.reduc_table[d]]
				thelist = map(add, thelist, list2)
		return FiniteFieldElt(self.field,map(lambda x: x%self.field.char, thelist))
	def __mul__(self,multip):  # Overload the "*" operator
		if ((type(multip) == types.IntType) or (type(multip) == types.LongType)): # Coerce if multiply int and finfld
			return self.mult(FiniteFieldElt(self.field,[multip]))
		elif (type(multip) == types.ListType): # Coerce if multiply list and finfld
			return self.mult(FiniteFieldElt(self.field,multip))
		else:
			return self.mult(multip)
	def __rmul__(self,multip):  # Overload the "*" operator
		if ((type(multip) == types.IntType) or (type(multip) == types.LongType)): # Coerce if mult int and and finfld
			return self.mult(FiniteFieldElt(self.field,[multip]))
		elif (type(multip) == types.ListType): # Coerce if mult list and and finfld
			return self.mult(FiniteFieldElt(self.field,multip))
		return self.mult(multip)
	def __imul__(self,multip): # Overload the "*=" operator
		self = self * multip
		return self
	def inv(self):
		"""inverse of element in a finite field"""
		return FiniteFieldElt(self.field,[numbthy.xgcd(self.coeffs[0],self.field.char)[1] % self.field.char])
	def div(self,divisor):
		"""divide elements of a finite field"""
		return self * inv(divisor)
	def __div__(self,divisor):
		if ((type(divisor) == types.IntType) or (type(divisor) == types.LongType)):
			divisor = FiniteFieldElt(self.field,[divisor])  # Coerce if dividing integer and FiniteFieldElt
		return self / divisor
	def __rdiv__(self,dividend):
		if ((type(dividend) == types.IntType) or (type(dividend) == types.LongType)):
			dividend = FiniteFieldElt(self.field,[dividend])  # Coerce if dividing integer and FiniteFieldElt
		return dividend / self
	def pow(self,exponent):
		"""pow(b,e) computes the eth power of finite field element b."""
		accum = FiniteFieldElt(self.field,[1])
		i = 0
		bpow2 = self
		while ((exponent>>i)>0):
			if((exponent>>i) & 1):
				accum = (accum*bpow2)
			bpow2 = (bpow2*bpow2)
			i+=1
		return accum
	def __pow__(self,exponent): # Overload the "**" operator
		if (exponent < 0):
			self = self.inv()
			exponent = -exponent
		return self.pow(exponent)
		

##################################################################################
# More Examples:
# (ref: Primitive Polynomials over Finite Fields, T. Hansen & G. L. Mullins, Math Comp, Oct 1992)
# GF32 = FiniteField(2,[1,0,1,0,0])        # Define GF(2^5) = Z_2[x]/<x^5 + x^2 + 1>
# GF256 = FiniteField(2,[1,0,1,1,1,0,0,0]) # Define GF(2^8) = Z_2[x]/<x^8 + x^4 + x^3 + x^2 + 1>
# GF27 = FiniteField(3,[1,2,0])            # Define GF(3^3) = Z_3[x]/<x^3 + 2x + 2>
# GF6561 = FiniteField(3,[2,0,0,1,0,0,0,0])# Define GF(3^8) = Z_3[x]/<x^8 + x^3 + 2>
# GF25 = FiniteField(5,[2,1])              # Define GF(5^2) = Z_5[x]/<x^2 + x + 2>
# GF125 = FiniteField(5,[2,3,0])           # Define GF(5^3) = Z_5[x]/<x^3 + 3x + 2>
# GF2197 = FiniteField(13,[6,1,0])         # Define GF(13^3) = Z_13[x]/<x^3 + x + 6>
##################################################################################

############################################################################
# License: Freely available for use, abuse and modification
# (this is the Simplified BSD License, aka FreeBSD license)
# Copyright 2001-2010 Robert Campbell. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in 
#       the documentation and/or other materials provided with the distribution.
############################################################################
