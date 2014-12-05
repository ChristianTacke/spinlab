from __future__ import division
import numpy
from numpy import exp, expm1, outer
from numpy.linalg import eig
from qmlib import recombine_eig

		

from qmlib import I2y, I2z, I4z, I4x, I4y, dot, tensorproduct
from qmlib import prepare_equil
from numpy import eye
from numpy import pi

if False:
	s1z = tensorproduct(I2z, eye(2))
	s2z = tensorproduct(eye(2), I2z)

	dens = prepare_equil(42e7 * 2*pi * (s1z + 1./7 * s2z), 300)
	print dens
	print dens.trace()

	print (dot(s2z, dens)).trace()

if True:
	dens = prepare_equil(42e7 * 2*pi * I2y, 300)
	print dens
	print dens.trace()

print "---"
f = I2y
print "Old:"
print f
(vals, vecs) = eig(f)
print "Vals, vecs:"
print vals
print vecs
f = recombine_eig(vals, vecs)
print "reconstructed:"
print f
