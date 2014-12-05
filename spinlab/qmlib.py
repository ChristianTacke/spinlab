from __future__ import division
import numpy
from numpy import inner, array, sqrt, exp, newaxis, eye
from numpy.linalg import eig, eigh

# Use eigh possibly.

if False:
	def dot(a, b):
		# print "."
		# inner() does rows by rows, not rows by column.
		return inner(a, b.T)
else:
	from numpy import dot

def commutator(a, b):
	return dot(a, b) - dot(b, a)

def recombine_eig_old(vals, vecs):
	E = numpy.diag(vals)
	return dot(dot(vecs, E), vecs.conj().T)

def recombine_eig(vals, vecs):
	return dot(vecs * vals[newaxis, :], vecs.conj().T)

def my_tensorproduct(a, b):
	assert(len(a.shape) == 2)
	assert(len(b.shape) == 2)
	newdtype = (a[0:0, 0:0] * b[0:0, 0:0]).dtype
	a_shape = array(a.shape)
	b_shape = array(b.shape)
	r = numpy.empty(a_shape*b_shape, dtype=newdtype)
	k = 0
	linelen = a_shape[1] * b_shape[1]
	for i in xrange(a_shape[0]):
	    for j in xrange(b_shape[0]):
		t = numpy.outer(a[i, :], b[j, :]).reshape((linelen,))
		# print a[i, :], b[j, :], t
		r[k, :] = t
		k += 1
	return r

from numpy import kron as tensorproduct


def rotate_spin_set(I, rot_1, rot_2, rot_3 = 0):
	from numpy import sin, cos
	# Rotate around Z
	if rot_1 != 0:
		I = \
		  (I[0] * cos(rot_1) - I[1] * sin(rot_1), \
		   I[1] * cos(rot_1) + I[0] * sin(rot_1), \
		   I[2])
	# Rotate around Y
	if rot_2 != 0:
		I = \
		  (I[0] * cos(rot_2) + I[2] * sin(rot_2), \
		   I[1], \
		   I[2] * cos(rot_2) - I[0] * sin(rot_2))
	# Rotate around Z
	if rot_3 != 0:
		I = \
		  (I[0] * cos(rot_3) - I[1] * sin(rot_3), \
		   I[1] * cos(rot_3) + I[0] * sin(rot_3), \
		   I[2])
	return I

def H_quadrupole(Iset, I, w_Q, eta):
	# Ernst, page 49
	(Ix, Iy, Iz) = Iset
	eta = eta / 3. # <<--- !!!! this is good and right!
	# I = (Ix.shape[0] - 1) / 2.
	# print Ix.shape[0], I
	return w_Q * \
		(dot(Iz, Iz) - eye(Ix.shape[0]) * 1./3 * I * (I+1) \
		 + eta * (dot(Ix, Ix) - dot(Iy, Iy)))

def H_quadrupole_tensor_set(I1set, I, w_Q):
	a_1x, a_1y, a_1z = I1set
	a_1plus  = a_1x + 1j*a_1y
	a_1minus = a_1x - 1j*a_1y

	# So that we have the usual definition of w_Q:
	w_Q *= sqrt(6)/3

	a_H_m2 = 0.5 * dot(a_1minus, a_1minus)
	a_H_p2 = 0.5 * dot(a_1plus, a_1plus)

	# print a_H_m2 + a_H_p2 - (dot(a_1x, a_1x) - dot(a_1y, a_1y))
	a_H_m1 =  0.5 * (dot(a_1z, a_1minus) + dot(a_1minus, a_1z))
	a_H_p1 = -0.5 * (dot(a_1z, a_1plus) + dot(a_1plus, a_1z))

	a_H_0 = 1/sqrt(6) * (3*dot(a_1z, a_1z) - eye(a_1z.shape[0]) * I * (I+1))

	a_H_m2 *= w_Q
	a_H_m1 *= w_Q
	a_H_0 *= w_Q
	a_H_p1 *= w_Q
	a_H_p2 *= w_Q

	return (a_H_m2, a_H_m1, a_H_0, a_H_p1, a_H_p2)

def H_dipoledipole(I1set, I2set, w_dd):
	# See for example:
	# (simple) Spiess, page 20, Eq. 2.21
	# Ernst, page 47, insert theta = 0.
	(s1x, s1y, s1z) = I1set
	(s2x, s2y, s2z) = I2set
	return w_dd * ( \
	       dot(s1x, s2x) \
	   +   dot(s1y, s2y) \
	   - 2*dot(s1z, s2z))


def H_dipoledipole_tensor_set(I1set, I2set, w_dd):
	a_1x, a_1y, a_1z = I1set
	a_2x, a_2y, a_2z = I2set
	a_1plus  = a_1x + 1j*a_1y
	a_1minus = a_1x - 1j*a_1y
	a_2plus  = a_2x + 1j*a_2y
	a_2minus = a_2x - 1j*a_2y

	a_H_m2 = -3./4 * dot(a_1minus, a_2minus)
	a_H_p2 = -3./4 * dot(a_1plus, a_2plus)
	a_H_m1 = -3./2 * \
	    ( dot(a_1z, a_2minus) + dot(a_1minus, a_2z) )
	a_H_p1 = -3./2 * \
	    ( dot(a_1z, a_2plus) + dot(a_1plus, a_2z) )
	a_H_0 = (dot(a_1z, a_2z) - \
	    1./4 * (dot(a_1plus, a_2minus) + dot(a_1minus, a_2plus)))
	a_H_m2 *= w_dd
	a_H_m1 *= w_dd
	a_H_0 *= w_dd
	a_H_p1 *= w_dd
	a_H_p2 *= w_dd
	return (a_H_m2, a_H_m1, a_H_0, a_H_p1, a_H_p2)


def calc_w_dd(gamma1, gamma2, r_dist):
	"(mu0 / (4*pi)) * (gamma1*gamma2*h_bar / (r_dist^3))"
	# Really probably wrong number of h_bar !
	return (mu0 / (4*pi)) * (gamma1 * gamma2 * h_bar / (r_dist**3))
	pass

def OP_time_evol(H, Dt):
	# Use eigh.
	# The Hamiltonian is (should really be) hermitian.
	(vals, vecs) = eigh(H)
	# print vals
	# no need for h_bar, as h_bar cancels out with the
	# implicit h_bar in the Hamiltonian
	vals = exp(-1j * Dt * vals)
	return recombine_eig(vals, vecs)

def apply_liouville(dens, H, Dt):
	U = OP_time_evol(H, Dt)
	r = dot(dot(U, dens), U.conj().T)
	return r

if True:
	# import from physconsts
	from numpy import pi
	kB = 1.3806505e-23
	h = 6.6260693e-34
	h_bar = h / (2*pi)
	# Ch. Kittel
	mu0 = 4*pi*1e-7


def prepare_equil(H, T):
	from numpy import expm1
	(vals, vecs) = eig(H)
	n = len(vals)
	# print vals
	# print vecs
	boltzmann = - vals * h_bar / (kB * T)
	# print boltzmann
	if True:
		redprobs = expm1(boltzmann)
		sumred = redprobs.sum()
		probs = (n*redprobs - sumred) / (n * (n + sumred))
		# print probs
	else:
		probs = exp(boltzmann)
		probs = probs / probs.sum()
		print probs
		probs -= 1./n
		print probs
	if True:
		dens = recombine_eig(probs, vecs)
	else:
		dens = numpy.zeros((n, n), dtype = "complex")
		for i in xrange(n):
			dens += probs[i] * outer(vecs[i], vecs[i])
	return dens


def _pauli_diag(N):
	k = numpy.arange(1, N+1)
	sq = k * (N + 1 - k)
	sq = numpy.sqrt(sq)
	return sq

def _pauli_x_gen(N):
	from numpy import diag
	s = _pauli_diag(N)
	a = diag(s, 1) + diag(s, -1)
	return 0.5 * a

def _pauli_y_gen(N):
	from numpy import diag
	s = _pauli_diag(N)
	a = 1j * (diag(s, -1) - diag(s, 1))
	return 0.5 * a

def Iz(n):
	return numpy.diag(numpy.arange((n-1.)/2, (-n-1.)/2, -1))
def Ix(n):
	return _pauli_x_gen(n-1)
def Iy(n):
	return _pauli_y_gen(n-1)


I2z = 0.5  * array([[1,  0], [0, -1]])
I2x = 0.5  * array([[0,  1], [1,  0]])
I2y = 0.5j * array([[0, -1], [1,  0]])

I3z =              array([[1,  0, 0], [0, 0,  0], [0, 0, -1]])
I3x = 1 /sqrt(2) * array([[0,  1, 0], [1, 0,  1], [0, 1,  0]])
I3y = 1j/sqrt(2) * array([[0, -1, 0], [1, 0, -1], [0, 1,  0]])

q3 =     sqrt(3)
q4 = 2 # sqrt(4)
I4z = 0.5 * array([[3,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,-3]])
I4x = 0.5 * array([[0,q3,0,0], [q3,0,q4,0], [0,q4,0,q3], [0,0,q3,0]])
I4y = -1j * commutator(I4z, I4x)

# I4x = Ix(4)
# I4y = Iy(4)


def SpinSet(x,y,z):
	return (x,y,z)

I2set = SpinSet(I2x, I2y, I2z)
I3set = SpinSet(I3x, I3y, I3z)
I4set = SpinSet(I4x, I4y, I4z)

def In_set(n):
	return SpinSet( Ix(n), Iy(n), Iz(n))


def spinset_tensorproduct(list_of_sets):
	eye_list = numpy.empty((len(list_of_sets), ), dtype="int")
	for i in xrange(len(list_of_sets)):
		eye_list[i] = list_of_sets[i][0].shape[0]
	left_eye_val = 1
	right_eye_val = eye_list.prod()
	result = []
	for i in xrange(len(list_of_sets)):
		q = list_of_sets[i]
		right_eye_val //= eye_list[i]

		left_eye = eye(left_eye_val)
		right_eye = eye(right_eye_val)
		x = tensorproduct(tensorproduct(left_eye, q[0]), right_eye)
		y = tensorproduct(tensorproduct(left_eye, q[1]), right_eye)
		z = tensorproduct(tensorproduct(left_eye, q[2]), right_eye)
		result.append(SpinSet(x,y,z))

		left_eye_val *= eye_list[i]
	#for q in result:
	#	print "VVVVVVVVVVVVV"
	#	for u in q:
	#		print u
	#	print "^^^^^^^^^^^^^"
	return result


class TimeEvol(object):
	"""Currently completely unoptimized..."""
	def __init__(self, dens, H, O_meas):
		self.startdens = dens
		self.H = H
		self.O_meas = O_meas
	def getval(self, Dt):
		r = apply_liouville(self.startdens, self.H, Dt)
		return dot(r, self.O_meas).trace()
	def calc_for_range(self, val_array):
		f = numpy.vectorize(self.getval)
		res = f(val_array)
		return res

import unittest

class SelfTest(unittest.TestCase):
	def assertclose(self, x, y):
		assert allclose(x, y)
	def test_Iz(self):
		self.assertclose(Iz(2), I2z)
		self.assertclose(Iz(3), I3z)
		self.assertclose(Iz(4), I4z)
	def test_Ix(self):
		self.assertclose(Ix(2), I2x)
		self.assertclose(Ix(3), I3x)
		self.assertclose(Ix(4), I4x)
	def test_Iy(self):
		self.assertclose(Iy(2), I2y)
		self.assertclose(Iy(3), I3y)
		self.assertclose(Iy(4), I4y)
	def test_spin2_len(self):
		I2 = dot(I2x, I2x) + dot(I2y, I2y) + dot(I2z, I2z)
		val = 1/2 * (1/2 + 1)
		self.assertclose(I2.trace() * 1/2 , val)
		self.assertclose(I2, val * eye(2))
	def test_spin3_len(self):
		I2 = dot(I3x, I3x) + dot(I3y, I3y) + dot(I3z, I3z)
		val = 1 * (1 + 1)
		self.assertclose(I2.trace() * 1/3 , val)
		self.assertclose(I2, val * eye(3))
	def test_spin4_len(self):
		I2 = dot(I4x, I4x) + dot(I4y, I4y) + dot(I4z, I4z)
		val = 3/2 * (3/2 + 1)
		self.assertclose(I2.trace() * 1/4 , val)
		self.assertclose(I2, val * eye(4))
	def test_spin_len_generic(self):
		for n in xrange(2, 11):
			x = Ix(n)
			y = Iy(n)
			z = Iz(n)
			I2 = dot(x,x) + dot(y,y) + dot(z,z)
			spinval = (n-1)/2.
			val = spinval * (spinval + 1)
			self.assertclose(I2, val * eye(n))
	def test_recombine_eig(self):
		(vals, vecs) = eig(I2y)
		self.assertclose(vals, array([ 0.5+0.j, -0.5+0.j]))
		recons = recombine_eig(vals, vecs)
		assert allclose(I2y, recons, atol=1e-15)
	def test_tensor_product(self):
		a = array([[1,2], [4,8]])
		b = array([[1,3], [9,27]])
		r1 = tensorproduct(a, b)
		expected = array( \
		  [[  1,  3,  2,  6], \
		   [  9, 27, 18, 54], \
		   [  4, 12,  8, 24], \
		   [ 36,108, 72,216]])
		self.assertclose(r1, expected)
		r2 = my_tensorproduct(a, b)
		self.assertclose(r1, r2)
	def test_spinset_tensorproduct(self):
		def compare_set(s1, s2):
			for i in xrange(3):
				self.assertclose(s1[i], s2[i])
		s_x = tensorproduct(tensorproduct(I3x, eye(2)), eye(4))
		s_y = tensorproduct(tensorproduct(I3y, eye(2)), eye(4))
		s_z = tensorproduct(tensorproduct(I3z, eye(2)), eye(4))
		set1 = SpinSet(s_x, s_y, s_z)
		s_x = tensorproduct(tensorproduct(eye(3), I2x), eye(4))
		s_y = tensorproduct(tensorproduct(eye(3), I2y), eye(4))
		s_z = tensorproduct(tensorproduct(eye(3), I2z), eye(4))
		set2 = SpinSet(s_x, s_y, s_z)
		s_x = tensorproduct(tensorproduct(eye(3), eye(2)), I4x)
		s_y = tensorproduct(tensorproduct(eye(3), eye(2)), I4y)
		s_z = tensorproduct(tensorproduct(eye(3), eye(2)), I4z)
		set3 = SpinSet(s_x, s_y, s_z)

		(t_1, t_2, t_3) = spinset_tensorproduct( ( \
			I3set, \
			I2set, \
			I4set, \
			))
		compare_set(set1, t_1)
		compare_set(set2, t_2)
		compare_set(set3, t_3)
	def test_calc_w_dd(self):
		from util import gamma_H
		a = calc_w_dd(gamma_H, gamma_H, 1e-10)/(2*pi)
		# According to Spiess (XXX)
		# page 19, Eq 2.15
		# but I'm 1.7 % off
		assert allclose(a, 122000, rtol = 1.7e-2)
		# My value:
		assert allclose(a, 120120)
	def test_liouville(self):
		# See Schmidt-Rohr / Spiess, Chapter 2.6.2
		#
		r = apply_liouville(I2z, - I2x, pi/2)
		assert allclose(r, I2y)
		r = apply_liouville(I4z, - I4x, pi/2)
		assert allclose(r, I4y)
		r = apply_liouville(I3z, I3x, pi)
		assert allclose(r, -I3z)
		r = apply_liouville(I3z, I3x, 2*pi)
		assert allclose(r, I3z)
	def test_TimeEvol_simple(self):
		"""Really basic test of the TimeEvol class"""
		te = TimeEvol(I4z, - I4x, I4y)
		v_should = dot(I4y, I4y).trace()
		from numpy import sin
		for Dt in numpy.linspace(0, pi, 178):
			v = te.getval(Dt)
			assert allclose(v, v_should * sin(Dt))
		inp = numpy.linspace(0, pi, 2*3*5*7)
		inp.resize((2*5, 3*7))
		exp_outp = v_should * sin(inp)
		outp = te.calc_for_range(inp)
		assert allclose(outp, exp_outp)
	def test_dipole_dipole_rotation(self):
		# Ernst, NMR in one and two dimensions, page 47
		from numpy import sin, cos, real_if_close
		(base_S1, base_S2) = spinset_tensorproduct( (I2set, I4set) )
		a_1x, a_1y, a_1z = base_S1
		a_2x, a_2y, a_2z = base_S2
		a_1plus  = a_1x + 1j*a_1y
		a_1minus = a_1x - 1j*a_1y
		a_2plus  = a_2x + 1j*a_2y
		a_2minus = a_2x - 1j*a_2y
		for theta in (0, pi/4, pi/3, pi/2, pi, 1):
		  for phi in (0, pi/4, pi/3, pi/2, pi, 1, 2*pi):
		    a_H_m2 = -3./4 * dot(a_1minus, a_2minus) * sin(theta)**2 \
		    	* exp(2j * phi)
		    a_H_p2 = -3./4 * dot(a_1plus, a_2plus) * sin(theta)**2 \
		    	* exp(-2j * phi)
		    a_H_m1 = -3./2 * \
		    	( dot(a_1z, a_2minus) + dot(a_1minus, a_2z) ) \
		    	* sin(theta) * cos(theta) * exp(1j * phi) 
		    a_H_p1 = -3./2 * \
		    	( dot(a_1z, a_2plus) + dot(a_1plus, a_2z) ) \
		    	* sin(theta) * cos(theta) * exp(-1j * phi) 
		    a_H_0 = (dot(a_1z, a_2z) - \
		    	1./4 * (dot(a_1plus, a_2minus) + dot(a_1minus, a_2plus))) \
		    	* (1 - 3*(cos(theta)**2))
		    a_H = a_H_m2 + a_H_p2 + a_H_m1 + a_H_p1 + a_H_0

		    b_S1 = rotate_spin_set(base_S1, -phi, -theta)
		    b_S2 = rotate_spin_set(base_S2, -phi, -theta)
		    b_H = H_dipoledipole(b_S1, b_S2, 1)
		    if not allclose(a_H, b_H):
		      print "*** ARRRRGGGGG", theta*180./pi, phi*180/pi
		      print real_if_close(a_H)
		      print real_if_close(b_H)
		      print "-----"
		      print real_if_close(a_H - b_H)
		      print allclose(a_H, b_H)
		      assert False
	def test_quad_1(self):
		w_Q = 1.2e6
		eta = 0.6
		Iset = I3set
		I = 1
		tset = array(H_quadrupole_tensor_set(Iset, I, w_Q))
		tset[0] *= eta/sqrt(6)
		tset[4] *= eta/sqrt(6)
		tset[1] *= 0
		tset[3] *= 0
		tset[2] *= 1
		H_q1 = tset.sum(axis=0)
		H_q2 = H_quadrupole(Iset, I, w_Q, eta)
		assert allclose(H_q1, H_q2)

if __name__ == '__main__':
	# Things needed for unittest:
	from numpy import pi, allclose

	unittest.main()
