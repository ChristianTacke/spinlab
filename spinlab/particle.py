from __future__ import division

from itertools import ifilter
from numpy import array, dot, sqrt, sin, cos, exp
from spinlab import qmlib
from spinlab.qmlib import In_set, spinset_tensorproduct, \
	rotate_spin_set, calc_w_dd
import numpy

class Particle(object):
	def __init__(self, spin_num, name = None):
		self.num_states = int(spin_num * 2 + 1)
		self.Isquared = spin_num * (spin_num + 1)
		self.nontens_spinset = In_set(self.num_states)
		self.gamma = None
		self.name = name
	def __repr__(self):
		return "<Spin %d/2>" % (self.num_states - 1)
	spinset = property(doc = "Don't use this!")
	def part_of_system(self):
		return hasattr(self, "in_system")
	def set_system(self, system):
		assert not self.part_of_system()
		self.in_system = system
	def set_base_spinset(self, spinset):
		"""set the operators for the whole hilbert space"""
		self.base_spinset = spinset
	def set_coords_A(self, coords):
		"""Set coordinates (in Angstrom)"""
		self.coords_A = array(coords)
	def set_gamma(self, gamma):
		self.gamma = gamma
	def set_quadrupole_params(self, w_Q, eta, theta, phi):
		self.quad_w_Q = w_Q
		self.quad_eta = eta
		self.quad_theta = theta
		self.quad_phi = phi
	def set_quadrupole_angles_tp(self, other_particle):
		"""Let the quadrupole base axis point to other_particle"""
		assert self.part_of_system()
		assert self.in_system == other_particle.in_system
		system = self.in_system
		theta, phi = system.tp_angles(self, other_particle)
		self.quad_theta = theta
		self.quad_phi = phi
		# XXX - Do we need to reset the system? Might be.

class SpinSystem(object):
	def __init__(self):
		self.particle_list = []
		self.particles = {}
		self.cached_internal = {}
		self.set_cryst_angles(0.0, 0.0)
	def pprint(self, x):
		import pprint
		pprint.pprint(x)
	def append(self, particle):
		particle.set_system(self)
		self.particle_list.append(particle)
		self.particles[particle.name] = particle
	def remove(self, particle):
		self.particle_list.remove(particle)
		del self.particles[particle.name]
		del particle.in_system
	def set_cryst_angles(self, theta, phi):
		self.cryst_angles = array((theta, phi))
		self.cached_internal.clear()
	def write_to_cml(self, f):
		"""Export structure to .cml file
		give either a file object or a file name"""
		import re
		if not hasattr(f, "write"):
			f = file(f, "w")
		f.write("<molecule>\n<atomArray>\n")
		for q in self.particle_list:
			f.write('  <atom')
			if q.name is not None:
				f.write(' title="%s"' % q.name)
				m = re.match(r"^[A-Z][a-z]?", q.name)
				if m is not None:
					f.write(' elementType="%s"' % \
						m.group())
			f.write(' x3="%f" y3="%f" z3="%f"/>\n' % \
				tuple(q.coords_A))
		f.write("</atomArray>\n</molecule>\n")
	def create_base_spinsets(self):
		self.particle_list = tuple(self.particle_list)
		list_spinset1 = map(lambda x: x.nontens_spinset, \
			self.particle_list)
		list_spinset1 = tuple(list_spinset1)
		if False:
			self.pprint("Original spinsets:")
			for q in list_spinset1:
				self.pprint(q)
		list_spinset2 = spinset_tensorproduct(list_spinset1)
		map(lambda p, s: p.set_base_spinset(s), \
			self.particle_list, \
			list_spinset2)
		if False:
			self.pprint("Old,new,intersleaved:")
			for q in self.particle_list:
				self.pprint(q.nontens_spinset)
				self.pprint(q.spinset)

	def state_names(self):
		l = []
		for q in self.particle_list:
			l.append(numpy.diagonal(q.spinset[2]))
		l = array(l).T
		return l

	def get_sp_rot_spinset(self, a):
		i_S1 = a.base_spinset
		theta, phi = - self.cryst_angles
		return rotate_spin_set(i_S1, 0, theta, phi)
	def tp_distance_A(self, a, b):
		"""Distance of Two Particles in Angstrom"""
		assert a.in_system == self
		assert b.in_system == self
		d = b.coords_A - a.coords_A
		return sqrt(dot(d, d))
	def tp_angles(self, a, b):
		"""spherical angles of the connection from a to b
		returns a tuple/array (theta, phi) where
		theta is 0..pi and is the angle to the positive z axis
		phi is -pi..pi and is the angle in the xy-plance to the positive x axis"""
		d = b.coords_A - a.coords_A
		l = sqrt(dot(d, d))
		theta = numpy.arccos(d[2] / l)
		phi = numpy.arctan2(d[1], d[0])
		return array((theta, phi))
	def tp_iter(self):
		"""Iterate over all pairs in the system"""
		max_num = len(self.particle_list)
		for i in xrange(max_num):
			a1 = self.particle_list[i]
			for j in xrange(i+1, max_num):
				a2 = self.particle_list[j]
				yield frozenset((a1, a2))
	def quadnuclei_iter(self):
		"""Iterate over all quadrupole nuclei"""
		return ifilter(lambda p: hasattr(p, "quad_w_Q"), \
			       self.particle_list)
	def tp_get_spinsets(self, a, b):
		try:
			return self.cached_internal[a,b]
		except KeyError:
			pass
		base_S1 = self.get_sp_rot_spinset(a)
		base_S2 = self.get_sp_rot_spinset(b)
		# We need to rotate in the opposite direction
		# That's the "-".
		theta, phi = - self.tp_angles(a, b)
		S1 = rotate_spin_set(base_S1, phi, theta)
		S2 = rotate_spin_set(base_S2, phi, theta)
		self.cached_internal[a,b] = (S1, S2)
		return (S1, S2)
	def sp_get_quad_spinset(self, a):
		base_S1 = self.get_sp_rot_spinset(a)
		return rotate_spin_set(base_S1, - a.quad_phi, - a.quad_theta)

	def H_Zeeman_1T(self, a, axis=2):
		# Don't use the rotated one here.
		# Feels a little inconsistent.
		return 1. * a.gamma * a.base_spinset[axis]
	def H_Zeeman_1T_all_1(self, axis=2):
		"""Zeeeman Hamiltonian for all, variant 1"""
		ret_H = None
		for p in self.particle_list:
			if ret_H is None:
				ret_H = self.H_Zeeman_1T(p, axis)
			else:
				ret_H += self.H_Zeeman_1T(p, axis)
		return ret_H
	def H_Zeeman_1T_all_2(self, axis=2):
		"""Zeeeman Hamiltonian for all, variant 2"""
		l = map(lambda p: self.H_Zeeman_1T(p, axis),
			self.particle_list)
		return numpy.sum(l, axis=0)
	H_Zeeman_1T_all = H_Zeeman_1T_all_2

	def w_dipoledipole(self, a, b):
		return calc_w_dd(a.gamma, b.gamma,
			self.tp_distance_A(a, b) * 1e-10)
	def H_dipoledipole(self, a, b):
		w_dd = calc_w_dd(a.gamma, b.gamma,
			self.tp_distance_A(a, b) * 1e-10)
		s1, s2 = self.tp_get_spinsets(a, b)
		H_dip = qmlib.H_dipoledipole(s1, s2, w_dd)
		return H_dip
	def H_dipoledipole_alphabet(self, a, b):
		w_dd = calc_w_dd(a.gamma, b.gamma,
			self.tp_distance_A(a, b) * 1e-10)

		base_S1 = self.get_sp_rot_spinset(a)
		base_S2 = self.get_sp_rot_spinset(b)

		a_1x, a_1y, a_1z = base_S1
		a_2x, a_2y, a_2z = base_S2
		a_1plus  = a_1x + 1j*a_1y
		a_1minus = a_1x - 1j*a_1y
		a_2plus  = a_2x + 1j*a_2y
		a_2minus = a_2x - 1j*a_2y
		theta, phi = self.tp_angles(a, b)

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
		return w_dd * a_H
	def H_dipoledipole_all(self):
		max_num = len(self.particle_list)
		l = []
		for i in xrange(max_num):
			a1 = self.particle_list[i]
			for j in xrange(i+1, max_num):
				a2 = self.particle_list[j]
				l.append(self.H_dipoledipole(a1, a2))
		return numpy.sum(l, axis = 0)

	def H_quadrupole(self, a):
		I = (a.num_states-1) / 2.
		S1 = self.sp_get_quad_spinset(a)
		return qmlib.H_quadrupole(S1, I, a.quad_w_Q, a.quad_eta)
	def H_quadrupole_all(self):
		l = filter(lambda p: hasattr(p, "quad_w_Q"), self.particle_list)
		l = map(lambda p: self.H_quadrupole(p), l)
		return numpy.sum(l, axis = 0)

	def O_measure_all(self, gamma, disp_cb = None):
		"""Zeeeman Hamiltonian for all, variant 1"""
		ret_O = None
		for p in self.particle_list:
			if p.gamma != gamma:
				continue
			if disp_cb is not None:
				disp_cb(" " + p.name)
			if ret_O is None:
				ret_O = p.base_spinset[2].copy()
			else:
				ret_O += p.base_spinset[2]
		return ret_O

import unittest

class SelfTest(unittest.TestCase):
	def assertclose(self, x, y):
		assert numpy.allclose(x, y)
	def assertnotclose(self, x, y):
		assert not numpy.allclose(x, y)
	def setUp(self):
		from util import gamma_H, gamma_N
		from numpy import arctan

		self.S1_a = Particle(1/2)
		self.S1_a.set_coords_A((0, 0, 0))
		self.S1_b = Particle(1)
		self.S1_b.set_coords_A((0, 0, 13))
		self.S1_a.set_gamma(gamma_H)
		self.S1_b.set_gamma(gamma_N)
		self.S1_b.set_quadrupole_params(1e6, 0.1, 0, 0)
		self.S1 = SpinSystem()
		self.S1.append(self.S1_a)
		self.S1.append(self.S1_b)
		self.S1.create_base_spinsets()

		self.S2_a = Particle(1/2)
		self.S2_a.set_coords_A((0, 0, 0))
		self.S2_b = Particle(1)
		self.S2_b.set_coords_A((4, 3, 12))
		self.S2_a.set_gamma(gamma_H)
		self.S2_b.set_gamma(gamma_N)
		self.S2_b.set_quadrupole_params(1e6, 0.1, arctan(5/12.), arctan(3/4.))
		self.S2 = SpinSystem()
		self.S2.append(self.S2_a)
		self.S2.append(self.S2_b)
		self.S2.create_base_spinsets()
	def test_dd_equal(self):
		H1 = self.S2.H_dipoledipole(self.S2_a, self.S2_b)
		H2 = self.S2.H_dipoledipole_alphabet(self.S2_a, self.S2_b)
		self.assertclose(H1, H2)
	def test_cryst_uneq(self):
		H1_dd = self.S1.H_dipoledipole_all()
		H2_dd = self.S2.H_dipoledipole_all()
		self.assertnotclose(H1_dd, H2_dd)
	def test_cryst_rot_wrong_dd(self):
		from numpy import arctan
		self.S1.set_cryst_angles(arctan(5/12.), arctan(3/4.))
		H1_dd = self.S1.H_dipoledipole_all()
		H2_dd = self.S2.H_dipoledipole_all()
		self.assertnotclose(H1_dd, H2_dd)
	def test_cryst_rot_dd(self):
		from numpy import arctan
		self.S2.set_cryst_angles(-arctan(5/12.), -arctan(3/4.))
		H1_dd = self.S1.H_dipoledipole_all()
		H2_dd = self.S2.H_dipoledipole_all()
		self.assertclose(H1_dd, H2_dd)
		for sign1, sign2 in ((1, 1), (1, -1), (-1, 1)):
			self.S2.set_cryst_angles(sign1*arctan(5/12.),
						 sign2*arctan(3/4.))
			H2_dd = self.S2.H_dipoledipole_all()
			self.assertnotclose(H1_dd, H2_dd)
	def test_cryst_rot_wrong_quad(self):
		from numpy import arctan
		self.S1.set_cryst_angles(arctan(5/12.), arctan(3/4.))
		H1_q = self.S1.H_quadrupole_all()
		H2_q = self.S2.H_quadrupole_all()
		self.assertnotclose(H1_q, H2_q)
	def test_cryst_rot_quad(self):
		from numpy import arctan
		self.S2.set_cryst_angles(-arctan(5/12.), -arctan(3/4.))
		H1_q = self.S1.H_quadrupole_all()
		H2_q = self.S2.H_quadrupole_all()
		self.assertclose(H1_q, H2_q)
		for sign1, sign2 in ((1, 1), (1, -1), (-1, 1)):
			self.S2.set_cryst_angles(sign1*arctan(5/12.),
						 sign2*arctan(3/4.))
			H2_q = self.S2.H_quadrupole_all()
			self.assertnotclose(H1_q, H2_q)
	def test_quad_angles_tp(self):
		from numpy import pi, abs
		H0_q = self.S2.H_quadrupole_all()
		theta1, phi1 = self.S2_b.quad_theta, self.S2_b.quad_phi
		self.S2_b.set_quadrupole_angles_tp(self.S2_a)
		theta2, phi2 = self.S2_b.quad_theta, self.S2_b.quad_phi
		self.assertclose(theta1+theta2, pi)
		self.assertclose(abs(phi2 - phi1), pi)
		H2_q = self.S2.H_quadrupole_all()
		self.assertclose(H0_q, H2_q)
	def test_basic_quad(self):
		self.S1.H_quadrupole_all()
		self.S2.H_quadrupole_all()
	def test_basic_zeeman(self):
		H1_1 = self.S1.H_Zeeman_1T_all_1()
		H1_2 = self.S1.H_Zeeman_1T_all_2()
		H1_P = self.S1.H_Zeeman_1T_all()
		H2_1 = self.S2.H_Zeeman_1T_all_1()
		H2_2 = self.S2.H_Zeeman_1T_all_2()
		H2_P = self.S2.H_Zeeman_1T_all()
		for q in (H1_2, H1_P, H2_1, H2_2, H2_P):
			self.assertclose(H1_1, q)
	def test_tp_iter(self):
		S = SpinSystem()
		a = Particle(1/2)
		b = Particle(2/2)
		c = Particle(3/2)
		for q in (a,b,c):
			S.append(q)
		l = list(S.tp_iter())
		assert len(l) == 3
		assert frozenset((a,b)) in l
		assert frozenset((a,c)) in l
		assert frozenset((b,c)) in l
	def test_quadnuclei_iter(self):
		S = SpinSystem()
		a = Particle(1/2)
		b = Particle(2/2)
		b.set_quadrupole_params(1e6, 0.1, 0, 0)
		c = Particle(3/2)
		c.set_quadrupole_params(1e6, 0.1, 0, 0)
		for q in (a,b,c):
			S.append(q)
		l = list(S.quadnuclei_iter())
		assert len(l) == 2
		assert b in l
		assert c in l


def try_something():
	a = Particle(1/2)
	a.set_coords_A((0, 0, 0))
	b = Particle(1/2)
	b.set_coords_A((-1, 0, 0))

	from util import gamma_H
	a.set_gamma(gamma_H)
	b.set_gamma(gamma_H)

	S = SpinSystem()
	S.append(a)
	S.append(b)
	# S.append(a)
	S.create_base_spinsets()
	(S1, S2) = S.tp_get_internal_spinsets(a, b)
	from pprint import pprint
	pprint(S1)
	pprint(S2)

	print "Distance:", S.tp_distance_A(a,b)
	print "Angles:", S.tp_angles(a,b) * 180. / numpy.pi

	H1 = S.H_dipoledipole(a, b)
	H2 = S.H_dipoledipole_alphabet(a, b)

	w_dd = qmlib.calc_w_dd(a.gamma, b.gamma, 1e-10)
	print "w_dd:", w_dd

	H1 /= w_dd
	H2 /= w_dd

	print(numpy.array2string(numpy.real_if_close(H1), suppress_small = True))
	print(numpy.array2string(numpy.real_if_close(H2), suppress_small = True))
	print "-" * 78

	v1 = 1/2.*array(( 1,  1,  1,  1))
	v2 = 1/2.*array(( 1, -1,  1, -1))
	v3 = 1/2.*array(( 1,  1, -1, -1))
	v4 = 1/2.*array(( 1, -1, -1,  1))

	for i in (v1, v2, v3, v4):
	   print dot(dot(i, a.spinset[0]), i), \
	      dot(dot(i, a.spinset[1]), i), \
	      dot(dot(i, a.spinset[2]), i)
	for i in (v1, v2, v3, v4):
	   print dot(dot(i, b.spinset[0]), i), \
	      dot(dot(i, b.spinset[1]), i), \
	      dot(dot(i, b.spinset[2]), i)

	for i in (v1, v2, v3, v4):
	  for j in (v1, v2, v3, v4):
	    print dot(dot(i, H2), j),
	  print

if __name__ == '__main__':
	unittest.main()
