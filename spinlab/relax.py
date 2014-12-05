from __future__ import division
from numpy import array, abs, newaxis, real_if_close, diag_indices_from, \
	exp, log10, logspace, empty, diag, \
	allclose, savetxt
from pprint import pprint
from spinlab.qmlib import dot, eigh, recombine_eig
from spinlab.qmlib import H_dipoledipole_tensor_set, H_quadrupole_tensor_set

def kimmich_t1(I, w_dd, omega, tau):
	# See Kimmich Chapter 12.
	J1 = 2 * tau / (1 + (  omega*tau)**2)
	J2 = 2 * tau / (1 + (2*omega*tau)**2) 

	R = 1./5 * w_dd**2 * I * (I+1) * (J1 + 4*J2)
	return 1/R


class Redfield_simple(object):
	def pprint(self, a):
		pprint(a)
	def info(self, s):
		print s
	def __init__(self, S, H_0, verbose=True):
		self.points = 23
		self.Rab_list = []

		if verbose == False:
			self.info = self.pprint = lambda v: v
		self.verbose = verbose
		self.dump_reldata = None
		self.S = S

		H0_vals, H0_vecs = eigh(H_0)
		self.info("H_0 eigenvectors:")
		self.pprint(H0_vecs)
		self.H0_vecs = H0_vecs

		H_0_ = dot(dot(H0_vecs.conj().T, H_0), H0_vecs)
		# pprint(H_0_)
		# pprint(H0_vals)
		assert allclose(diag(H_0_), H0_vals)

		self.w_diff = abs(H0_vals[newaxis, :] - H0_vals[:, newaxis])

	def set_dump_reldata(self, dump_reldata):
		self.dump_reldata = dump_reldata

	def add_dd_relaxation(self, a, b, J):
		i1set = a.base_spinset
		i2set = b.base_spinset
		w_dd = self.S.w_dipoledipole(a, b)

		J0 = lambda w: 12./15 * J(w)
		J1 = lambda w:  2./15 * J(w)
		J2 = lambda w:  8./15 * J(w)

		f_set = H_dipoledipole_tensor_set(i1set, i2set, w_dd)

		self._add_relaxation(f_set, J0, J1, J2)

	def add_quad_relaxation(self, a, J):
		i1set = a.base_spinset
		w_Q = a.quad_w_Q
		eta = a.quad_eta
		I = (a.num_states-1) / 2.

		eta_fac = 1. + eta*eta/3.
		J0 = lambda w: eta_fac * 12./15 * 1/4. * J(w)
		J1 = lambda w: eta_fac *  2./15 * 3/2. * J(w)
		J2 = lambda w: eta_fac *  8./15 * 3/8. * J(w)

		f_set = H_quadrupole_tensor_set(i1set, I, w_Q)

		self._add_relaxation(f_set, J0, J1, J2)

	def _add_relaxation(self, f_set, J0, J1, J2):
		H0_vecs = self.H0_vecs # cache locally

		J0ab = J0(self.w_diff)
		J1ab = J1(self.w_diff)
		J2ab = J2(self.w_diff)
		# pprint(J1ab)

		f2 = []
		for A, Jq in zip(f_set, (J2ab, J1ab, J0ab, J1ab, J2ab)):
			A = dot(dot(H0_vecs.conj().T, A), H0_vecs)
			A *= A.conj()
			A *= Jq
			A = real_if_close(A)
			f2.append(A)

		f2 = array(f2)
		# pprint(f2)

		Rab = f2.sum(axis=0)
		diag_idx = diag_indices_from(Rab)
		Rab[diag_idx] = 0
		assert allclose(Rab, Rab.T)

		Rab[diag_idx] = -Rab.sum(axis=1)
		self.info("Redfield matrix:")
		self.pprint(Rab)
		self.Rab_list.append(Rab)

	def final_Rab(self):
		Rab = array(self.Rab_list)
		self.Rab = Rab.sum(axis=0)

		self.Rab_vals, self.Rab_vecs = eigh(self.Rab)
		self.info("Rab_vals, vecs:")
		self.pprint(self.Rab_vals)
		self.pprint(self.Rab_vecs)

	def _setup_pop_meas_ana(self, start_dens, end_dens, O_meas):
		H0_vecs = self.H0_vecs # cache locally

		self.info("O_measure:")
		self.pprint(O_meas)
		O_meas = dot(dot(H0_vecs.conj().T, O_meas), H0_vecs)
		self.pprint(O_meas)
		self.info("Start_dens:")
		self.pprint(start_dens)
		start_dens = dot(dot(H0_vecs.conj().T, start_dens), H0_vecs)
		self.pprint(start_dens)
		self.info("End_dens ({0}):".format(end_dens.dtype))
		self.pprint(end_dens)
		end_dens = dot(dot(H0_vecs.conj().T, end_dens), H0_vecs)
		self.pprint(end_dens)

		self._pop_base = diag(end_dens)
		self._pop_diff = diag(start_dens) - self._pop_base
		self._pop_meas = diag(O_meas)

		m_start_d = dot(start_dens, O_meas).trace()
		m_end_d = dot(end_dens, O_meas).trace()
		# m_start_p = dot(pop_diff + pop_base, pop_meas)
		mod_start_dens = start_dens.copy()
		mod_start_dens[diag_indices_from(mod_start_dens)] = 0
		m_diff = dot(mod_start_dens, O_meas).trace()
		# m_err = (m_start_p - m_start_d) / (m_start_d - m_end_d)
		m_err = m_diff / (m_start_d - m_end_d)
		# print "(%r,\n %r,\n %r)" % (m_diff, m_start_d, m_end_d)
		# print "(%r)" % (m_err,)

	def population_measure_analyze(self, start_dens, end_dens, O_meas):
		from fitmodels import guess_kohlrausch

		R_vals, R_vecs = self.Rab_vals, self.Rab_vecs

		self._setup_pop_meas_ana(start_dens, end_dens, O_meas)
		pop_base = self._pop_base
		pop_diff = self._pop_diff
		pop_meas = self._pop_meas

		if self.dump_reldata is not None:
			self.dump_reldata(R_vals)

		T1guess = 1 / abs(R_vals).max()
		self.info("T1guess: %r" % T1guess)
		self.T1guess = None
		for points, log_left, log_right in \
				((7, 1, 5), (self.points, 2.5, 2.5)):
			print "\tT1guess: %r" % T1guess
			target = empty((points, 3))
			target[:, 0] = logspace(log10(T1guess) - log_left,
				log10(T1guess) + log_right, points)
			for i in xrange(points):
				Rt = recombine_eig(exp(R_vals * target[i, 0]), R_vecs)
				pop_final = dot(Rt, pop_diff)
				m = dot(pop_final + pop_base, pop_meas)
				target[i, 1] = m
			self.pprint(target[:, 0:2])
			p0 = guess_kohlrausch(target[:, 0], target[:, 1])
			T1guess = 1 / p0[1]
			if self.T1guess is None:
				self.T1guess = T1guess
		self.target = target

	def fit_T1(self, save_file=None):
		from fit import fit
		from fitmodels import func_kohlrausch, guess_kohlrausch	

		p0 = guess_kohlrausch(self.target[:, 0], self.target[:, 1])
		self.pprint(p0)
		fit_result = fit(func_kohlrausch, p0, self.target[:, 0], self.target[:, 1])
		self.pprint(fit_result)
		p0 = fit_result[0]
		self.info("Final t1: %r" % (1 / p0[1]))
		t1_misguess = log10(self.T1guess * p0[1])
		if abs(t1_misguess) > 0.74:
			print "*** WARNING: T1guess is off by 10**%r." % \
				t1_misguess

		if save_file is not None:
			self.target[:, 2] = self.target[:, 1] - \
				func_kohlrausch(p0, self.target[:, 0])
			savetxt(save_file, self.target)

		self.fit_result = fit_result
		return 1 / p0[1]

import unittest

class SelfTest(unittest.TestCase):
	def test_identical_spins(self):
		from spinlab.spingamma import get_gamma
		from spinlab.particle import Particle, SpinSystem
		from spinlab.qmlib import calc_w_dd, prepare_equil
		from spinlab.specdens import J_Lorentzian

		gamma = get_gamma("1H")  # Just some gamma, doesn't matter
		B0 = 11.74  # T, Just a B0

		tau = 0.2e-9  # sec
		def J(w):
			return J_Lorentzian(w, tau)

		for spin_num in (1/2, 1, 3/2, 7/2):
			a = Particle(spin_num)
			a.set_coords_A((0, 0, 0))
			a.set_gamma(gamma)
			b = Particle(spin_num)
			b.set_coords_A((-2, 0, 0))
			b.set_gamma(gamma)

			S = SpinSystem()
			S.append(a)
			S.append(b)
			S.create_base_spinsets()

			w_dd = calc_w_dd(gamma, gamma, 2e-10)
			# print w_dd

			H_0 = - B0 * S.H_Zeeman_1T_all()
			# print "H_0 (Zeeman):"
			# pprint(H_0)

			R = Redfield_simple(S, H_0, verbose=False)
			R.add_dd_relaxation(a, b, J)
			R.final_Rab()
			start_dens = prepare_equil(0.1 * H_0, 300)
			end_dens = prepare_equil(H_0, 300)
			O_meas = S.O_measure_all(gamma)
			R.population_measure_analyze(start_dens, end_dens, O_meas)
			T1_my = R.fit_T1()
			w = B0 * gamma
			T1_kim = kimmich_t1(spin_num, w_dd, w, tau)
			# print "Fitted t1:", T1_my
			# print "Kimmich  :", T1_kim
			# print "Rel Diff :", (T1_my - T1_kim) / T1_kim
			assert allclose(T1_my, T1_kim, rtol=1.23e-4)

	def test_quad_1(self):
		from spinlab.spingamma import get_gamma
		from spinlab.particle import Particle, SpinSystem
		from spinlab.qmlib import prepare_equil
		from spinlab.specdens import J_Lorentzian

		gamma = get_gamma("N")  # Just some gamma, doesn't matter
		B0 = 6.  # T, Just a B0

		tau = 0.2e-3  # sec
		def J(w):
			return J_Lorentzian(w, tau)

		a = Particle(1)
		a.set_coords_A((0, 0, 0))
		a.set_gamma(gamma)
		a.set_quadrupole_params(100e3, 0.1, 0, 0)

		S = SpinSystem()
		S.append(a)
		S.create_base_spinsets()

		H_0 = - B0 * S.H_Zeeman_1T_all()
		# print "H_0 (Zeeman):"
		# pprint(H_0)

		R = Redfield_simple(S, H_0, verbose=False)
		R.add_quad_relaxation(a, J)
		R.final_Rab()
		start_dens = prepare_equil(0.1 * H_0, 300)
		end_dens = prepare_equil(H_0, 300)
		O_meas = S.O_measure_all(gamma)
		R.population_measure_analyze(start_dens, end_dens, O_meas)
		T1_my = R.fit_T1()
		w = B0 * gamma
		T1_kim = kimmich_t1(1, 100e3, w, tau)
		# print "Fitted t1:", T1_my
		# print "Kimmich  :", T1_kim
		# print "Rel Diff :", (T1_my - T1_kim) / T1_kim
		# assert allclose(T1_my, T1_kim, rtol=1.23e-4)
	def test_quad_given_simple(self):
		from spinlab.specdens import J_HN
		from spinlab.spingamma import get_gamma

		B0 = 300e6 * 2 * pi / get_gamma("1H")
		gamma = get_gamma("2H")
		w = B0 * gamma

		# tau = 2.1e-5 # s
		tau = 1.6 / w
		# Cole-Cole (alpha = 0.45)
		eps = 1
		delta = 0.45
		def J(w):
			return J_HN(w, tau, delta, eps)

		# T1 = 1.5e-2 # s
		T1 = 3.6e-3
		w_Q = 166e3 * 2 * pi
		# print "B, w, w/(2 pi):", B0, w, w/(2*pi)
		# print 1/T1,
		# 2/15, if you don't have 2 in the Js.
		# print 1/15. * w_Q**2 * (J(w) + 4*J(2*w))

		from spinlab.particle import Particle, SpinSystem
		from spinlab.qmlib import prepare_equil

		a = Particle(1)
		a.set_coords_A((0, 0, 0))
		a.set_gamma(gamma)
		a.set_quadrupole_params(w_Q, 0.0, 0, 0)

		S = SpinSystem()
		S.append(a)
		S.create_base_spinsets()

		H_0 = - B0 * S.H_Zeeman_1T_all()
		# print "H_0 (Zeeman):"
		# pprint(H_0)

		R = Redfield_simple(S, H_0, verbose=False)
		R.add_quad_relaxation(a, J)
		R.final_Rab()
		start_dens = prepare_equil(0.1 * H_0, 300)
		end_dens = prepare_equil(H_0, 300)
		O_meas = S.O_measure_all(gamma)
		R.population_measure_analyze(start_dens, end_dens, O_meas)
		T1_my = R.fit_T1()
		# print "1/T1 from Pert", 1/T1_my

		# print "T1_my, T1", T1_my, T1
		assert allclose(T1_my, T1, rtol=5.8e-2)


if __name__ == '__main__':
	from numpy import pi
	unittest.main()
