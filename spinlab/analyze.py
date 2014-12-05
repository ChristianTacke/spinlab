from __future__ import division

import sys
from numpy import empty, savetxt
from spinlab.qmlib import prepare_equil, apply_liouville, dot
from spinlab.relax import Redfield_simple


class PolTransAnalyze(object):
	def __init__(self, spinsystem, B0_range, tPT):
		self.spinsystem = spinsystem
		self.B0_range = B0_range
		self.tPT = tPT
		self.temp = 3 # Kelvin

		self.msg = sys.stderr.write
		self.flush = sys.stderr.flush
	def fmsg(self, msg):
		self.msg("   [] " + msg)
		self.flush()
	def newline(self):
		self.msg("\n")

	def _prepare_0(self):
		b_S = self.spinsystem

		self.fmsg("Setting up base matrixes")
		b_S.create_base_spinsets()
		self.newline()

	def _prepare(self):
		b_S = self.spinsystem

		self.fmsg("Quadrupole")
		b_Hq = b_S.H_quadrupole_all()
		self.newline()

		self.fmsg("Dipole-Dipole")
		b_Hdd = b_S.H_dipoledipole_all()
		self.newline()

		self.fmsg("Zeeman")
		self.b_H_B_1T = b_S.H_Zeeman_1T_all()
		self.newline()

		self.b_H_om = b_Hq + b_Hdd

		b_H_init = self.b_H_om + self.b_H_B_1T

		self.b_start_dens = prepare_equil(b_H_init, self.temp)

	def O_measure(self, gamma):
		self.fmsg("Measure Operator:")
		O = self.spinsystem.O_measure_all(gamma, disp_cb=self.msg)
		self.newline()
		return O

	def set_O_measure(self, O):
		self.b_O_meas = O
		self.b_O_meas_T_flat = O.T.reshape(-1)

	def run_v1(self, resultline):
		assert resultline.shape == self.B0_range.shape
		self.fmsg("Starting")

		# Shadow variables locally for speed
		# ... and consistency with my old code ;o)
		msg = self.msg
		b_H_om = self.b_H_om
		b_H_B_1T = self.b_H_B_1T
		b_start_dens = self.b_start_dens
		tPT = self.tPT
		b_O_meas = self.b_O_meas
		b_O_meas_T_flat = self.b_O_meas_T_flat

		i = 0
		for b in self.B0_range:
			msg(".")
			b_H = b_H_om + b * b_H_B_1T

			r = apply_liouville(b_start_dens, b_H, tPT)
			# v = dot(r, b_O_meas).trace()
			v = dot(r.reshape(-1), b_O_meas_T_flat)
			resultline[i] = v.real

			i += 1

		self.newline()


class RelaxAnalyze_simple_dd(object):
	def __init__(self, spinsys, H_0_base):
		self.spinsys = spinsys
		self.H_0_base = H_0_base
	def set_start_dens(self, start_dens):
		self.start_dens = start_dens
	def set_specdens_fn(self, J, J_extraparams):
		self.J = J
		self.J_extraparams = J_extraparams
	def _J(self, w):
		return self.J(w, *self.J_extraparams)

	def add_other_relax(self, R):
		"Hook, so you can add more relaxation inside the inner loop"
		pass

	def calc_R1(self, b_range, measure_gamma, mag_save_file=None):
		H_0_base = self.H_0_base
		S = self.spinsys
		start_dens = self.start_dens
		H_0_1T = self.spinsys.H_Zeeman_1T_all()
		O_meas = self.spinsys.O_measure_all(measure_gamma)
		result_r = empty((len(b_range),))
		result_beta = empty((len(b_range),))

		if False:
			f_eig = file("noise-eig.dat", "w")

		loc_mag_save_file = mag_save_file
		for i in xrange(len(b_range)):
			if mag_save_file is not None:
				loc_mag_save_file = mag_save_file.format(i=i)
			if False:
				def dump_reldata(e_vals):
					f_eig.write("%d\t" % i)
					savetxt(f_eig, e_vals[None, :])

			B0 = b_range[i]

			H_0 = H_0_base - B0 * H_0_1T

			R = Redfield_simple(S, H_0, verbose=False)
			if False:
				R.set_dump_reldata(dump_reldata)
			for a, b in self.spinsys.tp_iter():
				# print "  Adding %r, %r" % (a,b)
				R.add_dd_relaxation(a, b, self._J)
			self.add_other_relax(R)
			R.final_Rab()

			end_dens = prepare_equil(H_0, 300)
			R.population_measure_analyze(start_dens, end_dens, O_meas)
			T1_my = R.fit_T1(loc_mag_save_file)

			print i, T1_my
			result_r[i] = 1/T1_my
			result_beta[i] = R.fit_result[0][3]

		if False:
			f_eig.close()
		return result_r, result_beta
