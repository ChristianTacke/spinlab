from __future__ import division
from sys import argv

from numpy import arange, logspace, pi, column_stack, savetxt, exp

from spinlab.qmlib import prepare_equil
from spinlab.particle import SpinSystem
from spinlab.spingamma import get_gamma
from spinlab.analyze import RelaxAnalyze_simple_dd

from spinlab.specdens import J_HN
from spinlab.specdens import J_Lorentzian
from model import b_S, b_N1, b_H1, modify_ring

gamma_H = get_gamma("1H")



# tau_c = 3.35e-7
tau_c = 9.1610e-09
# tau_c_2 = 400e-6 # 1/2e5
tau_c_2 = 2000e-3
# 0.14*J_HN(w, tau_c, 9.4646e-01, 1.0) \
hn1_delta = 1
hn1_eps = 0.94646
hn2_eps = 0.17
def J(w):
	return 0.14*J_HN(w, tau_c, hn1_delta, hn1_eps) \
		+ 0.20 * J_HN(w, tau_c_2, 1, hn2_eps)
	#	+ J_Lorentzian(w, tau_c_2)
	#	+ exp(-((w/1e5)**2)) * 8e-7
	# return 0.3*tau_c/(1 + (tau_c * w)**2) + tau_c_2/(1 + (tau_c_2 * w)**2)

def J_QR(w):
	return 0.004*J_HN(w, tau_c, hn1_delta, hn1_eps)



class Relax_dd_with_QR(RelaxAnalyze_simple_dd):
	def set_J_QR(self, J2):
		self._J_QR = J2
	def add_other_relax(self, R):
		for p in self.spinsys.quadnuclei_iter():
			print "  Adding quadrelax for %r" % (p,)
			R.add_quad_relaxation(p, self._J_QR)
	

out_prefix = "Im3-"

mod_name = "N1H1H2H5"
if len(argv) >= 2:
	mod_name = argv[1]

out_prefix += (mod_name + "-")
modify_ring(b_S, mod_name)

out_prefix += "v2-"


# b_S.remove(b_S.particles["H5"])
# b_S.remove(b_S.particles["H2"])
# b_S.remove(b_S.particles["N1"])

gamma1 = b_H1.gamma
b_S.create_base_spinsets()
H_0_q = b_S.H_quadrupole_all()
H_0_1T = b_S.H_Zeeman_1T_all()

B0_start = 1 # T
H_0_start = H_0_q - B0_start * H_0_1T
start_dens = prepare_equil(H_0_start, 300)

O_meas = b_S.O_measure_all(gamma1)

# freq_range = logspace(4.01, 7, 500)
# freq_range = arange(1e4, 2e6, 2e3)
freq_range = logspace(4.3, 7.6, 320)
# freq_range = logspace(7, 7.01, 3)

b_range = freq_range * 2 * pi / gamma_H
print freq_range
print b_range

analyzer = Relax_dd_with_QR(b_S, H_0_q)
analyzer.set_start_dens(start_dens)
analyzer.set_specdens_fn(J, ())
analyzer.set_J_QR(J_QR)
result_r, result_beta = analyzer.calc_R1(b_range, gamma1)

savetxt(out_prefix + "out.dat", column_stack((freq_range, result_r, result_beta)))

print "Output: %s" % out_prefix
