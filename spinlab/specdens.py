from __future__ import division

def J_Lorentzian(w, tau_c):
	return 2 * tau_c / (1. + ((w*tau_c)**2))

from numpy import sin, cos, arctan, pi, copy

# Havriliak-Negami (HN)
# eps=delta=1: HN -> Debeye spectral density
# eps=1: HN -> Cole-Cole (CC) spectral density
# delta=1: HN -> Davidson-Cole (DC) spectral density
def J_HN(omega, tau, delta, eps):
	if (0.0 < delta <= 1.0) and (0.0 < eps <= 1.0):
		omega = copy(omega)
		if omega.shape == ():
			if omega==0:
				omega = tau * 1e-30
		else:
			omega[omega==0] = tau * 1e-30
		delta *= 1.0
		eps *= 1.0
		a=(omega*tau)**delta
		b=0.5*pi*delta
		J=2.0* sin(eps*arctan((a*sin(b))/(1.0+a*cos(b))))/omega/(1.0+2.0*a*cos(b)+a**2.0)**(0.5*eps)
		return J
	else:
		# print 'input valid delta or epsilon'
		return 0.0*omega
