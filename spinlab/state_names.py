from __future__ import division

from numpy import diagonal, array
from spinlab.particle import Particle, SpinSystem

def state_names(S):
	l = []
	for q in S.particle_list:
		l.append(diagonal(q.spinset[2]))
	l = array(l).T
	return l


if __name__ == '__main__':
	a = Particle(1/2)
	b = Particle(1)
	S = SpinSystem()
	S.append(a)
	S.append(b)
	S.create_base_spinsets()

	l = state_names(S)
	print l
	l2 = S.state_names()
	print l2
