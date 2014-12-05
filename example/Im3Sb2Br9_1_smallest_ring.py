
from __future__ import division
from spinlab.spingamma import get_gamma
from spinlab.particle import Particle, SpinSystem

b_S = SpinSystem()

b_N1 = Particle(2/2, 'N1')
b_N1.set_coords_A((0.90775539248055437, 0.17541924877229512, 9.781624233426685))
b_N1.set_gamma(get_gamma('N'))
b_S.append(b_N1)

b_H1 = Particle(1/2, 'H1')
b_H1.set_coords_A((0.97855851165413232, 0.32452631511036528, 8.9178460838202902))
b_H1.set_gamma(get_gamma('H'))
b_S.append(b_H1)

b_H2 = Particle(1/2, 'H2')
b_H2.set_coords_A((-0.21582364924332287, -1.5287860586992259, 9.8789098270683624))
b_H2.set_gamma(get_gamma('H'))
b_S.append(b_H2)

b_N2 = Particle(2/2, 'N2')
b_N2.set_coords_A((0.39052198602447336, -0.71220609352626074, 11.634472585056786))
b_N2.set_gamma(get_gamma('N'))
b_S.append(b_N2)

b_H3 = Particle(1/2, 'H3')
b_H3.set_coords_A((0.055244023491561205, -1.2621478396006536, 12.234400412513786))
b_H3.set_gamma(get_gamma('H'))
b_S.append(b_H3)

b_H4 = Particle(1/2, 'H4')
b_H4.set_coords_A((1.3500632866642888, 0.7008019191927608, 12.77094520168909))
b_H4.set_gamma(get_gamma('H'))
b_S.append(b_H4)

b_H5 = Particle(1/2, 'H5')
b_H5.set_coords_A((1.9442055980544306, 1.7182380854907549, 10.60560373108876))
b_H5.set_gamma(get_gamma('H'))
b_S.append(b_H5)
