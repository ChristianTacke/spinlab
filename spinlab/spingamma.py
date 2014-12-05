from __future__ import division


# (gammas from wikipedia: not known for sure)
# Danuta uses nearly the same
gamma_H = 26.752e7
gamma_K = 1.248e7
gamma_N = 1.933e7

# gamma_H from NIST
# http://physics.nist.gov/cgi-bin/cuu/Value?gammap
gamma_H = 2.675222099e8

# According to Bruker table
gamma_K = gamma_H * 0.04667


gamma_table = {
	"1H": (1/2, gamma_H),
	"2H": (1, 4.107e7),  # wikipedia!
	"H": "1H",
	"39K": (3/2, gamma_K),
	"K": "39K",
	"14N": (1, gamma_N),
	"N": "14N",
	"139La": (7/2, 3.808e7),  # wikipedia!
	"La": "139La",
	"19F": (1/2, 25.17e7), # wikipedia!
	"F": "19F",
}


def get_spin_info(name):
	r = gamma_table[name]
	if isinstance(r, basestring):
		print "*** Warning: Using %s for %s" % (r, name)
		gamma_table[name] = gamma_table[r]
		r = gamma_table[r]
	return r

def get_spin_num(name):
	return get_spin_info(name)[0]

def get_gamma(name):
	return get_spin_info(name)[1]
