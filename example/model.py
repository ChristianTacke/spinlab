

from Im3Sb2Br9_1_smallest_ring import b_S, \
	b_N1, b_H1, b_N2, b_H3

# N1, H1
b_N1.set_quadrupole_params(6.06e6, 0.57, 0, 0)
b_N1.set_quadrupole_angles_tp(b_H1)

# N2, H3
b_N2.set_quadrupole_params(6.06e6, 0.57, 0, 0)
b_N2.set_quadrupole_angles_tp(b_H3)


def modify_ring(spinsys, modify_name):
	if modify_name == "full":
		pass
	elif modify_name == "wo-N2":
		spinsys.remove(spinsys.particles["N2"])
	elif modify_name == "wo-H4H5":
		spinsys.remove(spinsys.particles["H4"])
		spinsys.remove(spinsys.particles["H5"])
	elif modify_name in ("N1H1H2H5", "N1H1H2", "N1H1", "H1H2"):
		spinsys.remove(spinsys.particles["N2"])
		spinsys.remove(spinsys.particles["H3"])
		spinsys.remove(spinsys.particles["H4"])

		if modify_name == "N1H1H2":
			spinsys.remove(spinsys.particles["H5"])
		elif modify_name == "N1H1":
			spinsys.remove(spinsys.particles["H5"])
			spinsys.remove(spinsys.particles["H2"])
		elif modify_name == "H1H2":
			spinsys.remove(spinsys.particles["H5"])
			spinsys.remove(spinsys.particles["N1"])
	else:
		raise ValueError("Unknown modification %s" % modify_name)
