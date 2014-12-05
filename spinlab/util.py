from __future__ import division
import numpy
from numpy import pi, real_if_close, alltrue, allclose, array
from numpy.linalg import eig
from os.path import isfile, splitext
from spinlab.spingamma import gamma_H, gamma_N, gamma_K

b_factor = gamma_H / (2*pi)


def a2s(a, lenpref):
	return numpy.array2string(a, precision=4, suppress_small=True, \
		prefix = ("." * lenpref))

def write_energies_as_freq(out, H, format = "\t%s"):
		(vals, vecs) = eig(H)
		vals = real_if_close(vals)
		vals.sort()
		for q in vals:
			out.write(format % (q / (2*pi),))

class EnergyWriter(object):
	def __init__(self, f):
		self.f = f
		self.last_vals = None
		self.last_x = None
		self.delta = None
		self.delta_x = None
		self.last_idx = None
		self.prefix = None
		self.postfix = "\n"
	def write(self, s):
		self.f.write(s)
	def close(self):
		self.f.close()
	def write_energies(self, x, e):
		e = array(e).real
		e.sort()
		showthis = False
		if x and self.delta is not None:
			newdelta_x = x - self.last_x
			extpol = self.last_vals + newdelta_x * self.delta
			idx = extpol.argsort().argsort()
			# print extpol
			e = e[idx]

			# print idx, self.delta, \
			# 	(e - self.last_vals) / newdelta_x
			if newdelta_x > (self.delta_x * 2):
				showthis = True
			elif not alltrue(idx == self.last_idx):
				showthis = True
			elif not allclose(self.delta, \
					  (e - self.last_vals) / newdelta_x,
					  rtol=1e-3):
				showthis = True
			self.last_idx = idx
			# print e
		else:
			showthis = True
		if showthis:
			self._dump_update(x, e)
		# else:
		# 	print "[Dropped %s]" % str(x),
	def _dump_update(self, x, e):
		format = "\t%s"
		# print "Dump and update:", x,
		if x and self.last_x is not None:
			self.delta_x = (x - self.last_x)
			# print self.delta_x,
			self.delta = (e - self.last_vals) / self.delta_x
		if x is not None:
			# print "Setting last",
			self.last_vals = e
			self.last_x = x

		if self.prefix:
			self.write(self.prefix)
		for q in e:
			self.write(format % (q,))
		if self.postfix:
			self.write(self.postfix)
	def write_energies_as_freq(self, x, H):
		(vals, vecs) = eig(H)
		self.write_energies(x, vals / (2*pi))
	def set_prefix(self, p):
		self.prefix = p
	def write_plot_cmds_gnu(self, plotf, numlevels = None):
		if not hasattr(plotf, "write"):
			if isfile(plotf):
				return
			plotf = file(plotf, "w")
		if numlevels is not None:
			l = numlevels
		else:
			l = len(self.last_vals)
		plotf.write( \
"""# Generated
set term pdf
set size 15
set nokey

set xlabel "$f$ / Hz"
set ylabel "E"
""")
		plotf.write("""set output "%s"\n""" % \
			(splitext(plotf.name)[0] + "-e.pdf", ) )

		plotf.write("plot \\\n")
		for i in xrange(l):
			plotf.write("\t\"%s\" u 2:%d w l" % \
				    (self.f.name, i+3,))
			if i < (l-1):
				plotf.write(", \\")
			plotf.write("\n")
	def write_plot_cmds_gle(self, plotf, numlevels = None):
		if not hasattr(plotf, "write"):
			if isfile(plotf):
				return
			plotf = file(plotf, "w")
		if numlevels is not None:
			l = numlevels
		else:
			l = len(self.last_vals)
		plotf.write("  ! Generated\n")
		plotf.write('  data "%s"' % (self.f.name, ) )
		for i in xrange(l):
			plotf.write(" d%d=c2,c%d" % \
				    (i+1, i+3,))
		plotf.write("\n")

		for i in xrange(l):
			plotf.write("  d%d line color " % \
				    (i+1, ))
			plotf.write(("red", "blue", "black")[(i//2) % 3])
			if i % 2 == 1:
				plotf.write(" lstyle 2")
			plotf.write("\n")


def try_energywriter():
	import sys
	f = EnergyWriter(sys.stdout)
	f.write_energies(0, (10, 20, 30))
	f.write_energies(1, (15, 20, 30))
	f.write_energies(2, (20, 20, 30))
	f.write_energies(3, (25, 22, 28))
	f.write_energies(4, (30, 23, 27))
	f.write_energies(5, (30, 23.9, 26.1))
	f.write_energies(6, (30, 23, 27))
	f.write_energies(7, (30, 22, 28))
	f.write_plot_cmds_gnu(sys.stdout)
	f.write_plot_cmds_gle(sys.stdout)

	f = EnergyWriter(sys.stdout)
	for i in range(13):
		f.write_energies(i, (i, 12-i))
	f.write_energies(13, (4, 6))

if __name__ == '__main__':
	try_energywriter()
