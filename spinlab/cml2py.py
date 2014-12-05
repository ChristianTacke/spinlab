#! /usr/bin/python

from __future__ import division
from pybel import readfile
from openbabel import etab
from sys import argv
from spinlab.spingamma import get_spin_num
from os.path import basename, splitext
import re

nametable = {}

def get_id_for_sym(sym):
	if sym not in nametable:
		nametable[sym] = iter(xrange(1, 10000))
	return sym + str(nametable[sym].next())

class Outputter(object):
	def __init__(self, filename):
		self.of = file(filename, "w")
	def write(self, s):
		self.of.write(s)
	def print_header(self):
		self.write(
		"\n"
		"from __future__ import division\n"
		"from spinlab.spingamma import get_gamma\n"
		"from spinlab.particle import Particle, SpinSystem\n"
		"\n"
		"b_S = SpinSystem()\n")
	def print_atom(self, **kwargs):
		self.write(
		"\n"
		"b_{id} = Particle({spin_num_2}/2, {id!r})\n"
		"b_{id}.set_coords_A({coords!r})\n"
		"b_{id}.set_gamma(get_gamma({iso_sym!r}))\n"
		"b_S.append(b_{id})\n"
		"".format(**kwargs))

name_infile = argv[1]
name_outfile = splitext(basename(name_infile))[0]
name_outfile = re.sub(r"(\A[^A-Za-z]+|[^A-Za-z0-9_]+)", "", name_outfile)
name_outfile += ".py"

mol = readfile("cml", name_infile).next()

print "=== Writing to " + name_outfile

of = Outputter(name_outfile)
of.print_header()

for at in mol.atoms:
	sym = etab.GetSymbol(at.atomicnum)
	if at.isotope:
		iso_sym = str(at.isotope) + sym
	else:
		iso_sym = sym

	try:
		spin_num = get_spin_num(iso_sym)
	except:
		print "*** Ignoring unknown %s atom" % (iso_sym,)
		continue
	spin_num_2 = int(spin_num * 2)
	if spin_num_2 == 0:
		print "*** Atom %s has no spin" % (iso_sym,)
		continue

	id = get_id_for_sym(sym)

	of.print_atom(id = id, iso_sym = iso_sym, \
		spin_num_2 = spin_num_2, coords = at.coords)
