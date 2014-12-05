


from tacke.physconst import e, h_bar, m_Proton


nuclear_magneton = e * h_bar / (2 * m_Proton)

print nuclear_magneton

print "Spin number as float>",
spin = float(raw_input())

print "mu in units of nuclear magneton (nm)>",
mu = float(raw_input())

print " Spin:", spin
print "   mu:", mu

print "gamma:", mu * nuclear_magneton / (spin * h_bar)
