

BASE_DATA = /home/element/share/Tacke/Wlodek/Messdaten_good/T1_Anpassung_mit_mono_und_Kohlrausch/Kohlrausch
BASE_RELAX = ../../Relaxometrie/2012-04-Im3Sb2Br9

FILES = \
	plot-1.pdf \
	Im5Bi2Cl11_T1fit_293K.dat \
	Im3-N1H1H2H5-v2-out.dat

all: $(FILES)

plot-1.pdf: plot-1.gle Im3-N1H1H2H5-v2-out.dat
	gle -d pdf $<

Im3-N1H1H2H5-v2-out.dat: fullring-2.py model.py
	python fullring-2.py

Im5Bi2Cl11_T1fit_293K.dat: $(BASE_DATA)/Im5Bi2Cl11_T1fit_293K.dat
	cp -p $< $@


.PHONY: copy-things

copy-things:
	for f in fullring-2.py model.py Im3Sb2Br9_1_smallest_ring.py; \
	do \
		cp -puv $(BASE_RELAX)/$$f $$f; \
	done
