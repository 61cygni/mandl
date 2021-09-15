
CFLAGS=-Wall 
#INC=-I/usr/local/include/flint
INC=
#LIBS=-lm -lflint

all: arb_pythonlib mpfr_pythonlib

# In python, "import arb_fractalmath"
# arb_fractal_lib.c
# arb_fractal_lib.h
# (compiles into)
# libarbfractalmath.a
#
# arb_fractalmath.pyx
# (makes)
# arb_fractalmath.c
# arb_fractalmath.so

# In python, "import mpfr_fractalmath"
# mpfr_fractal_lib.c
# mpfr_fractal_lib.h
# (compiles into)
# libmpfrfractalmath.a
#
# mpfr_fractalmath.pyx
# (makes)
# mpfr_fractalmath.c
# mpfr_fractalmath.so

arb_pythonlib: libarbfractalmath.a arb_fractalmath.pyx
	python3.9 setup_arb.py build_ext --inplace
	
libarbfractalmath.a: arb_fractal_lib.o
	ar -ru $@ $^ $(LIBS)
	ranlib $@
#	$(CC) -shared $(LDFLAGS) -o $@ $^ $(LIBS)

mpfr_pythonlib: libmpfrfractalmath.a mpfr_fractalmath.pyx
	python3.9 setup_mpfr.py build_ext --inplace

libmpfrfractalmath.a: mpfr_fractal_lib.o
	ar -ru $@ $^ $(LIBS)
	ranlib $@
	
%.o: %.c
	$(CC) $(INC) $(CFLAGS) -c -o $@ $<

clean:
	python3.9 setup_arb.py clean --all
	python3.9 setup_mpfr.py clean --all
	rm -f *.o *.d *~ *.so *.a 
	rm -rf build

