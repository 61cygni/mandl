PROGS=hpnative ldnative mpfrnative
LIBS=-lm -lmpfr
CFLAGS=-Wall -I thirdparty/ -g $(PROFILE) -MMD

all: $(PROGS) pre-build

pre-build:
	python3 setup.py build_ext --inplace

mpfrnative: mpfrnative.o  libattopng.o
	$(CC) $(LDFLAGS) -I libbf/ -o $@ $^ $(LIBS)

hpnative: hpnative.o libbf.o cutils.o libattopng.o
	$(CC) $(LDFLAGS) -I libbf/ -o $@ $^ $(LIBS)

ldnative: ldnative.o libbf.o cutils.o libattopng.o
	$(CC) $(LDFLAGS) -I libbf/ -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: thirdparty/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(PROGS) *.o *.d *~ *.so

    
