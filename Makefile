PROGS=nativemandel 
LIBS=-lm
CFLAGS=-Wall -I libbf/ -g $(PROFILE) -MMD

all: $(PROGS)

nativemandel: nativemandel.o libbf.o cutils.o
	$(CC) $(LDFLAGS) -I libbf/ -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(PROGS) *.o *.d *~
