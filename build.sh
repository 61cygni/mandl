#touch *.pyx
#rm *.so *.o

python3 setup.py build_ext --inplace

gcc -Wall -g  -MMD -O2 -c -o libbf.o libbf/libbf.c
gcc -Wall -g  -MMD -O2 -c -o cutils.o libbf/cutils.c
gcc -Wall -g  -MMD -O2 -I libbf/ -o nativemandel nativemandel.c  libbf.o cutils.o -lm
