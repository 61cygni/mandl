#/usr/local/bin/python3 fractal.py --duration=4 --preview 

#/usr/local/bin/python3 fractal.py --color=exp2 --duration=4 --preview 

#/usr/local/bin/python3 fractal.py --algo=mandelbrot --color=exp2 --duration=4 --preview --gif=test.gif 
#/usr/local/bin/python3 fractal.py --color=exp2 --algo=julia --duration=4 --preview --gif=test.gif 

#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=32 --color=exp2 --max-iter=1000 --verbose=3 --preview

#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=32 --color=exp2 --max-iter=1000 --verbose=3 --preview --cache

#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=64 --color=exp2 --max-iter=3000 --verbose=3

#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=64 --color=exp2 --max-iter=1500 --img-w=600 --img-h=400 --fps=2 

#/usr/local/bin/python3 fractal.py --snapshot="snapshot.gif" --color --max-iter=2000 --zoom=1500 --img-w=2000 --img-h=1600

#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=8 --scaling-factor=.9 --color=exp2 --max-iter=800 --zoom=150 --img-w=600 --img-h=400 --fps=8 --palette-test=exp2


#/usr/local/bin/python3 fractal.py --snapshot="snapshot.gif" --color=list --max-iter=5000  --img-w=10000 --img-h=7000 --center="0+0j" --julia="-.8+.156j"

#/usr/local/bin/python3 fractal.py --color=list --max-iter=2000 --img-w=1600 --img-h=1300 --center="0+0j" --julia="-.8+.156j" --duration=32 --fps=16  --gif="julia.gif" --cmplx-w=3.2 --cmplx-h=2.5 --julia-walk="[0.355+0.355j, 0+0.8j,0.355+0.355j]"

# Dive into a Julia set
#/usr/local/bin/python3 fractal.py --color=list --max-iter=2000  --img-w=600 --img-h=400 --center=".255+.29j" --julia="0.285+0.01j" --gif="test.gif"  --duration=4

# Walk the Julia space 
#/usr/local/bin/python3 fractal.py --color=list --max-iter=200 --img-w=400 --img-h=300 --center="0+0j" --duration=32 --fps=16  --gif="julia.gif" --cmplx-w=3.2 --cmplx-h=2.5 --julia-walk="[0.355+0.355j, 0+0.8j,0.355+0.355j]"

# test cache
#/usr/local/bin/python3 fractal.py --gif=test.gif --duration=32 --color=exp2 --max-iter=1000 --verbose=3 --preview --cache

# take a snapshot of a busy section with a modest zoom into the  mandelbrought set. This is a good expose 
# of the ability for distance estimation to show detail 

#/usr/local/bin/python3 fractal.py --algo=mandeldistance --cmplx-w=.001 --cmplx-h=.00075 --snapshot="dsnap.gif" --img-w=2048 --img-h=1536 --max-iter=500

/usr/local/bin/python3 fractal.py --algo=smooth --gif="smooth.gif" --img-w=3840 --img-h=2160 --max-iter=512 --duration=16 --fps=16  --scaling=.9  --smooth --color="(.1,.2,.3)" --cache --center="-0.235125+0.827215j"


# test hpcsmooth with a deep dive ...
python3 fractal.py --algo=hpcsmooth --dive --duration=60  --img-w=160 --img-h=120 --max-iter=3000 --scaling=.80

# test csmooth with a keyframe dive 
python3 fractal.py --algo=csmooth --dive --keyframe=7

# generate a 12k snapshot with csmooth
python3 fractal.py --algo=csmooth --res=12k
