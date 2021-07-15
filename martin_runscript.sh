#/usr/local/bin/python3 mandelbrot.py --color=exp2 --duration=16 --preview 

#/usr/local/bin/python3 mandelbrot.py --gif=test.gif --duration=32 --color=exp2 --max-iter=1000 --verbose=3 --preview

#/usr/local/bin/python3 mandelbrot.py --gif=test.gif --duration=64 --color=exp2 --max-iter=3000 --verbose=3

/usr/local/bin/python3 mandelbrot.py --gif=test.gif --duration=64 --color=exp2 --max-iter=1500 --img-w=600 --img-h=400 --fps=2 

#/usr/local/bin/python3 mandelbrot.py --snapshot="snapshot.gif" --color --max-iter=2000 --zoom=1500 --img-w=2000 --img-h=1600

#/usr/local/bin/python3 mandelbrot.py --gif=test.gif --duration=8 --scaling-factor=.9 --color=exp2 --max-iter=800 --zoom=150 --img-w=600 --img-h=400 --fps=8 --palette-test=exp2

/usr/local/bin/python3 mandelbrot.py --color=list --max-iter=2000  --img-w=600 --img-h=400 --center=".255+.29j" --julia="0.285+0.01j" --gif="test.gif"  --duration=4

/usr/local/bin/python3 mandelbrot.py --snapshot="snapshot.gif" --color=list --max-iter=5000  --img-w=10000 --img-h=7000 --center="0+0j" --julia="-.8+.156j"

/usr/local/bin/python3 mandelbrot.py --color=list --max-iter=2000  --img-w=1600 --img-h=1300 --center="0+0j" --julia="-.8+.156j" --duration=24 --fps=16  --gif="julia.gif" --cmplx-w=3.2 --cmplx-h=2.5 --julia-walk="-0.52+0.57j"
