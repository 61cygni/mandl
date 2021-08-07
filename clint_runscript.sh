
#python3.9 fractal.py --gif=low_res_02-04.gif --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=2 --clip-frame-count=3
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=0 --clip-frame-count=10&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=10 --clip-frame-count=10&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=20 --clip-frame-count=10&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=30 --clip-frame-count=10&
#

#python3.9 fractal.py --gif=low_res_demo.gif --algo=smooth --demo --burn --color --invalidate-cache 
python3.9 fractal.py --gif=low_res_demo.gif --algo=smooth --demo --burn --color --invalidate-cache --flint

#python3.9 fractal.py --gif=low_res_demo.gif --algo=mandelbrot --demo --burn --invalidate-cache --flintcustom --clip-start-frame=10 --clip-frame-count=10
#python3.9 fractal.py --gif=low_res_demo.gif --algo=mandelbrot --demo --burn --invalidate-cache --flintcustom
#python3.9 fractal.py --gif=low_res_demo_e60.gif --algo=mandelbrot --demo --burn --invalidate-cache --flintcustom

#python3.9 fractal.py --gif=low_res_demo.gif --algo=mandelbrot --demo --burn --color --invalidate-cache --flintcustom --clip-start-frame=25 --clip-frame-count=5

#python3 -m cProfile fractal.py --gif=low_res_demo.gif --flint --algo=smooth --demo --burn --color --invalidate-cache

#python3 fractal.py --gif=native_demo.gif --color=exp2 --demo --burn
#python3 fractal.py --gif=native_demo_fresh.gif --color=exp2 --demo --burn --invalidate-cache

#python3 fractal.py --gif=native_distance_demo.gif --algo=mandeldistance --demo --burn --smooth
#python3 fractal.py --gif=native_distance_demo_fresh.gif --algo=mandeldistance --demo --burn --smooth --invalidate-cache

#python3 fractal.py --gif=smooth_demo.gif --algo=smooth --demo --burn --color 
#python3 fractal.py --gif=smooth_demo_fresh.gif --algo=smooth --demo --burn --color --invalidate-cache
#python3 fractal.py --gif=flint_smooth_demo_fresh.gif --algo=smooth --flint --demo --burn --color --invalidate-cache

#python3 fractal.py --gif=flint_distance_demo.gif --algo=mandeldistance --flint --demo --burn --smooth
#python3 fractal.py --gif=flint_distance_demo_fresh.gif --algo=mandeldistance --flint --demo --burn --smooth --invalidate-cache

#python3 fractal.py --gif=flint_demo.gif --color=exp2 --demo --flint
#python3 fractal.py --gif=flint_demo_fresh.gif --color=exp2 --demo --flint --invalidate-cache

#python3 fractal.py --gif=gmp_demo.gif --color=exp2 --demo --gmp
#python3 fractal.py --gif=gmp_demo_fresh.gif --color=exp2 --demo --gmp --invalidate-cache

#python3 fractal.py --algo='julia' --gif=native_julia_demo.gif --color=exp2 --demo-julia-walk
#python3 fractal.py --algo='julia' --gif=native_julia_demo_fresh.gif --color=exp2 --demo-julia-walk --invalidate-cache

# Let's test out range overlaps for separate invocations.
# If the floats are reliable enough, then the middle (overlapping) frames shouldn't have
# to render on the second invocation.

## Put some effort into trying to make multiple invocations around the same end-point values 
## actually end up with the same values for sub-clips.  Hopefully this means overlapping frames
## look up as identical in the cache.
##python3 fractal.py --gif=phase_01.gif --color=exp2 --burn --project-name='phase01' --duration='1.0' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.8 --clip-start-frame=0 --clip-total-frames=1 --build-cache
#python3 fractal.py --gif=phase_01.gif --color=exp2 --burn --project-name='phase01' --duration='1.0' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.8 --clip-start-frame=5 --clip-total-frames=6 --build-cache
##python3 fractal.py --gif=phase_01.gif --color=exp2 --burn --project-name='phase01' --duration='1.0' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.8 --clip-start-frame=1 --clip-total-frames=1 --build-cache
#python3 fractal.py --gif=phase_02.gif --color=exp2 --burn --project-name='phase01' --duration='1.0' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.8 --clip-start-frame=0 --clip-total-frames=11 --build-cache 

