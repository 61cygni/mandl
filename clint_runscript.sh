
#python3 fractal.py --gif=native_demo.gif --color=exp2 --demo
python3 fractal.py --gif=native_demo_fresh.gif --color=exp2 --demo --invalidate-cache

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

