
python3 mandelbrot.py --gif=native_demo.gif --color=exp2 --demo
#python3 mandelbrot.py --gif=native_demo_fresh.gif --color=exp2 --demo --invalidate-cache

#$python3 mandelbrot.py --gif=flint_demo.gif --color=exp2 --demo --flint
#python3 mandelbrot.py --gif=flint_demo_fresh.gif --color=exp2 --demo --flint --invalidate-cache

#python3 mandelbrot.py --gif=native_julia_demo.gif --color=exp2 --demo-julia-walk
#python3 mandelbrot.py --gif=native_julia_demo_fresh.gif --color=exp2 --demo-julia-walk --invalidate-cache

# Let's test out range overlaps for separate invocations.
# If the floats are reliable enough, then the middle (overlapping) frames shouldn't have
# to render on the second invocation.


#python3 mandelbrot.py --gif=phase_01.gif --color=exp2 --burn --project-name='phase01' --duration='0.5' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.9 --zoom=5 --build-cache 
#python3 mandelbrot.py --gif=phase_01.gif --color=exp2 --burn --project-name='phase01' --duration='0.5' --fps=11.988 --max-iter=255 --img-w=1024 --img-h=768 --cmplx-w='5.0' --cmplx-h='3.5' --center="-1.769383179195515018213+.00423684791873677221j" --scaling-factor=0.9 --zoom=0 --build-cache 

