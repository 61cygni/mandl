# Example of embarassingly parallel start/stop range commands that all write
# out tiffs that can be assembled with 'compile_video.py' after finishing.


# Make sure the params in --demo will allow the frame numbers to be made.
# Best to test out the LAST of these frame range manually to make sure 
# the subranges are all valid
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=0 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=25 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=50 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=75 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=100 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=125 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=150 --clip-frame-count=25&
python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=175 --clip-frame-count=25&

# AFTER all the processes finish
#python3.9 compile_video.py --dir='demo1_cache/image_frames/flint/mandelbrot_solo' --out='compiled.gif'




## Custom flint library version - attempt at optimization
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=0 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=25 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=50 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=75 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=100 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=125 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=150 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=175 --clip-frame-count=25&
