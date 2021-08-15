#!/bin/bash

# Example of embarassingly parallel start/stop range commands that all write
# out tiffs that can be assembled with 'compile_video.py' after finishing.


# Make sure the params in --demo will allow the frame numbers to be made.
# Best to test out the LAST of these frame range manually to make sure 
# the subranges are all valid
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=0 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=25 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=50 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=75 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=100 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=125 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=150 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flint --clip-start-frame=175 --clip-frame-count=25&


# Looks like it takes 13 or 14 processes to hit 100% cpu?
# Deeper depths, was still a bit over, so dropping to 12 for now?
# Seem to see it hit 122% cpu per process sometimes...

#processCount=12
# 12 is ending up in 13 processes... hrm.
# Honestly, I can't imagine cache is doing anything but thrashing at higher bit depths...
# So let's run 6 processes next attempt? (Seems like it sped up at the end, so probably right).
#processCount=12
# Tried 12 again, think it's very slow.

processCount=7

# 30 frames per batch, 4 batches, = 120 frames, going a little extra
# 30 frames per process, let's say 6 processes? = 180 frames
startFrame=750
lastNumber=778

stride=$(((lastNumber-startFrame)/processCount))

batchCount=0
pidList=()

until [ $startFrame -ge $lastNumber ]
do
    echo startFrame: $startFrame &
    python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=${startFrame} --clip-frame-count=${stride}&
    pidList[${batchCount}]=$!
    ((startFrame=startFrame+stride))
    ((batchCount=batchCount+1))
done

# wait for all pids
for currPID in ${pidList[*]}; do
    wait $currPID
    echo "Done"
done


#echo "Batch 4"
#date
#
#startFrame=700
#lastNumber=800
#
#stride=$(((lastNumber-startFrame)/processCount))
#
##stride=45
##lastNumber=359
#batchCount=0
#pidList=()
#
#until [ $startFrame -ge $lastNumber ]
#do
#    echo startFrame: $startFrame &
#    python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=${startFrame} --clip-frame-count=${stride}&
#    pidList[${batchCount}]=$!
#    ((startFrame=startFrame+stride))
#    ((batchCount=batchCount+1))
#done
#
## wait for all pids
#for currPID in ${pidList[*]}; do
#    wait $currPID
#    echo "Done"
#done
#
#
#echo "Batch 5"
#date
#
#startFrame=800
#lastNumber=900
#
#stride=$(((lastNumber-startFrame)/processCount))
#
#batchCount=0
#pidList=()
#
#until [ $startFrame -ge $lastNumber ]
#do
#    echo startFrame: $startFrame &
#    python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=${startFrame} --clip-frame-count=${stride}&
#    pidList[${batchCount}]=$!
#    ((startFrame=startFrame+stride))
#    ((batchCount=batchCount+1))
#done
#
## wait for all pids
#for currPID in ${pidList[*]}; do
#    wait $currPID
#    echo "Done"
#done


# AFTER all the processes finish
python3.9 compile_video.py --dir='demo1_cache/image_frames/flint/mandelbrot_solo' --out='compiled.mp4'




# Custom flint library version - attempt at optimization
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=0 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=25 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=50 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=75 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=100 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=125 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=150 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=175 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=0 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=25 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=50 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=75 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=100 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=125 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=150 --clip-frame-count=25&
#python3.9 fractal.py --algo=mandelbrot_solo --demo --burn --flintcustom --clip-start-frame=175 --clip-frame-count=25&
