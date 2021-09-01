# Simple pygame driver of main fractal program 

import pygame
import subprocess

from decimal import *

hpf = Decimal
getcontext().prec = 500 

ALGO = "ldnative"
BURN = True 
DISPLAY_WIDTH  = 640

image_w = 640
image_h = 480

red   = 0.1
green = 0.2
blue  = 0.3

real = hpf(-.745)
imag = hpf(.186)
#real = hpf(-1)
#imag = hpf(0)

c_width = hpf(5)
c_height = hpf(0)

scaling = .1
epoch = 0


DRIVER_VER = "0.01"

def display():
    global real
    global imag
    global epoch
    global c_width
    global c_height
    global ALGO
    global BURN
    global DISPLAY_WIDTH
    global DISPLAY_HEIGHT
    global image_w
    global image_h

    pygame.init()

    bg = pygame.image.load("pyfractal.gif")

    width    = bg.get_width()
    height   = bg.get_height()
    new_h    = int(DISPLAY_WIDTH * (float(height) / float(width)) )
    c_height = c_width * (hpf(height) / hpf(width)) 

    bg = pygame.transform.scale(bg, (DISPLAY_WIDTH, new_h))

    # Set up the drawing window
    screen = pygame.display.set_mode([DISPLAY_WIDTH, new_h])

    re_start = real - (c_width  / hpf(2.))
    im_start = imag - (c_height / hpf(2.))

    # Run until the user asks to quit
    running = True
    while running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h:
                    ALGO = "hpnative"
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    ALGO = "csmooth"
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    ALGO = "mpfrnative"
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    ALGO = "ldnative"
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_EQUALS:
                    image_w = image_w * 2
                    image_h = image_h * 2
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_MINUS:
                    image_w = image_w / 2
                    image_h = image_h / 2
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    ALGO = "mandeldistance"
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_b:
                    BURN = True
                    return (real, imag)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    # zoom in
                    epoch = epoch + 1
                    c_width  = hpf(scaling) * c_width
                    c_height = hpf(scaling) * c_height
                    return (real, imag)


            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()   
                fxoffset = float(pos[0])/DISPLAY_WIDTH
                fyoffset = float(pos[1])/float(new_h)

                print("x %d, y %d"%(pos[0],pos[1]))
                print("fxoff %f, fyoff %f"%(fxoffset,fyoffset))
                real = re_start + (hpf(fxoffset) * c_width)
                imag = im_start + (hpf(fyoffset) * c_height)
                #imag = imag * hpf(-1)

                print("Real %s, Image %s"%(str(real),str(imag)))
                # Done! Time to quit.

                # zoom in
                epoch = epoch + 1
                c_width  = hpf(scaling) * c_width
                c_height = hpf(scaling) * c_height

                return (real,imag)
                

        screen.blit(bg, (0, 0))

        pygame.display.flip()

def run():
    global real
    global imag
    global scaling
    global epoch
    global c_width
    global c_height
    global image_w
    global image_h
    global BURN

    burn_str = ""
    if BURN:
        burn_str = "--burn"

    while 1:
        cmd = "python3 fractal.py %s --verbose=3 --algo=%s --cmplx-w=%s --cmplx-h=%s --img-w=%d --img-h=%d --real=\"%s\" --imag=\"%s\" " \
              %(burn_str, str(ALGO), str(c_width), str(c_height),image_w,image_h,str(real),str(imag))
        print(" + Driver running comment: "+cmd)
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
        pygame.quit()
        real, imag = display()
        

if __name__ == "__main__":

    print("++ driver.py version %s" % (DRIVER_VER))
    
    # parse_options()

    run()
