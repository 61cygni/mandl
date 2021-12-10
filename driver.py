# --
# File: driver.py
#
# Simple pygame driver of main fractal program 
#
#
# --

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


class TextRectException:
    def __init__(self, message=None):
            self.message = message

    def __str__(self):
        return self.message


def multiLineSurface(string: str, font: pygame.font.Font, rect: pygame.rect.Rect, fontColour: tuple, BGColour: tuple, justification=0):
    """Returns a surface containing the passed text string, reformatted
    to fit within the given rect, word-wrapping as necessary. The text
    will be anti-aliased.

    Parameters
    ----------
    string - the text you wish to render. \n begins a new line.
    font - a Font object
    rect - a rect style giving the size of the surface requested.
    fontColour - a three-byte tuple of the rgb value of the
             text color. ex (0, 0, 0) = BLACK
    BGColour - a three-byte tuple of the rgb value of the surface.
    justification - 0 (default) left-justified
                1 horizontally centered
                2 right-justified

    Returns
    -------
    Success - a surface object with the text rendered onto it.
    Failure - raises a TextRectException if the text won't fit onto the surface.
    """

    finalLines = []
    requestedLines = string.splitlines()
    # Create a series of lines that will fit on the provided
    # rectangle.
    for requestedLine in requestedLines:
        if font.size(requestedLine)[0] > rect.width:
            words = requestedLine.split(' ')
            # if any of our words are too long to fit, return.
            for word in words:
                if font.size(word)[0] >= rect.width:
                    raise TextRectException("The word " + word + " is too long to fit in the rect passed.")
            # Start a new line
            accumulatedLine = ""
            for word in words:
                testLine = accumulatedLine + word + " "
                # Build the line while the words fit.
                if font.size(testLine)[0] < rect.width:
                    accumulatedLine = testLine
                else:
                    finalLines.append(accumulatedLine)
                    accumulatedLine = word + " "
            finalLines.append(accumulatedLine)
        else:
            finalLines.append(requestedLine)

    # Let's try to write the text out on the surface.
    surface = pygame.Surface(rect.size)
    surface.fill(BGColour)
    accumulatedHeight = 0
    for line in finalLines:
        if accumulatedHeight + font.size(line)[1] >= rect.height:
             raise TextRectException("Once word-wrapped, the text string was too tall to fit in the rect.")
        if line != "":
            tempSurface = font.render(line, 1, fontColour)
        if justification == 0:
            surface.blit(tempSurface, (0, accumulatedHeight))
        elif justification == 1:
            surface.blit(tempSurface, ((rect.width - tempSurface.get_width()) / 2, accumulatedHeight))
        elif justification == 2:
            surface.blit(tempSurface, (rect.width - tempSurface.get_width(), accumulatedHeight))
        else:
            raise TextRectException("Invalid justification argument: " + str(justification))
        accumulatedHeight += font.size(line)[1]
    return surface

def display_hud (screen):

    myfont = pygame.font.SysFont('courier new', 12)
    text_str = """\
Once upon a time
There was a large text string
that I was going to use"""

    textsurface = multiLineSurface(text_str,myfont, pygame.Rect(50,50,128,128) , (0,255,0), (0,0,0)) 
    #textsurface = myfont.render(text_str, False, (0, 255, 0))
    #rect        = textsurface.get_rect()
    #pygame.draw.rect(textsurface, (0,0,0), rect, 1)
    screen.blit(textsurface,(50,50))


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

        display_hud(screen)

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
