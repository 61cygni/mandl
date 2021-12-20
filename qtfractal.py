# --
# File : qtfractal.py
# Date: Sun Dec 05 22:53:44 PST 2021 
#
# Usage:
#
# - Click and drag mouse to create zoom box
# - Hold shift while creating zoom box to drag it around the screen
# - Use Command+Mouse Click to center the picture at the current zoom
#   level at the click location
#
# TODO:
# 
# - get julia sets plugged into UI
# - implement sampling with julia sets
# - clean up status / debug printing end to end
# - add a rollback button to get to the last picture
# - write raw calculations from C to binary file
#
# Done :
#
# - Fix mandeldistance
# - Add support for max-iter
# - Get colors working
# - increase number of sample points per pixel from UI
# - center on current mouse location picture if mouse pressed some key
#   held
#
# --


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys, os
import subprocess

from decimal import *

hpf = Decimal
getcontext().prec = 500 

BURN = False  
DEFAULT_ALGO = "ldnative"

splash_image_name = './images/qtfractal_splash.gif'
main_image_name   = './pyfractal.gif'

# Fixed with to display fractal image
DISPLAY_WIDTH   = 640
DISPLAY_HEIGHT  = 480

# runtime configurable paremeters

algo = DEFAULT_ALGO

image_w = DISPLAY_WIDTH
image_h = DISPLAY_HEIGHT 

red   = 0.1
green = 0.2
blue  = 0.6

real = hpf(-.745)
imag = hpf(.186)

c_width  = hpf(5)
c_height = hpf(0)

samples  = 17  # number of samples per pixel
max_iter = (2 << 10)  
julia_c = -.8+.156j


# --
# Run the command line fractal program. 
# --

def run(filename):
    global real
    global imag
    global samples
    global max_iter
    global c_width
    global c_height
    global image_w
    global image_h
    global BURN
    global julia_c

    burn_str = ""
    if BURN:
        burn_str = "--burn"

    julia_str = ""
    if str(algo) == 'julia':    
        julia_str = format('--julia-c="%s"'%(str(julia_c)))

    cmd = "python3 fractal.py %s --verbose=3 --algo=%s %s --sample=%d --max-iter=%d --setcolor='(%f,%f,%f)' --cmplx-w=%s --cmplx-h=%s --img-w=%d --img-h=%d --real=\"%s\" --imag=\"%s\" --gif=%s" \
          %(burn_str, str(algo), julia_str, samples, max_iter, red, green, blue, str(c_width), str(c_height),image_w,image_h,str(real),str(imag),filename)
    print(" + Driver running comment: "+cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

# --
# Popup that we use to generate a large snapshot
# --

class SnapshotPopup(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.filename = './snapshot.gif'
        self.initUI()

    def run(self):
        self.sync_config_from_ui()
        run(self.filename)
        self.refresh_ui()

    def refresh_ui(self):
        global image_w
        global image_h
        global real
        global imag
        global samples
        global max_iter
        global c_height
        global c_width
        global red
        global green
        global blue

        self.filename_text.setText(self.filename)
        self.img_width_text.setText(str(image_w))
        self.img_height_text.setText(str(image_h))
        self.samples_text.setText(str(samples))
        self.iter_text.setText(str(max_iter))

        self.red_edit.setText(str(red))
        self.green_edit.setText(str(green))
        self.blue_edit.setText(str(blue))

        self.julia_c_edit.setText(str(julia_c))

    def sync_config_from_ui(self):
        global samples
        global max_iter
        global image_w
        global image_h
        global algo
        global red
        global blue 
        global green 
        global julia_c 


        #set values from UI
        algo    = self.algo_combo.currentText()
        samples  = int(self.samples_text.text())
        max_iter = int(self.iter_text.text())
        image_w = int(float(self.img_width_text.text()))
        image_h = int(float(self.img_height_text.text())) 
        red     = float(self.red_edit.text())
        green   = float(self.green_edit.text()) 
        blue    = float(self.blue_edit.text()) 
        julia_c = complex(self.julia_c_edit.text()) 


    def set_res(self, event):
        res = self.res_combo.currentText()
        if res == '1k':
            self.img_width_text.setText("1024")
            self.img_height_text.setText("768")
        elif res == '2k':
            self.img_width_text.setText("2048")
            self.img_height_text.setText("1536")
        elif res == '4k':
            self.img_width_text.setText("3840")
            self.img_height_text.setText("2160")
        elif res == '8k':
            self.img_width_text.setText("7680")
            self.img_height_text.setText("4320")
        elif res == '12k':
            self.img_width_text.setText("12288")
            self.img_height_text.setText("6480")
        elif res == '16k':
            self.img_width_text.setText("15360")
            self.img_height_text.setText("8640")
        else:
            print(" * Error ... unknown resultion "+res)

        self.update()    

    def initUI(self):

        filename_label      = QLabel('Filename')
        self.filename_text  = QLineEdit(self)

        algo_label = QLabel('Fractal Algo')
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("ldnative")
        self.algo_combo.addItem("hpnative")
        self.algo_combo.addItem("mandeldistance")
        self.algo_combo.addItem("csmooth")
        self.algo_combo.addItem("julia")
        self.algo_combo.addItem("cjulia")

        self.res_combo = QComboBox()
        self.res_combo.addItem('1k')
        self.res_combo.addItem('2k')
        self.res_combo.addItem('4k')
        self.res_combo.addItem('8k')
        self.res_combo.addItem('12k')
        self.res_combo.addItem('16k')

        # adding action to combo box
        self.res_combo.activated.connect(self.set_res)

        img_width_label = QLabel('Image width')
        img_height_label = QLabel('Image height')

        self.img_width_text  = QLineEdit(self)
        self.img_height_text  = QLineEdit(self)

        samples_label     = QLabel("Samples: ")
        self.samples_text = QLineEdit(self) 

        iter_label     = QLabel("Max iter: ")
        self.iter_text = QLineEdit(self) 

        red_label   = QLabel("Red: ")
        self.red_edit    = QLineEdit(self) 
        green_label = QLabel("Green: ")
        self.green_edit  = QLineEdit(self) 
        blue_label  = QLabel("Blue: ")
        self.blue_edit   = QLineEdit(self) 

        julia_c_label = QLabel("Julia c: ")
        self.julia_c_edit = QLineEdit(self)



        btn_run = QPushButton("Go!")
        btn_run.clicked.connect(self.run)

        res_config = QGridLayout()
        res_config.addWidget(self.img_width_text, 0, 1)
        res_config.addWidget(self.res_combo,      0, 2)

        self.grid_config = QGridLayout()
        self.grid_config.addWidget(filename_label,0, 0)
        self.grid_config.addWidget(self.filename_text, 0, 1)
        self.grid_config.addWidget(algo_label ,1, 0)
        self.grid_config.addWidget(self.algo_combo ,1, 1)
        self.grid_config.addLayout(res_config, 2,1)
        self.grid_config.addWidget(self.res_combo, 2,1)
        self.grid_config.addWidget(img_width_label,2, 0)
        self.grid_config.addWidget(img_height_label,3, 0)
        self.grid_config.addWidget(self.img_height_text, 3, 1)
        self.grid_config.addWidget(samples_label ,5, 0)
        self.grid_config.addWidget(self.samples_text, 5, 1)
        self.grid_config.addWidget(iter_label ,6, 0)
        self.grid_config.addWidget(self.iter_text, 6, 1)
        self.grid_config.addWidget(red_label ,7, 0)
        self.grid_config.addWidget(self.red_edit, 7, 1)
        self.grid_config.addWidget(green_label ,8, 0)
        self.grid_config.addWidget(self.green_edit, 8, 1)
        self.grid_config.addWidget(blue_label ,9, 0)
        self.grid_config.addWidget(self.blue_edit, 9, 1)
        self.grid_config.addWidget(julia_c_label ,10, 0)
        self.grid_config.addWidget(self.julia_c_edit, 10, 1)

        self.grid_config.addWidget(btn_run, 11, 1)

        self.setLayout(self.grid_config)
        self.refresh_ui()
        self.update()


# --
# Main image widget. This loads in the fractal snapshot after it was
# generated and responds to mouse events
# --

class FractalImgQLabel(QLabel):
    def __init__(self, parent=None):
        super(FractalImgQLabel, self).__init__(parent)
        self.setAttribute(Qt.WA_Hover)
        self.parent = parent
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setSizePolicy(sizePolicy)
        self.begin = QPoint()
        self.end   = QPoint()
        self.pos   = QPoint()
        
    def event(self, event):

        if event.type() == QEvent.HoverMove:
            x = event.pos().x() 
            y = event.pos().y() 
            status = 'Image (%f,%f)'%(x,y)
            self.parent.parent.statusBar().showMessage(status)

            self.pos = event.pos()
        
            self.update()


        return super().event(event)


    def paintEvent(self, event):
        super().paintEvent(event)

        qp = QPainter(self)
        br = QBrush(QColor(100, 10, 10, 40))  
        qp.setBrush(br)   
        nrect = QRect(self.begin, self.end)
        qp.drawRect(nrect)

        # draw crosshairs
        qp.setPen(QColor(128, 0, 64, 127))
        qp.drawLine(0,self.pos.y(),DISPLAY_WIDTH,self.pos.y())
        qp.drawLine(self.pos.x(),0,self.pos.x(), DISPLAY_HEIGHT)


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            print("escape")


    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):

        # if shift is held down, then move the rectangel as is
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            rect = QRect(self.begin, self.end)
            rect.moveCenter(event.pos())
            self.begin = rect.topLeft()
            self.end   = rect.bottomRight()
        else:
            self.end = event.pos()
            # make sure rectangle keeps aspect ration of display
            nrect = QRect(self.begin, self.end)
            rw = nrect.width()
            rh = int(float(rw) * (DISPLAY_HEIGHT / DISPLAY_WIDTH)) 
            nrect.setWidth(rw)
            nrect.setHeight(rh)
            self.end = nrect.bottomRight()

        self.update()

    def mouseReleaseEvent(self, event):
        global real
        global imag
        global c_width
        global c_height
        global ALGO
        global BURN
        global samples
        global max_iter
        global DISPLAY_WIDTH
        global DISPLAY_HEIGHT
        global image_w
        global image_h
        global main_image_name

        nrect = QRect(self.begin, self.end)

        size = nrect.size()

        x = nrect.center().x()
        y = nrect.center().y()

        # on click with no bounding box, check for control modifier and
        # if pressed, use that to center the picture around the mouse
        if size.height() * size.width() < 4:
            modifiers = QApplication.keyboardModifiers()
            if modifiers == Qt.ControlModifier:
                x = event.pos().x()
                y = event.pos().y()
                print(" + Centering picture")

            else:    
                self.update()
                return


        self.parent.sync_config_from_ui()

        # Use the center the calculate the edges
        re_start = real - (c_width  / hpf(2.))
        im_start = imag - (c_height / hpf(2.))

        fxoffset = float(x)/DISPLAY_WIDTH
        fyoffset = float(y)/DISPLAY_HEIGHT

        real = re_start + (hpf(fxoffset) * c_width)
        imag = im_start + (hpf(fyoffset) * c_height)

        print("x %d, y %d"%(x,y))
        print("fxoff %f, fyoff %f"%(fxoffset,fyoffset))
        print("Real %s, Image %s"%(str(real),str(imag)))
        
        # only zoom in if there is a bounding box 
        if size.height() * size.width() >= 4:
            c_width  =  hpf(float(nrect.width()) / DISPLAY_HEIGHT)  * c_width
            c_height =  hpf(float(nrect.height()) / DISPLAY_HEIGHT) * c_height

        run(main_image_name)

        self.parent.refresh_ui()

        self.begin = event.pos()
        self.end = event.pos()
        self.update()
        

# --
# Main Window
# --

class QTFractalMainWindow(QWidget):

    def __init__(self,parent):
        super(QTFractalMainWindow, self).__init__()
        self.parent = parent
        self.mode = 5

        self.initUI()

    def sync_config_from_ui(self):
        global samples
        global max_iter
        global image_w
        global image_h
        global algo
        global red
        global blue 
        global green 
        global julia_c


        #set values from UI
        samples  = int(self.samples_text.text())
        max_iter = int(self.iter_text.text())
        image_w = int(float(self.img_width_text.text()))
        image_h = int(float(self.img_height_text.text())) 
        algo    = self.algo_combo.currentText()
        red     = float(self.red_edit.text())
        green   = float(self.green_edit.text()) 
        blue    = float(self.blue_edit.text()) 
        julia_c = complex(self.julia_c_edit.text())

    def refresh_ui(self):
        global image_w
        global image_h
        global real
        global imag
        global samples
        global max_iter
        global c_height
        global c_width
        global red
        global green
        global blue
        global main_image_name
        global julia_c

        c_height    = c_width * (hpf(DISPLAY_HEIGHT) / hpf(DISPLAY_WIDTH))

        self.c_width_text.setPlainText(str(c_width))
        self.c_height_text.setPlainText(str(c_height))
        self.img_width_text.setText(str(image_w))
        self.img_height_text.setText(str(image_h))
        self.c_real_edit.setPlainText(str(real))
        self.c_imag_edit.setPlainText(str(imag))
        self.samples_text.setText(str(samples))
        self.iter_text.setText(str(max_iter))

        self.red_edit.setText(str(red))
        self.green_edit.setText(str(green))
        self.blue_edit.setText(str(blue))
        self.julia_c_edit.setText(str(julia_c))


        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(main_image_name)
        pixmap = pixmap.scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.main_image.resize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.grid.addWidget(self.main_image, 0, 1)

    def run(self):
        global main_image_name

        self.sync_config_from_ui()
        run(main_image_name)
        self.refresh_ui()

    def snapshot(self):
        self.snap_pop = SnapshotPopup()
        self.snap_pop.setGeometry(QRect(100, 100, 400, 200))
        self.snap_pop.show()
        

    def initUI(self):
        global splash_image_name

        #QToolTip.setFont(QFont('SansSerif', 10))
        #QMainWindow.statusBar().showMessage('Ready')

        self.setGeometry(300, 300, 250, 150)
        self.resize(640, 480)
        self.center()

        self.main_image = FractalImgQLabel(self)
        self.main_image.setPixmap(QPixmap(splash_image_name))

        btn_run = QPushButton("run")
        btn_run.clicked.connect(self.run)

        btn_snapshot = QPushButton("snapshot")
        btn_snapshot.clicked.connect(self.snapshot)

        self.main_image.setAlignment(Qt.AlignCenter)

        # Basic config

        algo_label = QLabel('Fractal Algo')
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("ldnative")
        self.algo_combo.addItem("hpnative")
        self.algo_combo.addItem("mandeldistance")
        self.algo_combo.addItem("csmooth")
        self.algo_combo.addItem("julia")
        self.algo_combo.addItem("cjulia")

        c_width_label   = QLabel('Complex width')
        c_height_label = QLabel('Complex height')

        self.c_width_text   = QPlainTextEdit(self)
        self.c_height_text  = QPlainTextEdit(self)

        img_width_label = QLabel('Image width')
        img_height_label = QLabel('Image height')

        self.img_width_text  = QLineEdit(self)
        self.img_height_text  = QLineEdit(self)

        samples_label     = QLabel("Samples: ")
        self.samples_text = QLineEdit(self) 

        iter_label     = QLabel("Max iter: ")
        self.iter_text = QLineEdit(self) 



        red_label   = QLabel("Red: ")
        self.red_edit    = QLineEdit(self) 
        green_label = QLabel("Green: ")
        self.green_edit  = QLineEdit(self) 
        blue_label  = QLabel("Blue: ")
        self.blue_edit   = QLineEdit(self) 

        julia_c_label      = QLabel("Julia c: ")
        self.julia_c_edit  = QLineEdit(self) 


        c_real_label = QLabel("Center Real:") 
        self.c_real_edit  = QPlainTextEdit()
        c_imag_label = QLabel("Center Imaginary:") 
        self.c_imag_edit  = QPlainTextEdit()

        
        # Left side config params
        grid_config = QGridLayout()
        grid_config.addWidget(algo_label ,0, 0)
        grid_config.addWidget(self.algo_combo ,0, 1)
        grid_config.addWidget(c_width_label,1, 0)
        grid_config.addWidget(self.c_width_text, 1, 1)
        grid_config.addWidget(c_height_label,2, 0)
        grid_config.addWidget(self.c_height_text, 2, 1)
        grid_config.addWidget(img_width_label,3, 0)
        grid_config.addWidget(self.img_width_text, 3, 1)
        grid_config.addWidget(img_height_label,4, 0)
        grid_config.addWidget(self.img_height_text, 4, 1)
        grid_config.addWidget(samples_label ,5, 0)
        grid_config.addWidget(self.samples_text, 5, 1)
        grid_config.addWidget(iter_label ,6, 0)
        grid_config.addWidget(self.iter_text, 6, 1)
        grid_config.addWidget(red_label ,7, 0)
        grid_config.addWidget(self.red_edit, 7, 1)
        grid_config.addWidget(green_label ,8, 0)
        grid_config.addWidget(self.green_edit, 8, 1)
        grid_config.addWidget(blue_label ,9, 0)
        grid_config.addWidget(self.blue_edit, 9, 1)
        grid_config.addWidget(julia_c_label ,10, 0)
        grid_config.addWidget(self.julia_c_edit, 10, 1)

        grid_config.addWidget(btn_run, 11, 0)
        grid_config.addWidget(btn_snapshot, 11, 1)

        # Right side inputs for c_real and c_imag 
        grid_center = QGridLayout()
        grid_center.addWidget(c_real_label ,0, 0)
        grid_center.addWidget(self.c_real_edit  ,1, 0)
        grid_center.addWidget(c_imag_label ,2, 0)
        grid_center.addWidget(self.c_imag_edit  ,3, 0)


        # MAIN GRID

        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        self.grid.addLayout(grid_config,     0, 0)
        self.grid.addWidget(self.main_image, 0, 1)
        self.grid.addLayout(grid_center,     0, 2)

        self.setLayout(self.grid)

        self.refresh_ui()

        # use splash screen to start
        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(splash_image_name)
        pixmap = pixmap.scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.main_image.resize(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.grid.addWidget(self.main_image, 0, 1)

        self.show()

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class ParentWindow(QMainWindow):

    def __init__(self, parent=None):
        super(ParentWindow, self).__init__(parent)
        self.fractalmain = QTFractalMainWindow(self)
        self.setCentralWidget(self.fractalmain)

def main():
    app = QApplication(sys.argv)
    parent = ParentWindow()
    parent.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
