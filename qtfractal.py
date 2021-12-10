# --
# File : qtfractal.py
# Date: Sun Dec 05 22:53:44 PST 2021 
#
# TODO:
# - Add support for max-iter
# - Get colors working
# - Fix mandeldistance
# - add a rollback button to get to the last picture
# - increase number of sample points per pixel from UI
#
# Next Up:
# - Clean up entire execution path through csmooth
##
# --
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys, os
import subprocess

from decimal import *

hpf = Decimal
getcontext().prec = 500 

DEFAULT_ALGO = "ldnative"
BURN = True 

# Fixed with to display fractal image
DISPLAY_WIDTH   = 640.
DISPLAY_HEIGHT  = 480.

# runtime configurable paremeters

algo = DEFAULT_ALGO

image_w = DISPLAY_WIDTH
image_h = DISPLAY_HEIGHT 

red   = 0.1
green = 0.2
blue  = 0.3

real = hpf(-.745)
imag = hpf(.186)
#real = hpf(-1)
#imag = hpf(0)

c_width  = hpf(5)
c_height = hpf(0)

# number of samples per pixel
samples = 9


def run():
    global real
    global imag
    global samples
    global c_width
    global c_height
    global image_w
    global image_h
    global BURN

    burn_str = ""
    if BURN:
        burn_str = "--burn"

    #cmd = "python3 fractal.py %s --verbose=3 --algo=%s --setcolor='(%f,%f,%f)' --cmplx-w=%s --cmplx-h=%s --img-w=%d --img-h=%d --real=\"%s\" --imag=\"%s\" " \
    #      %(burn_str, str(algo), red, green, blue, str(c_width), str(c_height),image_w,image_h,str(real),str(imag))
    cmd = "python3 fractal.py %s --verbose=3 --algo=%s --sample=%d --setcolor='(%f,%f,%f)' --cmplx-w=%s --cmplx-h=%s --img-w=%d --img-h=%d --real=\"%s\" --imag=\"%s\" " \
          %(burn_str, str(algo), samples, red, green, blue, str(c_width), str(c_height),image_w,image_h,str(real),str(imag))
    print(" + Driver running comment: "+cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()

    #real, imag = display()


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
        
    def event(self, event):
        if event.type() == QEvent.HoverMove:
            x = event.pos().x() 
            y = event.pos().y() 
            status = 'Image (%f,%f)'%(x,y)
            self.parent.parent.statusBar().showMessage(status)

        return super().event(event)


    def paintEvent(self, event):
        super().paintEvent(event)
        qp = QPainter(self)
        br = QBrush(QColor(100, 10, 10, 40))  
        qp.setBrush(br)   
        nrect = QRect(self.begin, self.end)
        qp.drawRect(nrect)


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
        global DISPLAY_WIDTH
        global DISPLAY_HEIGHT
        global image_w
        global image_h

        nrect = QRect(self.begin, self.end)

        x = nrect.center().x()
        y = nrect.center().y()

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
        
        # zoom in
        c_width  =  hpf(float(nrect.width()) / DISPLAY_HEIGHT)  * c_width
        c_height =  hpf(float(nrect.height()) / DISPLAY_HEIGHT) * c_height

        run()
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
        self.main_image_name="./pyfractal.gif"
        self.mode = 5

        self.initUI()

    def sync_config_from_ui(self):
        global samples
        global image_w
        global image_h
        global algo
        global red
        global blue 
        global green 


        #set values from UI
        samples = int(self.samples_text.text())
        image_w = int(float(self.img_width_text.text()))
        image_h = int(float(self.img_height_text.text())) 
        algo    = self.algo_combo.currentText()
        red     = float(self.red_edit.text())
        green   = float(self.green_edit.text()) 
        blue    = float(self.blue_edit.text()) 

    def refresh_ui(self):
        global image_w
        global image_h
        global real
        global imag
        global samples
        global c_height
        global c_width
        global red
        global green
        global blue

        c_height    = c_width * (hpf(DISPLAY_HEIGHT) / hpf(DISPLAY_WIDTH))

        self.c_width_text.setPlainText(str(c_width))
        self.c_height_text.setPlainText(str(c_height))
        self.img_width_text.setText(str(image_w))
        self.img_height_text.setText(str(image_h))
        self.c_real_edit.setPlainText(str(real))
        self.c_imag_edit.setPlainText(str(imag))
        self.samples_text.setText(str(samples))

        self.red_edit.setText(str(red))
        self.green_edit.setText(str(green))
        self.blue_edit.setText(str(blue))

        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(self.main_image_name)
        pixmap = pixmap.scaled(DISPLAY_WIDTH, DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.grid.addWidget(self.main_image, 0, 1)

    def run(self):
        self.sync_config_from_ui()
        run()
        self.refresh_ui()
        

    def initUI(self):

        #QToolTip.setFont(QFont('SansSerif', 10))
        #QMainWindow.statusBar().showMessage('Ready')

        self.setGeometry(300, 300, 250, 150)
        self.resize(640, 480)
        self.center()

        self.main_image = FractalImgQLabel(self)
        self.main_image.setPixmap(QPixmap(self.main_image_name))

        #btn = QPushButton("Make setup file")
        #btn.setToolTip('Press <b>Detect</b> button for detecting objects by your settings')
        #btn.resize(btn.sizeHint())
        #btn.clicked.connect(QCoreApplication.instance().quit)

        btn_run = QPushButton("run")
        btn_run.clicked.connect(self.run)
        #btn_set = QPushButton("Set name")

        #fullscreen
        #self.main_image.setScaledContents(True)
        #just centered
        self.main_image.setAlignment(Qt.AlignCenter)

        # Allow user to specify the complex width


        # Basic config

        algo_label = QLabel('Fractal Algo')
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("ldnative")
        self.algo_combo.addItem("hpnative")
        self.algo_combo.addItem("mandeldistance")
        self.algo_combo.addItem("csmooth")

        c_width_label   = QLabel('Complex width')
        c_height_label = QLabel('Complex height')

        self.c_width_text   = QPlainTextEdit(self)
        self.c_height_text  = QPlainTextEdit(self)

        img_width_label = QLabel('Image width')
        img_heigh_label = QLabel('Image height')

        self.img_width_text  = QLineEdit(self)
        self.img_height_text  = QLineEdit(self)

        samples_label     = QLabel("Samples: ")
        self.samples_text = QLineEdit(self) 

        red_label   = QLabel("Red: ")
        self.red_edit    = QLineEdit(self) 
        green_label = QLabel("Green: ")
        self.green_edit  = QLineEdit(self) 
        blue_label  = QLabel("Blue: ")
        self.blue_edit   = QLineEdit(self) 

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
        grid_config.addWidget(img_heigh_label,4, 0)
        grid_config.addWidget(self.img_height_text, 4, 1)
        grid_config.addWidget(samples_label ,5, 0)
        grid_config.addWidget(self.samples_text, 5, 1)
        grid_config.addWidget(red_label ,6, 0)
        grid_config.addWidget(self.red_edit, 6, 1)
        grid_config.addWidget(green_label ,7, 0)
        grid_config.addWidget(self.green_edit, 7, 1)
        grid_config.addWidget(blue_label ,8, 0)
        grid_config.addWidget(self.blue_edit, 8, 1)
        grid_config.addWidget(btn_run, 9, 0)

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


        run()
        self.refresh_ui()

        self.show()

    def browse(self):
        w = QWidget()
        w.resize(320, 240)
        w.setWindowTitle("Select Picture")
        filename = QFileDialog.getOpenFileName(w, 'Open File', '/')
        self.main_image_name = filename
        #self.main_image.setPixmap(QPixmap(self.main_image_name))
        self.main_image.setPixmap(QPixmap('./pyfractal.gif'))

    def center(self):

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class menubarex(QMainWindow):
    def __init__(self, parent=None):
        super(menubarex, self).__init__(parent)
        self.form_widget = QTFractalMainWindow(self)
        self.setCentralWidget(self.form_widget)

        self.initUI()

    def initUI(self):
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        #self.toolbar = self.addToolBar('Exit')
        #self.toolbar.addAction(exitAction)

        self.statusBar().showMessage('Ready')
        self.setWindowTitle('QTFractal')
        self.setWindowIcon(QIcon('icon.png'))

    def closeEvent(self, event):
        event.accept()

        #reply = QMessageBox.question(self, 'Message',
        #    "Are you sure to quit?", QMessageBox.Yes |
        #    QMessageBox.No)

        #if reply == QMessageBox.Yes:
        #    event.accept()
        #else:
        #    event.ignore()

def main():
    app = QApplication(sys.argv)
    #ex = QTFractalMainWindow()
    menubar = menubarex()
    menubar.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
