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
# - ability to save current config (should be easy now that we have
#   context)
# - clean up status / debug printing end to end
# - add a rollback button to get to the last picture
# - write raw calculations from C to binary file
#
# Done :
#
# - preview radio button for snapshot window 
# - add a "reset" button to get to the initial config
# - get julia sets plugged into UI
# - implement sampling with julia sets
# - Fix mandeldistance
# - Add support for max-iter
# - Get colors working
# - increase number of sample points per pixel from UI
# - center on current mouse location picture if mouse pressed some key
#   held
#
# Really nice colors : 0, .19, 1.0
#
# --

import sys, os
import subprocess
import pickle
from dataclasses import dataclass


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


from decimal import *

hpf = Decimal
getcontext().prec = 500 

BURN = False  
DEFAULT_ALGO = "ldnative"

splash_image_name = './images/qtfractal_splash.gif'
main_image_name   = './pyfractal.gif'

# Fixed with to display fractal image
DEFAULT_DISPLAY_WIDTH   = 640
DEFAULT_DISPLAY_HEIGHT  = 480

# runtime configurable paremeters

DEFAULT_RED   = 0.1
DEFAULT_GREEN = 0.2
DEFAULT_BLUE  = 0.6

DEFAULT_REAL = hpf(-.745)
DEFAULT_IMAG = hpf(.186)

DEFAULT_C_WIDTH  = hpf(5)
DEFAULT_C_HEIGHT = hpf(3.75)

DEFAULT_SAMPLES  = 17  # number of samples per pixel
DEFAULT_MAX_ITER = (2 << 10)  
DEFAULT_JULIA_C = -.8+.156j

DEFAULT_SAVEDIR = "./savedfiles/"

def FileDialog(directory='', forOpen=True, fmt='', isFolder=False):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)

    dialog.setFilter(dialog.filter() | QDir.Hidden)

    # ARE WE TALKING ABOUT FILES OR FOLDERS
    if isFolder:
        dialog.setFileMode(QFileDialog.DirectoryOnly)
    else:
        dialog.setFileMode(QFileDialog.AnyFile)
    # OPENING OR SAVING
    dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

    # SET FORMAT, IF SPECIFIED
    if fmt != '' and isFolder is False:
        dialog.setDefaultSuffix(fmt)
        dialog.setNameFilters([f'{fmt} (*.{fmt})'])

    # SET THE STARTING DIRECTORY
    if directory != '':
        dialog.setDirectory(str(directory))
    else:
        dialog.setDirectory(DEFAULT_SAVEDIR)


    if dialog.exec_() == QDialog.Accepted:
        path = dialog.selectedFiles()[0]  # returns a list
        return path
    else:
        return ''


# --
# global::RunContext
# --

class RunContext:

    def __init__(self):
        self.algo     = "" 
        self.image_w  = DEFAULT_DISPLAY_WIDTH 
        self.image_h  = DEFAULT_DISPLAY_HEIGHT
        self.red      = 0. 
        self.green    = 0. 
        self.blue     = 0.
        self.samples  = 0
        self.max_iter = 0
        self.julia_c  = complex(0)
        self.c_width  = hpf(0)
        self.c_height = hpf(0)
        self.c_real   = hpf(0)
        self.c_imag   = hpf(0)


# --
# global::run
# Run the command line fractal program. 
# --

def run(fn, context):

    al = context.algo
    jc = context.julia_c
    sm = context.samples
    r  = context.red
    g  = context.green
    b  = context.blue
    cw = str(context.c_width)
    ch = str(context.c_height)
    iw = context.image_w
    ih = context.image_h
    mi = context.max_iter
    cr = str(context.c_real)
    ci = str(context.c_imag)

    bstr = ""
    if BURN:
        bstr = "--burn"

    jstr = ""
    if str(al) == 'julia' or str(al) == 'cjulia':    
        jstr = format('--julia-c="%s"'%(str(jc)))

    cmd = "python3 fractal.py %s --verbose=3 --algo=%s %s --sample=%d --max-iter=%d --setcolor='(%f,%f,%f)' --cmplx-w=%s --cmplx-h=%s --img-w=%d --img-h=%d --real=\"%s\" --imag=\"%s\" --gif=%s" \
          %(bstr, al, jstr, sm, mi, r, g, b, cw, ch,iw,ih,cr,ci,fn)
    print(" + Driver running comment: "+cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


class DisplaySnap(QWidget):

    def __init__(self, filename):
        super().__init__()
        self.title = 'QTFractal Snapshot'
        self.filename = filename
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
    
        # Create widget
        label = QLabel(self)
        pixmap = QPixmap(self.filename)
        label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())
        
        self.show()


# --
# Popup that we use to generate a large snapshot
# --

class SnapshotPopup(QWidget):

    def __init__(self, parent, context):
        QWidget.__init__(self)

        self.parent  = parent
        self.filename = './snapshot.gif'
        self.init_ui(context)

    # --    
    # SnapshotPopup::run
    # --    

    def run(self):
        context = self.sync_config_from_ui()
        run(self.filename, context)
        if self.display_check.isChecked():
            self.ds = DisplaySnap(self.filename)


    # --    
    # SnapshotPopup::refresh_ui
    # --    

    def set_ui_defaults(self, context):

        self.filename_text.setText(self.filename)
        self.img_width_text.setText(str(context.image_w))
        self.img_height_text.setText(str(context.image_h))
        self.samples_text.setText(str(context.samples))
        self.iter_text.setText(str(context.max_iter))

        self.red_edit.setText(str(context.red))
        self.green_edit.setText(str(context.green))
        self.blue_edit.setText(str(context.blue))

        self.res_combo.setCurrentText("2k")
        self.img_width_text.setText("2048")
        self.img_height_text.setText("1536")
        self.samples_text.setText("65")
        self.algo_combo.setCurrentText(self.parent.algo_combo.currentText())

        self.set_algo(None) # force setting julia-c

    # --    
    # SnapshotPopup::sync_config_from_ui
    # --    

    def sync_config_from_ui(self):

        ctx = RunContext()


        #set values from UI
        ctx.algo    = self.algo_combo.currentText()
        ctx.samples  = int(self.samples_text.text())
        ctx.max_iter = int(self.iter_text.text())
        ctx.image_w = int(float(self.img_width_text.text()))
        ctx.image_h = int(float(self.img_height_text.text())) 
        ctx.red     = float(self.red_edit.text())
        ctx.green   = float(self.green_edit.text()) 
        ctx.blue    = float(self.blue_edit.text()) 
        if self.julia_c_edit:
            ctx.julia_c = complex(self.julia_c_edit.text()) 
        else:
            ctx.julia_c = None

        ctx.c_width  = hpf(self.parent.c_width_text.toPlainText())
        ctx.c_height = hpf(self.parent.c_height_text.toPlainText())
        ctx.c_real = hpf(self.parent.c_real_edit.toPlainText())
        ctx.c_imag = hpf(self.parent.c_imag_edit.toPlainText())

        return ctx 

    # --    
    # SnapshotPopup::set_res
    # --    

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

    # --    
    # SnapshotPopup::set_algo
    # --    

    def set_algo(self, event):
        res = self.algo_combo.currentText()

        if res == 'julia' or res == 'cjulia':
            if not self.julia_c_edit:
                self.julia_c_label = QLabel("Julia c: ")
                self.julia_c_edit  = QLineEdit(self) 
                self.grid_config.addWidget(self.julia_c_label, 10, 0)
                self.grid_config.addWidget(self.julia_c_edit,  10, 1)
                if self.parent.julia_c_edit:
                    self.julia_c_edit.setText(self.parent.julia_c_edit.text())
                else:    
                    self.julia_c_edit.setText(str(DEFAULT_JULIA_C))
        else:
            if self.julia_c_edit:
                self.grid_config.removeWidget(self.julia_c_label)
                self.grid_config.removeWidget(self.julia_c_edit)
                self.julia_c_label.deleteLater()
                self.julia_c_edit.deleteLater()
                self.julia_c_edit  = None 
                self.julia_c_label = None 

        #self.refresh_ui()
        #self.sync_config_from_ui()
        self.update()    

    # --    
    # SnapshotPopup::init_ui
    # --    

    def init_ui(self, context):

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

        self.algo_combo.activated.connect(self.set_algo)

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

        self.julia_c_label = None
        self.julia_c_edit = None


        btn_run = QPushButton("Go!")
        btn_run.clicked.connect(self.run)

        display_label      = QLabel("display")
        self.display_check = QCheckBox()

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


        dlayout = QHBoxLayout()
        dlayout.addWidget(display_label)
        dlayout.addWidget(self.display_check)
        self.grid_config.addLayout(dlayout, 11, 0)

        self.grid_config.addWidget(btn_run, 11, 1)


        self.setLayout(self.grid_config)

        context = self.parent.sync_config_from_ui()
        self.set_ui_defaults(context)

        # add some snapshot specific defaults here


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

    # --
    # FractalImgQLabel::event
    #
    # On mouse hover over main image, update status bar
    # --
        
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
        qp.drawLine(0,self.pos.y(),DEFAULT_DISPLAY_WIDTH,self.pos.y())
        qp.drawLine(self.pos.x(),0,self.pos.x(), DEFAULT_DISPLAY_HEIGHT)


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
            rh = int(float(rw) * (DEFAULT_DISPLAY_HEIGHT / DEFAULT_DISPLAY_WIDTH)) 
            nrect.setWidth(rw)
            nrect.setHeight(rh)
            self.end = nrect.bottomRight()

        self.update()

    def mouseReleaseEvent(self, event):

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


        ctx = self.parent.sync_config_from_ui()

        # Use the center the calculate the edges
        re_start = ctx.c_real - (ctx.c_width  / hpf(2.))
        im_start = ctx.c_imag - (ctx.c_height / hpf(2.))

        fxoffset = float(x)/DEFAULT_DISPLAY_WIDTH
        fyoffset = float(y)/DEFAULT_DISPLAY_HEIGHT

        ctx.c_real = re_start + (hpf(fxoffset) * ctx.c_width)
        ctx.c_imag = im_start + (hpf(fyoffset) * ctx.c_height)

        print("x %d, y %d"%(x,y))
        print("fxoff %f, fyoff %f"%(fxoffset,fyoffset))
        print("Real %s, Image %s"%(str(ctx.c_real),str(ctx.c_imag)))
        
        # only zoom in if there is a bounding box 
        if size.height() * size.width() >= 4:
            ctx.c_width  =  hpf(float(nrect.width()) / DEFAULT_DISPLAY_WIDTH) * ctx.c_width
            ctx.c_height =  hpf(float(nrect.height()) /DEFAULT_DISPLAY_HEIGHT) * ctx.c_height

        run(main_image_name, ctx)

        self.parent.refresh_ui(ctx)

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

        self.init_ui()


    # --
    # QTFractalMainWindow::sync_config_from_ui
    # --

    def sync_config_from_ui(self):

        ctx = RunContext()

        #set values from UI
        ctx.samples  = int(self.samples_text.text())
        ctx.max_iter = int(self.iter_text.text())
        ctx.image_w = int(float(self.img_width_text.text()))
        ctx.image_h = int(float(self.img_height_text.text())) 
        ctx.algo    = self.algo_combo.currentText()
        ctx.red     = float(self.red_edit.text())
        ctx.green   = float(self.green_edit.text()) 
        ctx.blue    = float(self.blue_edit.text()) 
        if self.julia_c_edit:
            ctx.julia_c = complex(self.julia_c_edit.text())

        ctx.c_width  = hpf(self.c_width_text.toPlainText())
        ctx.c_height = hpf(self.c_height_text.toPlainText())
        ctx.c_real = hpf(self.c_real_edit.toPlainText())
        ctx.c_imag = hpf(self.c_imag_edit.toPlainText())

        return ctx


    # --
    # QTFractalMainWindow::set_ui_defaults
    # --

    def set_ui_defaults(self):


        self.c_width_text.setPlainText(str(DEFAULT_C_WIDTH))
        self.c_height_text.setPlainText(str(DEFAULT_C_HEIGHT))
        self.img_width_text.setText(str(DEFAULT_DISPLAY_WIDTH))
        self.img_height_text.setText(str(DEFAULT_DISPLAY_HEIGHT))
        self.c_real_edit.setPlainText(str(DEFAULT_REAL))
        self.c_imag_edit.setPlainText(str(DEFAULT_IMAG))
        self.samples_text.setText(str(DEFAULT_SAMPLES))
        self.iter_text.setText(str(DEFAULT_MAX_ITER))

        self.red_edit.setText(str(DEFAULT_RED))
        self.green_edit.setText(str(DEFAULT_GREEN))
        self.blue_edit.setText(str(DEFAULT_BLUE))

        if self.julia_c_edit:
            self.julia_c_edit.setText(str(DEFAULT_JULIA_C))


        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(main_image_name)
        pixmap = pixmap.scaled(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.main_image.resize(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        self.grid.addWidget(self.main_image, 0, 1)

    # --
    # QTFractalMainWindow::refresh_ui
    # --

    def refresh_ui(self, context = None):

        if context:
            self.c_width_text.setPlainText(str(context.c_width))
            self.c_height_text.setPlainText(str(context.c_height))
            self.c_real_edit.setPlainText(str(context.c_real))
            self.c_imag_edit.setPlainText(str(context.c_imag))
            self.iter_text.setText(str(context.max_iter))
            self.samples_text.setText(str(context.samples))
            self.red_edit.setText(str(context.red))
            self.green_edit.setText(str(context.green))
            self.blue_edit.setText(str(context.blue))
            self.algo_combo.setCurrentText(context.algo)
            self.set_algo(None)
            if self.julia_c_edit:
                self.julia_c_edit.setText(str(context.julia_c))

        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(main_image_name)
        pixmap = pixmap.scaled(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.main_image.resize(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        self.grid.addWidget(self.main_image, 0, 1)

    # --
    # QTFractalMainWindow::run
    # --

    def run(self):
        global main_image_name

        ctx = self.sync_config_from_ui()
        run(main_image_name, ctx)
        self.refresh_ui()

    # --
    # QTFractalMainWindow::save
    # --

    def save(self):
        fname = QFileDialog.getSaveFileName(self, 'Save File', DEFAULT_SAVEDIR)
        fname = fname[0]
        if fname == '':
            return
        fd = open(fname, "wb")
        context = self.sync_config_from_ui()
        pickle.dump(context, fd)
        fd.close()

    # --
    # QTFractalMainWindow::load
    # --

    def load(self):
        fname = QFileDialog.getOpenFileNames(self, "Open File", DEFAULT_SAVEDIR)
        fname = fname[0]
        if len(fname) == 0:
            return
        fname = fname[0]    
        if fname == '':
            return
        fd = open(fname, "rb")
        context = pickle.load(fd)
        fd.close()
        self.refresh_ui(context)
        self.update()


    # --
    # QTFractalMainWindow::snapshot
    # --

    def snapshot(self):
        ctx = self.sync_config_from_ui()
        self.snap_pop = SnapshotPopup(self, ctx)
        self.snap_pop.setGeometry(QRect(100, 100, 400, 200))
        self.snap_pop.show()

    # --
    # QTFractalMainWindow::reset
    # --

    def reset(self):
        self.set_ui_defaults()
        ctx = self.sync_config_from_ui()
        run(main_image_name, ctx)
        self.refresh_ui()


    # --    
    # QTFractalMainWindow::set_algo
    # --    

    def set_algo(self, event):
        res = self.algo_combo.currentText()
        if res == 'julia' or res == 'cjulia':
            if not self.julia_c_edit:
                self.julia_c_label = QLabel("Julia c: ")
                self.julia_c_edit  = QLineEdit(self) 
                self.left_config_grid.addWidget(self.julia_c_label, 10, 0)
                self.left_config_grid.addWidget(self.julia_c_edit,  10, 1)
                self.julia_c_edit.setText(str(DEFAULT_JULIA_C))
        else:
            if self.julia_c_edit:
                print("REMOVE!")
                self.left_config_grid.removeWidget(self.julia_c_label)
                self.left_config_grid.removeWidget(self.julia_c_edit)
                self.julia_c_label.deleteLater()
                self.julia_c_edit.deleteLater()
                self.julia_c_edit  = None 
                self.julia_c_label = None 

        #self.refresh_ui()
        #self.sync_config_from_ui()
        self.update()    

    # --
    # QTFractalMainWindow::unit_ui
    # --

    def init_ui(self):
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

        btn_save = QPushButton("save")
        btn_save.clicked.connect(self.save)

        btn_load = QPushButton("load")
        btn_load.clicked.connect(self.load)

        btn_snapshot = QPushButton("snapshot")
        btn_snapshot.clicked.connect(self.snapshot)

        btn_reset = QPushButton("reset")
        btn_reset.clicked.connect(self.reset)

        self.main_image.setAlignment(Qt.AlignCenter)

        # Basic config

        algo_label = QLabel('Algorithm')
        self.algo_combo = QComboBox()
        self.algo_combo.addItem("ldnative")
        self.algo_combo.addItem("hpnative")
        self.algo_combo.addItem("mandeldistance")
        self.algo_combo.addItem("csmooth")
        self.algo_combo.addItem("julia")
        self.algo_combo.addItem("cjulia")

        # adding action to combo box
        self.algo_combo.activated.connect(self.set_algo)

        c_width_label   = QLabel('Cmplx w')
        c_height_label = QLabel('Cmplx h')

        self.c_width_text   = QPlainTextEdit(self)
        self.c_height_text  = QPlainTextEdit(self)

        img_width_label = QLabel('Image w')
        img_height_label = QLabel('Image h')

        self.img_width_text  = QLineEdit(self)
        self.img_width_text.setReadOnly(True)
        self.img_height_text  = QLineEdit(self)
        self.img_height_text.setReadOnly(True)

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

        # Only show julia c if julia selected as lago
        # julia_c_label      = QLabel("Julia c: ")
        self.julia_c_edit = None
        # self.julia_c_edit  = QLineEdit(self) 


        c_real_label = QLabel("Center Real:") 
        self.c_real_edit  = QPlainTextEdit()
        c_imag_label = QLabel("Center Imaginary:") 
        self.c_imag_edit  = QPlainTextEdit()

        
        # Left side config params
        grid_config = QGridLayout()
        self.left_config_grid = grid_config
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

        #grid_config.addWidget(julia_c_label ,10, 0)
        #grid_config.addWidget(self.julia_c_edit, 10, 1)

        grid_config.addWidget(btn_run, 11, 0)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(btn_snapshot)
        buttonLayout.addWidget(btn_reset)
        grid_config.addLayout(buttonLayout, 11, 1)

        # Right side inputs for c_real and c_imag 
        grid_right = QGridLayout()
        grid_right.addWidget(c_real_label ,0, 0)
        grid_right.addWidget(self.c_real_edit  ,1, 0)
        grid_right.addWidget(c_imag_label ,2, 0)
        grid_right.addWidget(self.c_imag_edit  ,3, 0)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(btn_save)
        buttonLayout.addWidget(btn_load)
        grid_right.addLayout(buttonLayout, 4,0)


        # MAIN GRID

        self.grid = QGridLayout()
        self.grid.setSpacing(10)

        self.grid.addLayout(grid_config,     0, 0)
        self.grid.addWidget(self.main_image, 0, 1)
        self.grid.addLayout(grid_right,      0, 2)

        self.setLayout(self.grid)

        self.set_ui_defaults()
        self.refresh_ui()

        # use splash screen to start
        self.main_image = FractalImgQLabel(self)
        pixmap = QPixmap(splash_image_name)
        pixmap = pixmap.scaled(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
        self.main_image.setPixmap(pixmap)
        self.main_image.resize(DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT)
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
