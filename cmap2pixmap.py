from matplotlib import cm
import numpy as np

from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5 import QtGui
from PyQt5.QtCore import (QRectF, Qt)
from PyQt5.QtGui import (QPainter, QPixmap, QColor, QImage)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout,
                             QGraphicsObject, QGraphicsView, QWidget,
                             QGraphicsScene, QGraphicsItem, QLabel, 
                             QDoubleSpinBox, QGridLayout, QComboBox)

def cmap2pixmap(cmap, steps=50):
    """Convert a maplotlib colormap into a QPixmap
    :param cmap: The colormap to use
    :type cmap: Matplotlib colormap instance (e.g. matplotlib.cm.gray)
    :param steps: The number of color steps in the output. Default=50
    :type steps: int
    :rtype: QPixmap
    """
    sm = cm.ScalarMappable(cmap=cmap)
    sm.norm.vmin = 0.0
    sm.norm.vmax = 1.0
    inds = np.linspace(0, 1, steps)
    rgbas = sm.to_rgba(inds)
    rgbas = [QColor(int(r * 255), int(g * 255),
                    int(b * 255), int(a * 255)).rgba() for r, g, b, a in rgbas]
    im = QImage(steps, 1, QImage.Format_Indexed8)
    im.setColorTable(rgbas)
    for i in range(steps):
        im.setPixel(i, 0, i)
    im = im.scaled(100, 100)
    pm = QPixmap.fromImage(im)
    return pm

class Example(QMainWindow):    
    def __init__(self):
        super(Example, self).__init__()

        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
        self.coordsLabel = QLabel()
        self.coordsLabel.setPixmap(cmap2pixmap('plasma', steps=50))
        
        self.centralwidget.layout().addWidget(self.coordsLabel)
        

import sys

##from PyQt5 import QtCore, QtGui
##from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor
##from PyQt5.QtCore import Qt, QSize

#from PyQt5 import QtGui, QtCore
#from PyQt5.QtCore import (QRectF, Qt, QSize)
#from PyQt5.QtGui import (QPainter, QPixmap, QColor, QImage, QPalette)
#from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout,
#                             QGraphicsObject, QGraphicsView, QWidget,
#                             QGraphicsScene, QGraphicsItem, QLabel, 
#                             QDoubleSpinBox, QGridLayout, QComboBox, QScrollArea)

#import numpy as np
#import matplotlib as mpl
#import matplotlib.cm


#class STAWindow(QMainWindow):
#    def __init__(self, parent=None):
#        QMainWindow.__init__(self)
#        ts = np.arange(0, 200, 20)
#        nids = np.arange(100)
#        width, height = 32, 32
#        scale = 2 # setting to a float will give uneven sized pixels

#        cmap = mpl.cm.jet(np.arange(256), alpha=None, bytes=True) # 8 bit RGBA colormap
#        #cmap = mpl.cm.get_cmap('jet')
        
#        # from Qt docs, need to use ARGB format:
#        # http://qt-project.org/doc/qt-4.8/qimage.html#image-formats
#        # convert to 8 bit ARGB colormap, but due to little-endianness, need to arrange
#        # array columns in reverse BGRA order:
#        cmap[:, [0, 1, 2, 3]] = cmap[:, [2, 1, 0, 3]]
#        colortable = cmap.view(dtype=np.uint32).ravel().tolist() # QVector<QRgb> colors 
#        layout =QGridLayout() # can set vert and horiz spacing
#        #layout.setContentsMargins(0, 0, 0, 0) # doesn't seem to do anything

#        # place time labels along top
#        for ti, t in enumerate(ts):
#            label =QLabel(str(t))
#            layout.addWidget(label, 0, ti+1)
#        # plot each row, with its nid label
#        for ni, nid in enumerate(nids):
#            label =QLabel('n'+str(nid)) # nid label on left
#            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
#            layout.addWidget(label, ni+1, 0)
#            for ti, t in enumerate(ts):
#                data = np.uint8(np.random.randint(0, 255, size=(height, width)))
#                image = QImage(data.data, width, height, QImage.Format_Indexed8)
#                image.ndarray = data # hold a ref, prevent gc
#                image.setColorTable(colortable)
#                image = image.scaled(QSize(scale*width, scale*height)) # scale it
#                pixmap = QPixmap.fromImage(image)
#                label =QLabel()
#                label.setPixmap(pixmap)
#                layout.addWidget(label, ni+1, ti+1) # can also control alignment

#        mainwidget =QWidget(self)
#        mainwidget.setLayout(layout)

#        scrollarea =QScrollArea()
#        scrollarea.setWidget(mainwidget)

#        self.setCentralWidget(scrollarea)
#        self.setWindowTitle('STA')
#        #palette = QPalette(QColor(255, 255, 255))
#        #self.setPalette(palette) # set white background, or perhaps more


if __name__ == "__main__":
    app = QApplication(sys.argv)
    stawindow = Example()
    stawindow.show()
    sys.exit(app.exec_())