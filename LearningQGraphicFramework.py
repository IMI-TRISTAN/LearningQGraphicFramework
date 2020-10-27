from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsPixmapItem
from PyQt5.QtGui import QPen, QBrush, QPixmap
from PyQt5.Qt import Qt
from PyQt5.QtCore import QRectF
from PIL import Image, ImageQt, ImageEnhance
 
import sys



class pixArrayItem(QGraphicsItem):
    def __init__(self, qimage):
      super().__init__()
      self.image = qimage
      self.setFlags(QGraphicsItem.ItemIsSelectable)
      self.setSelected(True)
      self.setAcceptHoverEvents(True)

    def boundingRect(self):
        penWidth = 1.0
        return QRectF(-10 - penWidth / 2, -10 - penWidth / 2,
                      20 + penWidth, 20 + penWidth)

    def paint(self, painter, option, widget):
        painter.drawImage(-550,-550, self.image)

    def mousePressEvent(self, event):
        QGraphicsItem.mousePressEvent(event)
        print('mouseMoveEvent: pos {}'.format(event.pos()))

    def mouseReleaseEvent(self, e):
        pass

 
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.title = "PyQt5 QGraphicView"
        self.top = 200
        self.left = 500
        #self.width = 600
        #self.height = 500
        #self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        
 
        self.scene  = QGraphicsScene()
        self.pen = QPen(Qt.red)
        self.whiteBrush = QBrush(Qt.white)
        
        graphicView = QGraphicsView(self.scene, self)
        graphicView.setMouseTracking(True)
        graphicView.viewport().installEventFilter(self)
       
        
        self.setCentralWidget(graphicView)

        img = Image.open('KarlaOnMyShoulder.jpg')
        self.imgQ = ImageQt.ImageQt(img)  # we need to hold reference to imgQ, or it will crash
        pixMap = QPixmap.fromImage(self.imgQ)
        #pixMapItem = Canvas(pixMap)
        pixMapItem = pixArrayItem(self.imgQ)

        w = pixMap.width() 
        h = pixMap.height()
        graphicView.setGeometry(0,0,w +10,h +10)
        #graphicView.scale(.75, .75)
        self.width = w
        self.height = h
        #pixMap.scaled(w*.75,h*.75, Qt.KeepAspectRatio)
        self.scene.addItem(pixMapItem)
       
        #displayWindow = QRectF(self.top + 10,  self.left + 10, 
        #                      self.width - 10, self.height - 10)
        
        self.setGeometry(self.left, self.top, self.width + 100, self.height + 100)
       
        self.show()
 
 
   
 
 
 
 
 
App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
