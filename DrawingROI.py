from PyQt5.QtCore import (QRectF, Qt)
from PyQt5.QtGui import (QPainter, QPixmap, QColor)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout,
                             QGraphicsObject, QGraphicsView, QWidget,
                             QGraphicsScene, QGraphicsItem, QLabel)
import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib.path import Path as MplPath
import readDICOM_Image as readDICOM_Image


class GraphicsItem(QGraphicsItem):  
    def __init__(self, coordLabel, meanLabel): 
        super(GraphicsItem, self).__init__()
        #self.mypixmap = readDICOM_Image.returnPixelArray('IM_0001')
        self.mypixmap = QPixmap("KarlaOnMyShoulder.jpg")
        self.mypixmap = self.mypixmap.scaled(2000, 2000)
        self.width = float(self.mypixmap.width())
        self.height = float(self.mypixmap.height())
        self.last_x, self.last_y = None, None
        self.pen_color = QColor("#FF0000")
        self.coordLabel = coordLabel
        self.meanLabel = meanLabel
        self.mainList = []
        self.setAcceptHoverEvents(True)


    def paint(self, painter, option, widget):
        painter.setOpacity(1)
        painter.drawPixmap(0,0, self.width, self.height, self.mypixmap)
        

    def boundingRect(self):  
        return QRectF(0,0,self.width, self.height)


    def hoverMoveEvent(self, event):
        xCoord = event.pos().x()
        yCoord = event.pos().y()
        qimage = self.mypixmap.toImage()
        pixelColour = qimage.pixelColor(xCoord,  yCoord ).getRgb()[:-1]
        pixelValue = qimage.pixelColor(xCoord,  yCoord ).value()
        self.coordLabel.setText("Pixel value {}, pixel colour {} @ X:{}, Y:{}".format(pixelValue, 
                                                                                      pixelColour,
                                                                                      xCoord, 
                                                                                      yCoord))


    def mouseMoveEvent(self, event):
        if self.last_x is None: # First event.
          self.last_x = (event.pos()).x()
          self.last_y = (event.pos()).y()
          return #  Ignore the first time.
        myPainter = QPainter(self.mypixmap)
        p = myPainter.pen()
        p.setWidth(4)
        p.setColor(self.pen_color)
        myPainter.setPen(p)
        #Draws a line from (x1 , y1 ) to (x2 , y2 ).
        xCoord = event.pos().x()
        yCoord = event.pos().y()
       # qimage = self.mypixmap.toImage()
       # pixelValue = qimage.pixel(xCoord,  yCoord ).value()
       # self.coordLabel.setText("Pixel value {} @ X:{}, Y:{}".format(pixelValue, xCoord, yCoord))
        #print(self.last_x, self.last_y, xCoord, yCoord)
        myPainter.drawLine(self.last_x, self.last_y, xCoord, yCoord)
        myPainter.end()
        self.update()
        # Update the origin for next time.
        self.last_x = xCoord
        self.last_y = yCoord
        self.mainList.append([self.last_x, self.last_y])


    def get_mask(self, poly_verts, data):
        ny, nx, nz = data.shape
        #poly_verts = ([(self.x[0], self.y[0])]
          #            + list(zip(reversed(self.x), reversed(self.y))))
        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        roi_path = MplPath(poly_verts)
        grid = roi_path.contains_points(points).reshape((ny, nx))

        ##Initialze QtGui.QImage() with arguments data, height, width, and QImage.Format
        ##grid = np.array(grid).reshape(700150,700150).astype(np.int32) 
        #qimage = QtGui.QImage(grid, grid.shape[0],grid.shape[1],QtGui.QImage.Format_RGB32)
        ##img = PrintImage(QPixmap(qimage))
        
        #print('mask={}'.format(np.extract(grid, current_image)))

        #pix = QtGui.QPixmap.fromImage(roi_path)

        #mask = QtGui.QPixmap(np.extract(grid, current_image))
        #self.maskLabel.setPixmap(mask)
        return grid
    

    def get_mean_and_std(self, poly_verts, current_image):
        image = Image.open(current_image)
        data = asarray(image)
        mask = self.get_mask(poly_verts, data)
        mean = round(np.mean(np.extract(mask, data)), 3)
        std = round(np.std(np.extract(mask, data)), 3)
        print('mean {}, std {}'.format(mean, std))
        self.meanLabel.setText('mean {}, std {}'.format(mean, std))
        #return mean, std

    def mouseReleaseEvent(self, event):
        self.last_x = None
        self.last_y = None
        #print("mouseReleaseEvent ", event.pos())
        #print(self.mainList)
        #self.meanLabel
        self.get_mean_and_std( self.mainList, "KarlaOnMyShoulder.jpg")
        self.mainList = []


    def mousePressEvent(self, event):
        pass
   

class graphicsView(QGraphicsView):
    def __init__(self, coordLabel, meanLabel):
        super(graphicsView, self).__init__()
        self.scene = QGraphicsScene(self)
        self._zoom = 0
        self.graphicsItem = GraphicsItem(coordLabel, meanLabel)
        
        #self.myScale = 2
        #self.graphicsItem.setScale(self.myScale)
        self.fitInView()

        self.setScene(self.scene)
        #self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        #self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
       # self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scene.addItem(self.graphicsItem)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def fitInView(self, scale=True):
        rect = QRectF(self.graphicsItem.mypixmap.rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
            self.scale(1 / unity.width(), 1 / unity.height())
            viewrect = self.viewport().rect()
            scenerect = self.transform().mapRect(rect)
            factor = min(viewrect.width() / scenerect.width(),
                            viewrect.height() / scenerect.height())
            self.scale(factor, factor)
            self._zoom = 0

class Example(QMainWindow):    
    def __init__(self):
        super(Example, self).__init__()

        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.centralwidget.setLayout(QVBoxLayout(self.centralwidget))
        self.coordsLabel = QLabel("Mouse Coords")
        self.meanLabel = QLabel("ROI Mean Value")
        self.graphicsView = graphicsView(self.coordsLabel, self.meanLabel)
        self.centralwidget.layout().addWidget(self.graphicsView)
        self.centralwidget.layout().addWidget(self.coordsLabel)
        self.centralwidget.layout().addWidget(self.meanLabel)

        #self.setCentralWidget(self.y)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Example()
    w.show()
    sys.exit(app.exec_())
