
 #def createGraphicView(self):
    #    self.scene  =QGraphicsScene()
    #    self.greenBrush = QBrush(Qt.green)
    #    self.grayBrush = QBrush(Qt.gray)
 
    #    self.pen = QPen(Qt.red)
 
    #    graphicView = QGraphicsView(self.scene, self)
    #    graphicView.setGeometry(0,0,600,500)
 
    #    self.shapes()
 
 
    #def shapes(self):
    #    ellipse = self.scene.addEllipse(20,20, 200,200, self.pen, self.greenBrush)
    #    rect = self.scene.addRect(-100,-100, 200,200, self.pen, self.grayBrush)
 
    #    ellipse.setFlag(QGraphicsItem.ItemIsMovable)
    #    rect.setFlag(QGraphicsItem.ItemIsMovable)
    #    ellipse.setFlag(QGraphicsItem.ItemIsSelectable)

    #rect = self.scene.addRect( displayWindow, self.pen, self.whiteBrush)
 
        #graphicView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        #auto rectItem = new QGraphicsRectItem(0,0,20,20);
        #rectItem->setPos(50,50);
        #scene->addItem(rectItem)


class Canvas(QGraphicsPixmapItem):
  def __init__(self, pixMap):
      super().__init__()
      self.setPixmap(pixMap)
      

  def mouseMoveEvent(self, event):
        print('mouseMoveEvent: pos {}'.format(event.pos()))

  #def mouseMoveEvent(self, e):
  #    if self.last_x is None: # First event.
  #        self.last_x = e.x()
  #        self.last_y = e.y()
  #        return #  Ignore the first time.
  #    painter = QtGui.QPainter(self.pixmap())
  #    p = painter.pen()
  #    p.setWidth(4)
  #    p.setColor(self.pen_color)
  #    painter.setPen(p)
  #    painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
  #    painter.end()
  #    self.update()
  #    # Update the origin for next time.
  #    self.last_x = e.x()
  #    self.last_y = e.y()
  #    self.mainList.append([self.last_x, self.last_y])
    

  #def mouseReleaseEvent(self, e):
  #  self.last_x = None
  #  self.last_y = None
  #  #print(self.mainList)
  #  #image = Image.open('FERRET_LOGO.png')
  # # pixmap = QtGui.QPixmap(600, 300)
  #  #pixmap.fill(Qt.white)
  #  #data = asarray(pixmap.toImage())

  #  img = np.zeros([600,300,3],dtype=np.uint8)
  #  img.fill(255)
  #  data = asarray(img)
  #  self.get_mean_and_std( self.mainList, data) #,data
  #  self.mainList = []

#class MyGraphicsView(QGraphicsView):

#    def __init__(self, parent):
#        QGraphicsView.__init__(self, parent)
#        self.setMouseTracking(True)

#    def mouseMoveEvent(self, event):
#        print('mouseMoveEvent: pos {}'.format(event.pos()))

#The easiest way to do this is by picking out pixels in your image that correspond to places 
#where the mask is white. If you want pixel on the boundary use the mask as you have shown it.
# If you want pixel in (and on) the boundary; draw it instead as a filled contour (thickness=-1). 
# Here's an example:

img = cv2.imread('image.jpg')
mask = cv2.imread('mask.png', 0)
locs = np.where(mask == 255)
pixels = img[locs]
print(np.mean(pixels))