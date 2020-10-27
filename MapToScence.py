from PyQt5.QtCore import (QRectF)
from PyQt5.QtGui import (QPainter, QPixmap)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QGraphicsObject, QGraphicsView, QGraphicsScene, QGraphicsItem)


class TicTacToe(QGraphicsItem):
    def __init__(self, helper):
        super(TicTacToe, self).__init__()
        self.mypixmap = QPixmap("KarlaOnMyShoulder.jpg")
        self.setAcceptHoverEvents(True)

    def paint(self, painter, option, widget):
        painter.setOpacity(1)
        painter.drawPixmap(0,0, 512, 512, self.mypixmap)
       

    def boundingRect(self):
        return QRectF(0,0,512, 512)

    def hoverMoveEvent(self, event):
        #print("hoverMoveEvent ", event.pos())
        print("hoverMoveEvent", self.mapToScene(event.pos()))

    def mouseMoveEvent(self, event):
        print("mouseMoveEvent", self.mapToScene(event.pos()))

    def mousePressEvent(self, event):
        #print("mousePressEvent", event.pos())
        print("mousePressEvent", self.mapToScene(event.pos()))


class MyGraphicsView(QGraphicsView):
    def __init__(self):
        super(MyGraphicsView, self).__init__()
        self.scene = QGraphicsScene(self)
        self.setMouseTracking(True)

        self.tic_tac_toe = TicTacToe(self)

        self.myScale = 2
        self.tic_tac_toe.setScale(self.myScale)

        self.setScene(self.scene)
        self.scene.addItem(self.tic_tac_toe)

    def mouseMoveEvent(self, event):
        print("mouseMoveEvent", self.mapToScene(event.pos()))

class Example(QMainWindow):    
    def __init__(self):
        super(Example, self).__init__()
        self.y = MyGraphicsView()
        self.setCentralWidget(self.y)

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Example()
    w.show()
    sys.exit(app.exec_())
