from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsPixmapItem
from PyQt5.QtGui import QPen, QBrush, QPixmap
from PyQt5.Qt import Qt
from PyQt5.QtCore import QRectF, QPointF, QEvent
from PIL import Image, ImageQt, ImageEnhance
 
import sys
class MainScreen(QMainWindow):
    def __init__(self):

        super().__init__()
        #self.setupUi(self)

        # Graphic Screen set
        self.img = QGraphicsPixmapItem(QPixmap('KarlaOnMyShoulder.jpg'))
        self.scene = QGraphicsScene()
        self.graphicsView = QGraphicsView()
        self.scene.addItem(self.img)
        self.graphicsView.setScene(self.scene)


        # Full Screen set size
        _WIDTH_ADD = 25
        _HEIGHT_ADD = 25
        self.setGeometry(0, 0, 640 + _WIDTH_ADD, 500 + _HEIGHT_ADD)

        self.graphicsView.viewport().installEventFilter(self)

        self.current_item = None
        self.start_pos = QPointF()
        self.end_pos = QPointF()
        self.show()

    def eventFilter(self, o, e):
        if self.graphicsView.viewport() is o:
            if e.type() == QEvent.MouseButtonPress:
                if e.buttons() & Qt.LeftButton:
                    print("press")
                    self.start_pos = self.end_pos = self.graphicsView.mapToScene(
                        e.pos()
                    )
                    pen = QPen(QColor(240, 240, 240))
                    pen.setWidth(3)
                    brush = QBrush(QColor(100, 255, 100, 100))
                    self.current_item = self.scene.addRect(QRectF(), pen, brush)
                    self._update_item()
            elif e.type() == QEvent.MouseMove:
                if e.buttons() & Qt.LeftButton and self.current_item is not None:
                    print("move")
                    self.end_pos = self.graphicsView.mapToScene(e.pos())
                    self._update_item()
            elif e.type() == QEvent.MouseButtonRelease:
                print("release")
                self.end_pos = self.graphicsView.mapToScene(e.pos())
                self._update_item()
                self.current_item = None

        return super().eventFilter(o, e)

    def _update_item(self):
        if self.current_item is not None:
            self.current_item.setRect(QRectF(self.start_pos, self.end_pos).normalized())


App = QApplication(sys.argv)
window = MainScreen()
sys.exit(App.exec())