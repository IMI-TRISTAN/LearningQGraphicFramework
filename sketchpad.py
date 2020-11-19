def getLookupTable(self, img=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this 
        HistogramLUTItem.
        """
        if n is None:
            if img.dtype == np.uint8:
                n = 256
            else:
                n = 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut


def getLookupTable(self, nPts, alpha=None):
        """
        Return an RGB(A) lookup table (ndarray). 
        
        ==============  ============================================================================
        **Arguments:**
        nPts            The number of points in the returned lookup table.
        alpha           True, False, or None - Specifies whether or not alpha values are included
                        in the table.If alpha is None, alpha will be automatically determined.
        ==============  ============================================================================
        """
        if alpha is None:
            alpha = self.usesAlpha()
        if alpha:
            table = np.empty((nPts,4), dtype=np.ubyte)
        else:
            table = np.empty((nPts,3), dtype=np.ubyte)
            
        for i in range(nPts):
            x = float(i)/(nPts-1)
            color = self.getColor(x, toQColor=False)
            table[i] = color[:table.shape[1]]
            
        return table

def setColorMap(self, cm):
        self.setColorMode('rgb')
        for t in list(self.ticks.keys()):
            self.removeTick(t, finish=False)
        colors = cm.getColors(mode='qcolor')
        for i in range(len(cm.pos)):
            x = cm.pos[i]
            c = colors[i]
            self.addTick(x, c, finish=False)
        self.updateGradient()
        self.sigGradientChangeFinished.emit(self)

def getColor(self, x, toQColor=True):
        """
        Return a color for a given value.
        
        ==============  ==================================================================
        **Arguments:**
        x               Value (position on gradient) of requested color.
        toQColor        If true, returns a QColor object, else returns a (r,g,b,a) tuple.
        ==============  ==================================================================
        """
        ticks = self.listTicks()
        if x <= ticks[0][1]:
            c = ticks[0][0].color
            if toQColor:
                return QtGui.QColor(c)  # always copy colors before handing them out
            else:
                return (c.red(), c.green(), c.blue(), c.alpha())
        if x >= ticks[-1][1]:
            c = ticks[-1][0].color
            if toQColor:
                return QtGui.QColor(c)  # always copy colors before handing them out
            else:
                return (c.red(), c.green(), c.blue(), c.alpha())
            
        x2 = ticks[0][1]
        for i in range(1,len(ticks)):
            x1 = x2
            x2 = ticks[i][1]
            if x1 <= x and x2 >= x:
                break
                
        dx = (x2-x1)
        if dx == 0:
            f = 0.
        else:
            f = (x-x1) / dx
        c1 = ticks[i-1][0].color
        c2 = ticks[i][0].color
        if self.colorMode == 'rgb':
            r = c1.red() * (1.-f) + c2.red() * f
            g = c1.green() * (1.-f) + c2.green() * f
            b = c1.blue() * (1.-f) + c2.blue() * f
            a = c1.alpha() * (1.-f) + c2.alpha() * f
            if toQColor:
                return QtGui.QColor(int(r), int(g), int(b), int(a))
            else:
                return (r,g,b,a)
        elif self.colorMode == 'hsv':
            h1,s1,v1,_ = c1.getHsv()
            h2,s2,v2,_ = c2.getHsv()
            h = h1 * (1.-f) + h2 * f
            s = s1 * (1.-f) + s2 * f
            v = v1 * (1.-f) + v2 * f
            c = QtGui.QColor()
            c.setHsv(h,s,v)
            if toQColor:
                return c
            else:
                return (c.red(), c.green(), c.blue(), c.alpha())


    # -*- coding: utf-8 -*-
"""
Demonstrate ability of ImageItem to be used as a canvas for painting with
the mouse.

"""

#import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with GraphicsView widget
w = pg.GraphicsView()
w.show()
w.resize(800,800)
w.setWindowTitle('pyqtgraph example: Draw')

view = pg.ViewBox()
w.setCentralItem(view)

## lock the aspect ratio
view.setAspectLocked(True)

## Create image item
img = pg.ImageItem(np.zeros((200,200)))
view.addItem(img)

## Set initial view bounds
view.setRange(QtCore.QRectF(0, 0, 200, 200))

## start drawing with 3x3 brush
kern = np.array([
    [0.0, 0.5, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.5, 0.0]
])
img.setDrawKernel( kern, mask=kern, center=(1,1), mode='add')
img.setLevels([0, 10])

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




# -*- coding: utf-8 -*-
"""
Demonstrates a variety of uses for ROI. This class provides a user-adjustable
region of interest marker. It is possible to customize the layout and 
function of the scale/rotate handles in very flexible ways. 
"""

#import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

pg.setConfigOptions(imageAxisOrder='row-major')

## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10
arr += np.sin(np.linspace(0, 20, 100)).reshape(1, 100)
arr += np.random.normal(size=(100,100))

# add an arrow for asymmetry
arr[10, :50] = 10
arr[9:12, 44:48] = 10
arr[8:13, 44:46] = 10


## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(1000,800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')

text = """Data Selection From Image.<br>\n
Drag an ROI or its handles to update the selected image.<br>
Hold CTRL while dragging to snap to pixel boundaries<br>
and 15-degree rotation angles.
"""
w1 = w.addLayout(row=0, col=0)
label1 = w1.addLabel(text, row=0, col=0)
v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
v1b = w1.addViewBox(row=2, col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
img1b = pg.ImageItem()
v1b.addItem(img1b)
v1a.disableAutoRange('xy')
v1b.disableAutoRange('xy')
v1a.autoRange()
v1b.autoRange()

rois = []
rois.append(pg.RectROI([20, 20], [20, 20], pen=(0,9)))
rois[-1].addRotateHandle([1,0], [0.5, 0.5])
rois.append(pg.LineROI([0, 60], [20, 80], width=5, pen=(1,9)))
rois.append(pg.MultiRectROI([[20, 90], [50, 60], [60, 90]], width=5, pen=(2,9)))
rois.append(pg.EllipseROI([60, 10], [30, 20], pen=(3,9)))
rois.append(pg.CircleROI([80, 50], [20, 20], pen=(4,9)))
#rois.append(pg.LineSegmentROI([[110, 50], [20, 20]], pen=(5,9)))
rois.append(pg.PolyLineROI([[80, 60], [90, 30], [60, 40]], pen=(6,9), closed=True))

def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    v1b.autoRange()
    
for roi in rois:
    roi.sigRegionChanged.connect(update)
    v1a.addItem(roi)

update(rois[-1])
    


text = """User-Modifiable ROIs<br>
Click on a line segment to add a new handle.
Right click on a handle to remove.
"""
w2 = w.addLayout(row=0, col=1)
label2 = w2.addLabel(text, row=0, col=0)
v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
v2a.addItem(r2a)
r2b = pg.PolyLineROI([[0,-20], [10,-10], [10,-30]], closed=False)
v2a.addItem(r2b)
v2a.disableAutoRange('xy')
#v2b.disableAutoRange('xy')
v2a.autoRange()
#v2b.autoRange()

text = """Building custom ROI types<Br>
ROIs can be built with a variety of different handle types<br>
that scale and rotate the roi around an arbitrary center location
"""
w3 = w.addLayout(row=1, col=0)
label3 = w3.addLabel(text, row=0, col=0)
v3 = w3.addViewBox(row=1, col=0, lockAspect=True)

r3a = pg.ROI([0,0], [10,10])
v3.addItem(r3a)
## handles scaling horizontally around center
r3a.addScaleHandle([1, 0.5], [0.5, 0.5])
r3a.addScaleHandle([0, 0.5], [0.5, 0.5])

## handles scaling vertically from opposite edge
r3a.addScaleHandle([0.5, 0], [0.5, 1])
r3a.addScaleHandle([0.5, 1], [0.5, 0])

## handles scaling both vertically and horizontally
r3a.addScaleHandle([1, 1], [0, 0])
r3a.addScaleHandle([0, 0], [1, 1])

r3b = pg.ROI([20,0], [10,10])
v3.addItem(r3b)
## handles rotating around center
r3b.addRotateHandle([1, 1], [0.5, 0.5])
r3b.addRotateHandle([0, 0], [0.5, 0.5])

## handles rotating around opposite corner
r3b.addRotateHandle([1, 0], [0, 1])
r3b.addRotateHandle([0, 1], [1, 0])

## handles rotating/scaling around center
r3b.addScaleRotateHandle([0, 0.5], [0.5, 0.5])
r3b.addScaleRotateHandle([1, 0.5], [0.5, 0.5])

v3.disableAutoRange('xy')
v3.autoRange()


text = """Transforming objects with ROI"""
w4 = w.addLayout(row=1, col=1)
label4 = w4.addLabel(text, row=0, col=0)
v4 = w4.addViewBox(row=1, col=0, lockAspect=True)
g = pg.GridItem()
v4.addItem(g)
r4 = pg.ROI([0,0], [100,100], removable=True)
r4.addRotateHandle([1,0], [0.5, 0.5])
r4.addRotateHandle([0,1], [0.5, 0.5])
img4 = pg.ImageItem(arr)
v4.addItem(r4)
img4.setParentItem(r4)

v4.disableAutoRange('xy')
v4.autoRange()

# Provide a callback to remove the ROI (and its children) when
# "remove" is selected from the context menu.
def remove():
    v4.removeItem(r4)
r4.sigRemoveRequested.connect(remove)








## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()



# -*- coding: utf-8 -*-
"""
This example demonstrates the use of ImageView, which is a high-level widget for 
displaying and analyzing 2D and 3D data. ImageView provides:

  1. A zoomable region (ViewBox) for displaying the image
  2. A combination histogram and gradient editor (HistogramLUTItem) for
     controlling the visual appearance of the image
  3. A timeline for selecting the currently displayed frame (for 3D data only).
  4. Tools for very basic analysis of image data (see ROI and Norm buttons)

"""
## Add path to library (just for examples; you do not need this)
#import initExample

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

# Interpret image data as row-major instead of col-major
pg.setConfigOptions(imageAxisOrder='row-major')

app = QtGui.QApplication([])

## Create window with ImageView widget
win = QtGui.QMainWindow()
win.resize(800,800)
imv = pg.ImageView()
win.setCentralWidget(imv)
win.show()
win.setWindowTitle('pyqtgraph example: ImageView')

## Create random 3D data set with noisy signals
img = pg.gaussianFilter(np.random.normal(size=(200, 200)), (5, 5)) * 20 + 100
img = img[np.newaxis,:,:]
decay = np.exp(-np.linspace(0,0.3,100))[:,np.newaxis,np.newaxis]
data = np.random.normal(size=(100, 200, 200))
data += img * decay
data += 2

## Add time-varying signal
sig = np.zeros(data.shape[0])
sig[30:] += np.exp(-np.linspace(1,10, 70))
sig[40:] += np.exp(-np.linspace(1,10, 60))
sig[70:] += np.exp(-np.linspace(1,10, 30))

sig = sig[:,np.newaxis,np.newaxis] * 3
data[:,50:60,30:40] += sig


## Display the data and assign each frame a time value from 1.0 to 3.0
imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))

## Set a custom color map
colors = [
    (0, 0, 0),
    (45, 5, 61),
    (84, 42, 55),
    (150, 87, 60),
    (208, 171, 141),
    (255, 255, 255)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
imv.setColorMap(cmap)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

