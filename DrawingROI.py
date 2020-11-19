from PyQt5 import QtGui
from PyQt5.QtCore import (QRectF, Qt)
from PyQt5.QtGui import (QPainter, QPixmap, QColor, QImage, qRgb)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout,
                             QGraphicsObject, QGraphicsView, QWidget,
                             QGraphicsScene, QGraphicsItem, QLabel, 
                             QDoubleSpinBox, QGridLayout, QComboBox)
import numpy as np
from numpy import nanmin, nanmax
from matplotlib.path import Path as MplPath
import readDICOM_Image as readDICOM_Image
import scipy

#'useOpenGL': useOpenGL, ## by default, this is platform-dependent (see widgets/GraphicsView). Set to True or False to explicitly enable/disable opengl.

CONFIG_OPTIONS = {
    'leftButtonPan': True,  ## if false, left button drags a rubber band for zooming in viewbox
    'foreground': 'd',  ## default foreground color for axes, labels, etc.
    'background': 'k',        ## default background for GraphicsWidget
    'antialias': False,
    'editorCommand': None,  ## command used to invoke code editor from ConsoleWidgets
    'useWeave': False,       ## Use weave to speed up some operations, if it is available
    'weaveDebug': False,    ## Print full error message if weave compile fails
    'exitCleanup': True,    ## Attempt to work around some exit crash bugs in PyQt and PySide
    'enableExperimental': False, ## Enable experimental features (the curious can search for this key in the code)
    'crashWarning': False,  # If True, print warnings about situations that may result in a crash
} 

def setConfigOption(opt, value):
    CONFIG_OPTIONS[opt] = value


def setConfigOptions(**opts):
    CONFIG_OPTIONS.update(opts)


def getConfigOption(opt):
    return CONFIG_OPTIONS[opt]


def quickMinMax(data):
        """
        Estimate the min/max values of *data* by subsampling.
        """
        while data.size > 1e6:
            ax = np.argmax(data.shape)
            sl = [slice(None)] * data.ndim
            sl[ax] = slice(None, None, 2)
            data = data[sl]
        return nanmin(data), nanmax(data)


def rescaleData(data, scale, offset, dtype=None):
    """Return data rescaled and optionally cast to a new dtype::
    
        data => (data-offset) * scale
        
    Uses scipy.weave (if available) to improve performance.
    """
   # print("rescaleData")
    if dtype is None:
        dtype = data.dtype
    else:
        dtype = np.dtype(dtype)
    
    try:
        if not getConfigOption('useWeave'):
            raise Exception('Weave is disabled; falling back to slower version.')
        #try:
        #    import scipy.weave
        #except ImportError:
        #    raise Exception('scipy.weave is not importable; falling back to slower version.')
        
        ## require native dtype when using weave
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder('='))
        if not dtype.isnative:
            weaveDtype = dtype.newbyteorder('=')
        else:
            weaveDtype = dtype
        
        newData = np.empty((data.size,), dtype=weaveDtype)
        flat = np.ascontiguousarray(data).reshape(data.size)
        size = data.size
        
        code = """
        double sc = (double)scale;
        double off = (double)offset;
        for( int i=0; i<size; i++ ) {
            newData[i] = ((double)flat[i] - off) * sc;
        }
        """
        scipy.weave.inline(code, ['flat', 'newData', 'size', 'offset', 'scale'], compiler='gcc')
        if dtype != weaveDtype:
            newData = newData.astype(dtype)
        data = newData.reshape(data.shape)
    except:
        if getConfigOption('useWeave'):
            if getConfigOption('weaveDebug'):
                debug.printExc("Error; disabling weave.")
            setConfigOptions(useWeave=False)
        
        #p = np.poly1d([scale, -offset*scale])
        #data = p(data).astype(dtype)
        d2 = data-offset
        d2 *= scale
        data = d2.astype(dtype)
    return data
    

def applyLookupTable(data, lut):
    """
    Uses values in *data* as indexes to select values from *lut*.
    The returned data has shape data.shape + lut.shape[1:]
    
    Note: color gradient lookup tables can be generated using GradientWidget.
    """
    if data.dtype.kind not in ('i', 'u'):
        data = data.astype(int)
    
    return np.take(lut, data, axis=0, mode='clip')  


def makeARGB(data, lut=None, levels=None, scale=None, useRGBA=False): 
    """ 
    Convert an array of values into an ARGB array suitable for building QImages, OpenGL textures, etc.
    
    Returns the ARGB array (values 0-255) and a boolean indicating whether there is alpha channel data.
    This is a two stage process:
    
        1) Rescale the data based on the values in the *levels* argument (min, max).
        2) Determine the final output by passing the rescaled values through a lookup table.
   
    Both stages are optional.
    
    ============== ==================================================================================
    **Arguments:**
    data           numpy array of int/float types. If 
    levels         List [min, max]; optionally rescale data before converting through the
                   lookup table. The data is rescaled such that min->0 and max->*scale*::
                   
                      rescaled = (clip(data, min, max) - min) * (*scale* / (max - min))
                   
                   It is also possible to use a 2D (N,2) array of values for levels. In this case,
                   it is assumed that each pair of min,max values in the levels array should be 
                   applied to a different subset of the input data (for example, the input data may 
                   already have RGB values and the levels are used to independently scale each 
                   channel). The use of this feature requires that levels.shape[0] == data.shape[-1].
    scale          The maximum value to which data will be rescaled before being passed through the 
                   lookup table (or returned if there is no lookup table). By default this will
                   be set to the length of the lookup table, or 256 is no lookup table is provided.
                   For OpenGL color specifications (as in GLColor4f) use scale=1.0
    lut            Optional lookup table (array with dtype=ubyte).
                   Values in data will be converted to color by indexing directly from lut.
                   The output data shape will be input.shape + lut.shape[1:].
                   
                   Note: the output of makeARGB will have the same dtype as the lookup table, so
                   for conversion to QImage, the dtype must be ubyte.
                   
                   Lookup tables can be built using GradientWidget.
    useRGBA        If True, the data is returned in RGBA order (useful for building OpenGL textures). 
                   The default is False, which returns in ARGB order for use with QImage 
                   (Note that 'ARGB' is a term used by the Qt documentation; the _actual_ order 
                   is BGRA).
    ============== ==================================================================================
    """
    if lut is not None and not isinstance(lut, np.ndarray):
        lut = np.array(lut)
    if levels is not None and not isinstance(levels, np.ndarray):
        levels = np.array(levels)
    
    if levels is not None:
        if levels.ndim == 1:
            if len(levels) != 2:
                raise Exception('levels argument must have length 2')
        elif levels.ndim == 2:
            if lut is not None and lut.ndim > 1:
                raise Exception('Cannot make ARGB data when bot levels and lut have ndim > 2')
            if levels.shape != (data.shape[-1], 2):
                raise Exception('levels must have shape (data.shape[-1], 2)')
        else:
            print(levels)
            raise Exception("levels argument must be 1D or 2D.")

    if scale is None:
        if lut is not None:
            scale = lut.shape[0]
        else:
            scale = 255.

    ## Apply levels if given
    if levels is not None:
        
        if isinstance(levels, np.ndarray) and levels.ndim == 2:
            ## we are going to rescale each channel independently
            if levels.shape[0] != data.shape[-1]:
                raise Exception("When rescaling multi-channel data, there must be the same number of levels as channels (data.shape[-1] == levels.shape[0])")
            newData = np.empty(data.shape, dtype=int)
            for i in range(data.shape[-1]):
                minVal, maxVal = levels[i]
                if minVal == maxVal:
                    maxVal += 1e-16
                newData[...,i] = rescaleData(data[...,i], scale/(maxVal-minVal), minVal, dtype=int)
            data = newData
        else:
            minVal, maxVal = levels
            if minVal == maxVal:
                maxVal += 1e-16
            if maxVal == minVal:
                data = rescaleData(data, 1, minVal, dtype=int)
            else:
                data = rescaleData(data, scale/(maxVal-minVal), minVal, dtype=int)

    ## apply LUT if given
    if lut is not None:
        data = applyLookupTable(data, lut)
    else:
        if data.dtype is not np.ubyte:
            data = np.clip(data, 0, 255).astype(np.ubyte)

    ## copy data into ARGB ordered array
    imgData = np.empty(data.shape[:2]+(4,), dtype=np.ubyte)

    if useRGBA:
        order = [0,1,2,3] ## array comes out RGBA
    else:
        order = [2,1,0,3] ## for some reason, the colors line up as BGR in the final image.
        
    if data.ndim == 2:
        # This is tempting:
        #   imgData[..., :3] = data[..., np.newaxis]
        # ..but it turns out this is faster:
        for i in range(3):
            imgData[..., i] = data
    elif data.shape[2] == 1:
        for i in range(3):
            imgData[..., i] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., i] = data[..., order[i]] 
        
    if data.ndim == 2 or data.shape[2] == 3:
        alpha = False
        imgData[..., 3] = 255
    else:
        alpha = True
        
    return imgData, alpha


def makeQImage(imgData, alpha=None, copy=True, transpose=True):
    """
    Turn an ARGB array into QImage.
    By default, the data is copied; changes to the array will not
    be reflected in the image. The image will be given a 'data' attribute
    pointing to the array which shares its data to prevent python
    freeing that memory while the image is in use.
    
    ============== ===================================================================
    **Arguments:**
    imgData        Array of data to convert. Must have shape (width, height, 3 or 4) 
                   and dtype=ubyte. The order of values in the 3rd axis must be 
                   (b, g, r, a).
    alpha          If True, the QImage returned will have format ARGB32. If False,
                   the format will be RGB32. By default, _alpha_ is True if
                   array.shape[2] == 4.
    copy           If True, the data is copied before converting to QImage.
                   If False, the new QImage points directly to the data in the array.
                   Note that the array must be contiguous for this to work
                   (see numpy.ascontiguousarray).
    transpose      If True (the default), the array x/y axes are transposed before 
                   creating the image. Note that Qt expects the axes to be in 
                   (height, width) order whereas pyqtgraph usually prefers the 
                   opposite.
    ============== ===================================================================    
    """
    ## create QImage from buffer
    
    ## If we didn't explicitly specify alpha, check the array shape.
    if alpha is None:
        alpha = (imgData.shape[2] == 4)
        
    copied = False
    if imgData.shape[2] == 3:  ## need to make alpha channel (even if alpha==False; QImage requires 32 bpp)
        if copy is True:
            d2 = np.empty(imgData.shape[:2] + (4,), dtype=imgData.dtype)
            d2[:,:,:3] = imgData
            d2[:,:,3] = 255
            imgData = d2
            copied = True
        else:
            raise Exception('Array has only 3 channels; cannot make QImage without copying.')
    
    if alpha:
        imgFormat = QtGui.QImage.Format_ARGB32
    else:
        imgFormat = QtGui.QImage.Format_RGB32
        
    if transpose:
        imgData = imgData.transpose((1, 0, 2))  ## QImage expects the row/column order to be opposite

    if not imgData.flags['C_CONTIGUOUS']:
        if copy is False:
            extra = ' (try setting transpose=False)' if transpose else ''
            raise Exception('Array is not contiguous; cannot make QImage without copying.'+extra)
        imgData = np.ascontiguousarray(imgData)
        copied = True
        
    if copy is True and copied is False:
        imgData = imgData.copy()
        
    img = QImage(imgData.ctypes.data, imgData.shape[1], imgData.shape[0], imgFormat)            
    img.data = imgData
    return img
   

class GraphicsItem(QGraphicsItem):  
    def __init__(self, coordLabel, meanLabel): 
        super(GraphicsItem, self).__init__()
        self.pixelArray = readDICOM_Image.returnPixelArray('IM_0001').copy()
        minValue, maxValue = quickMinMax(self.pixelArray)
        self.contrast = maxValue - minValue
        self.intensity = minValue + (maxValue - minValue)/2
        imgData, alpha = makeARGB(data=self.pixelArray, levels=[minValue, maxValue])
        self.qimage = makeQImage(imgData, alpha)
        self.pixMap = QPixmap.fromImage(self.qimage)
        self.width = float(self.pixMap.width())
        self.height = float(self.pixMap.height())
        self.last_x, self.last_y = None, None
        self.start_x = None
        self.start_y = None
        self.coordLabel = coordLabel
        self.meanLabel = meanLabel
        self.pathCoordsList = []
        self.setAcceptHoverEvents(True)
        self.mask = None
        self.drawRoi = True
        self.pixelColour = None
        self.pixelValue = None


    def paint(self, painter, option, widget):
        painter.setOpacity(1)
        painter.drawPixmap(0,0, self.width, self.height, self.pixMap)
        

    def boundingRect(self):  
        return QRectF(0,0,self.width, self.height)


    def hoverMoveEvent(self, event):
        if self.isUnderMouse():
            xCoord = int(event.pos().x())
            yCoord = int(event.pos().y())
            #qimage = self.pixMap.toImage()
            self.pixelColour = self.qimage.pixelColor(xCoord,  yCoord ).getRgb()[:-1]
            self.pixelValue = self.qimage.pixelColor(xCoord,  yCoord ).value()
            self.coordLabel.setText("Pixel value {}, pixel colour {} @ X:{}, Y:{}".format(self.pixelValue, 
                                                                                      self.pixelColour,
                                                                                      xCoord, 
                                                                                      yCoord))


    def mouseMoveEvent(self, event):
        if self.drawRoi:
            if self.last_x is None: # First event.
                self.last_x = (event.pos()).x()
                self.last_y = (event.pos()).y()
                self.start_x = int(self.last_x)
                self.start_y = int(self.last_y)
                return #  Ignore the first time.
            self.myPainter = QtGui.QPainter(self.pixMap)
            p = self.myPainter.pen()
            p.setWidth(1) # 1 pixel
            p.setColor(QtGui.QColor("#FF0000")) #red
            self.myPainter.setPen(p)
            #Draws a line from (x1 , y1 ) to (x2 , y2 ).
            xCoord = event.pos().x()
            yCoord = event.pos().y()
            self.myPainter.drawLine(self.last_x, self.last_y, xCoord, yCoord)
            self.myPainter.end() 
            #The pixmap has changed (it was drawn on), so update it
            #back to the original image
            self.qimage =  self.pixMap.toImage()
            self.update()

            # Update the origin for next time.
            self.last_x = xCoord
            self.last_y = yCoord
            self.pathCoordsList.append([self.last_x, self.last_y])
        

    def mouseReleaseEvent(self, event):
        if self.drawRoi:
            if  (self.last_x != None and self.start_x != None 
                 and self.last_y != None and self.start_y != None):
                if int(self.last_x) == self.start_x and int(self.last_y) == self.start_y:
                    #free hand drawn ROI is closed, so no further action needed
                    pass
                else:
                    #free hand drawn ROI is not closed, so need to draw a
                    #straight line from the coordinates of its start to
                    #the coordinates of its last point
                    self.myPainter = QtGui.QPainter(self.pixMap)
                    p = self.myPainter.pen()
                    p.setWidth(1) #1 pixel
                    p.setColor(QtGui.QColor("#FF0000")) #red
                    self.myPainter.setPen(p)
                    #self.myPainter.setRenderHint(QtGui.QPainter.Antialiasing)
                    self.myPainter.drawLine(self.last_x, self.last_y, self.start_x, self.start_y)
                    self.myPainter.end()
                    self.qimage =  self.pixMap.toImage()
                    self.update()
                self.getMask(self.pathCoordsList)
                listCoords = self.getListRoiInnerPoints(self.mask)
                self.fillFreeHandRoi(listCoords)
                self.start_x = None 
                self.start_y = None
                self.last_x = None
                self.last_y = None
                self.pathCoordsList = []


    def fillFreeHandRoi(self, listCoords):
        for coords in listCoords:
            #x = coords[0]
            #y = coords[1]
            x = coords[1]
            y = coords[0]
            pixelColour = self.qimage.pixel(x, y) 
            pixelRGB =  QtGui.QColor(pixelColour).getRgb()
            redVal = pixelRGB[0]
            greenVal = pixelRGB[1]
            blueVal = pixelRGB[2]
            if greenVal == 255 and blueVal == 255:
                #This pixel would be white if red channel set to 255
                #so set the green and blue channels to zero
                greenVal = blueVal = 0
            value = qRgb(255, greenVal, blueVal)
            self.qimage.setPixel(x, y, value)
            #convert QImage to QPixmap to be able to update image
            #with filled ROI
            self.pixMap = QPixmap.fromImage(self.qimage)
            self.update()


    def getRoiMeanAndStd(self):
        mean = round(np.mean(np.extract(self.mask, self.pixelArray)), 3)
        std = round(np.std(np.extract(self.mask, self.pixelArray)), 3)
        return mean, std


    def getListRoiInnerPoints(self, mask):
        #result = np.nonzero(self.mask)
        result = np.where(mask == True)
        return list(zip(result[0], result[1]))


    def getMask(self, roiLineCoords):
            ny, nx = np.shape(self.pixelArray)
            #print("roiLineCoords ={}".format(roiLineCoords))
            # Create vertex coordinates for each grid cell...
            # (<0,0> is at the top left of the grid in this system)
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T #points = every [x,y] pair within the original image
        
            #print("roiLineCoords={}".format(roiLineCoords))
            roiPath = MplPath(roiLineCoords)
            #print("roiPath={}".format(roiPath))
            self.mask = roiPath.contains_points(points).reshape((ny, nx))
            

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

    def returnGraphicsItem(self):
        return self.graphicsItem

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
        rect = QRectF(self.graphicsItem.pixMap.rect())
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
        self.spinBoxIntensity = QDoubleSpinBox()
        self.spinBoxContrast = QDoubleSpinBox()

        self.graphicsView = graphicsView(self.coordsLabel, self.meanLabel)
        
        lblIntensity = QLabel("Centre (Intensity)")
        lblContrast = QLabel("Width (Contrast)")
        lblIntensity.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lblContrast.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
        self.spinBoxIntensity.setMinimum(-100000.00)
        self.spinBoxContrast.setMinimum(-100000.00)
        self.spinBoxIntensity.setMaximum(1000000000.00)
        self.spinBoxContrast.setMaximum(1000000000.00)
        self.spinBoxIntensity.setWrapping(True)
        self.spinBoxContrast.setWrapping(True)
        self.spinBoxIntensity.setValue(self.graphicsView.graphicsItem.intensity)
        self.spinBoxContrast.setValue(self.graphicsView.graphicsItem.contrast)
        gridLayoutLevels = QGridLayout()
        gridLayoutLevels.addWidget(lblIntensity, 0,0)
        gridLayoutLevels.addWidget(self.spinBoxIntensity, 0, 1)
        gridLayoutLevels.addWidget(lblContrast, 0,2)
        gridLayoutLevels.addWidget(self.spinBoxContrast, 0,3)

        self.spinBoxIntensity.valueChanged.connect(lambda: self.updateImageLevels(self.spinBoxIntensity, self.spinBoxContrast))
        self.spinBoxContrast.valueChanged.connect(lambda: self.updateImageLevels(self.spinBoxIntensity, self.spinBoxContrast))
        #self.graphicsView.graphicsItem
        self.centralwidget.layout().addWidget(self.graphicsView)
        self.centralwidget.layout().addLayout(gridLayoutLevels)
        self.centralwidget.layout().addWidget(self.coordsLabel)
        self.centralwidget.layout().addWidget(self.meanLabel)


    def updateImageLevels(self, spinBoxIntensity, spinBoxContrast):
        """When the contrast and intensity values are adjusted using the spinboxes, 
        this function sets the corresponding values in the image being viewed. 
    
        Input Parmeters
        ***************
            self - an object reference to the WEASEL interface.
            imv - pyqtGraph imageView widget
            spinBoxIntensity - name of the spinbox widget that displays/sets image intensity.
            spinBoxContrast - name of the spinbox widget that displays/sets image contrast.
        """
        try:
            intensityValue = spinBoxIntensity.value()
            contrastValue = spinBoxContrast.value()
            halfWidth = contrastValue/2
            minimumValue = intensityValue - halfWidth
            maximumValue = intensityValue + halfWidth

            imageItemPointer = self.graphicsView.graphicsItem
            imgData, alpha = makeARGB(data=imageItemPointer.pixelArray, levels=[minimumValue,  maximumValue])
            imageItemPointer.qimage = makeQImage(imgData, alpha)
            imageItemPointer.pixmap = QPixmap.fromImage(imageItemPointer.qimage)
            imageItemPointer.update()
        except Exception as e:
            print('Error in updateImageLevels: ' + str(e))
       


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Example()
    w.show()
    sys.exit(app.exec_())
