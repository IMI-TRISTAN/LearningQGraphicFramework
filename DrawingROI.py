from PyQt5 import QtGui
from PyQt5.QtCore import (QRectF, Qt)
from PyQt5.QtGui import (QPainter, QPixmap, QColor, QImage)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QVBoxLayout,
                             QGraphicsObject, QGraphicsView, QWidget,
                             QGraphicsScene, QGraphicsItem, QLabel, 
                             QDoubleSpinBox, QGridLayout, QComboBox)
import numpy as np
from numpy import nanmin, nanmax
from PIL import Image
from numpy import asarray
from matplotlib.path import Path as MplPath
import readDICOM_Image as readDICOM_Image
import scipy
from matplotlib import cm

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

#List of colour tables supported by matplotlib
listColours = ['gray', 'cividis',  'magma', 'plasma', 'viridis', 
             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar', 'custom']

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
    if dtype is None:
        dtype = data.dtype
    else:
        dtype = np.dtype(dtype)
    
    try:
        if not getConfigOption('useWeave'):
            raise Exception('Weave is disabled; falling back to slower version.')
        try:
            import scipy.weave
        except ImportError:
            raise Exception('scipy.weave is not importable; falling back to slower version.')
        
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
        self.mypixmap = QPixmap.fromImage(self.qimage)
        #self.mypixmap = QPixmap("KarlaOnMyShoulder.jpg")
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
        #painter.drawImage(QRectF(0,0,self.width,self.height), self.qimage)
        #p.drawImage(QtCore.QRectF(0,0,self.image.shape[0],self.image.shape[1]), self.qimage)
        

    def boundingRect(self):  
        return QRectF(0,0,self.width, self.height)


    #def updateImageLevels(self, spinBoxIntensity, spinBoxContrast):
    #    """When the contrast and intensity values are adjusted using the spinboxes, 
    #    this function sets the corresponding values in the image being viewed. 
    
    #    Input Parmeters
    #    ***************
    #        self - an object reference to the WEASEL interface.
    #        imv - pyqtGraph imageView widget
    #        spinBoxIntensity - name of the spinbox widget that displays/sets image intensity.
    #        spinBoxContrast - name of the spinbox widget that displays/sets image contrast.
    #    """
    #try:
    #    intensityValue = spinBoxIntensity.value()
    #    contrastValue = spinBoxContrast.value()
    #    print('intensityValue {} contrastValue {}'.format(intensityValue, contrastValue))
    #    halfWidth = contrastValue/2
    #    minimumValue = intensityValue - halfWidth
    #    maximumValue = intensityValue + halfWidth
    #    imgData, alpha = makeARGB(data=self.pixelArray, levels=[minimumValue,  maximumValue])
    #    self.qimage = makeQImage(imgData, alpha)
    #    self.mypixmap = QPixmap.fromImage(self.qimage)
    #    self.update()
    #except Exception as e:
    #    print('Error in updateImageLevels: ' + str(e))


    def hoverMoveEvent(self, event):
        if self.isUnderMouse():
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
        #self.get_mean_and_std( self.mainList, "KarlaOnMyShoulder.jpg")
        #self.mainList = []


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

        self.cmbColours = QComboBox()
        self.cmbColours.blockSignals(True)
        self.cmbColours.addItems(listColours)
        self.cmbColours.setCurrentIndex(0)
        self.cmbColours.blockSignals(False)
        

        self.spinBoxIntensity.valueChanged.connect(lambda: self.updateImageLevels(self.spinBoxIntensity, self.spinBoxContrast))
        self.spinBoxContrast.valueChanged.connect(lambda: self.updateImageLevels(self.spinBoxIntensity, self.spinBoxContrast))
        #self.graphicsView.graphicsItem
        self.centralwidget.layout().addWidget(self.graphicsView)
        self.centralwidget.layout().addLayout(gridLayoutLevels)
        #self.centralwidget.layout().addWidget(self.cmbColours)
        self.centralwidget.layout().addWidget(self.coordsLabel)
        self.centralwidget.layout().addWidget(self.meanLabel)
        #self.cmbColours.currentIndexChanged.connect(lambda: self.setColourMap(self.cmbColours))

    def setColourMap(self, colourList):
        """This function converts a matplotlib colour map into
        a colour map that can be used by the pyqtGraph imageView widget.
    
        Input Parmeters
        ***************
            colourTable - name of the colour map
            imv - name of the imageView widget
            cmbColours - name of the dropdown lists of colour map names
            lut - name of the look up table containing raw colour data
        """

        try:
            colourTable = colourList.currentText()
            if colourTable == None:
                colourTable = 'gray'

           # if cmbColours:
            #    displayColourTableInComboBox(cmbColours, colourTable)   
        
            if colourTable == 'custom':
                colors = lut
            elif colourTable == 'gray':
                colors = [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
            else:
                cmMap = cm.get_cmap(colourTable)
                colourClassName = cmMap.__class__.__name__
                if colourClassName == 'ListedColormap':
                    colors = cmMap.colors
                elif colourClassName == 'LinearSegmentedColormap':
                    colors = cmMap(np.linspace(0, 1))
          
            positions = np.linspace(0, 1, len(colors))

            imageItemPointer = self.graphicsView.graphicsItem
            imgData, alpha = makeARGB(data=imageItemPointer.pixelArray, lut=colors)
            imageItemPointer.qimage = makeQImage(imgData, alpha)
            imageItemPointer.mypixmap = QPixmap.fromImage(imageItemPointer.qimage)
            imageItemPointer.update()
            #pgMap = pg.ColorMap(positions, colors)
            #imv.setColorMap(pgMap)        
        except Exception as e:
            print('Error in setPgColourMap: ' + str(e))


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
            imageItemPointer.mypixmap = QPixmap.fromImage(imageItemPointer.qimage)
            imageItemPointer.update()
        except Exception as e:
            print('Error in updateImageLevels: ' + str(e))
       


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = Example()
    w.show()
    sys.exit(app.exec_())
