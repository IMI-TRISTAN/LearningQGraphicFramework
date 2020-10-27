import os
import struct
import numpy as np
import pydicom
from scipy.spatial.transform import Rotation


def returnPixelArray(imagePath):
    """This method reads the DICOM file in imagePath and returns the Image/Pixel array"""
    try:
        if os.path.exists(imagePath):
            dataset = getDicomDataset(imagePath)
            pixelArray = getPixelArray(dataset)
            return pixelArray
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.returnPixelArray: ' + str(e))


def returnAffineArray(imagePath):
    """This method reads the DICOM file in imagePath and returns the Affine/Orientation matrix"""
    try:
        if os.path.exists(imagePath):
            dataset = getDicomDataset(imagePath)
            affineArray = getAffineArray(dataset)
            return affineArray
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.returnAffineArray: ' + str(e))


def returnSeriesPixelArray(imagePathList):
    """This method reads the DICOM files in imagePathList and 
    returns a list where each element is a DICOM Dataset object/class"""
    try:
        datasetList = getSeriesDicomDataset(imagePathList)
        imageList = [getPixelArray(dataset) for dataset in datasetList]
        if imageList:
            volumeArray = np.array(imageList)
            del imageList
            return volumeArray
        else:
            del imageList
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.getDicomDataset: ' + str(e))


def getMultiframeBySlices(dataset, sliceList=None, sort=False):
    try:
        if any(hasattr(dataset, attr) for attr in ['PixelData', 'FloatPixelData', 'DoubleFloatPixelData']) and hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
            originalArray = getPixelArray(dataset)
            numberSlices = dataset[0x20011018].value
            sortedArray = list()
            sortedSlicesList = list()
            if sliceList is None:
                numberFrames = dataset.NumberOfFrames
            else:
                numberFrames = len(sliceList)
            
            if (sort == True):
                if '2D' in dataset.MRAcquisitionType:
                    for start in range(int(numberFrames/numberSlices)):
                        sortedSlicesList.append(list(range(start, numberFrames, int(numberFrames/numberSlices))))
                    sortedSlicesList = np.array(sortedSlicesList).flatten()
                elif '3D' in dataset.MRAcquisitionType:
                    sortedSlicesList = list(range(numberFrames))
            elif (sort == False) and ((type(sliceList) is list) or (type(sliceList) is np.array)):
                sortedSlicesList = sliceList
            else:
                sortedSlicesList = list(range(numberFrames))
                
            for index in sortedSlicesList:
                sortedArray.append(np.squeeze(originalArray[index, ...]))
            sortedArray = np.array(sortedArray)
            return sortedArray, sortedSlicesList, numberSlices
        else:
            return None, None, None
    except Exception as e:
        print('Error in function readDICOM_Image.getMultiframeBySlices: ' + str(e))


def getSeriesTagValues(imagePathList, dicomTag):
    """This method reads the DICOM files in imagePathList and returns the list of values in the given DICOM tag
        Outputs are : attributeList, numAttribute
        The output attributeList may have repeated values. 
        Removing these repetitions will be up to the developer of the specific algorithm
    """
    try:
        if os.path.exists(imagePathList[0]):
            datasetList = getSeriesDicomDataset(imagePathList)
            if not hasattr(datasetList[0], 'PerFrameFunctionalGroupsSequence'):
                # This is not for Enhanced MRI. Only Classic DICOM
                if isinstance(dicomTag, str):
                    attributeList = [dataset.data_element(dicomTag).value for dataset in datasetList]
                else:
                    attributeList = [dataset[hex(dicomTag)].value for dataset in datasetList]
                numAttribute = len(np.unique(attributeList))
 
                del datasetList
                return attributeList, numAttribute
        else:
            return None, None
    except Exception as e:
        print('Error in function readDICOM_Image.getSeriesTagValues: ' + str(e))


def sortSequenceByTag(imagePathList, dicomTag):
    """This method reads the DICOM files in imagePathList and sorts the list according to the given DICOM tag
        Outputs are : sortedSequencePath, attributeList, numAttribute
        The output attributeList may have repeated values. 
        Removing these repetitions will be up to the developer of the specific algorithm
    """
    try:
        if os.path.exists(imagePathList[0]):
            dataset = getDicomDataset(imagePathList[0])
            if not hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
                # This is not for Enhanced MRI. Only Classic DICOM
                attributeList, numAttribute = getSeriesTagValues(imagePathList, dicomTag)
                attributeListUnique = sorted(np.unique(attributeList))
                indicesSorted = list()
                for i in range(numAttribute):
                    indices = [index for index, value in enumerate(attributeList) if value == attributeListUnique[i]]
                    indicesSorted.extend(indices)
                sortedSequencePath = [imagePathList[index] for index in indicesSorted]
                attributeListSorted = [attributeList[index] for index in indicesSorted]
                del dataset, attributeList, attributeListUnique
                return sortedSequencePath, attributeListSorted, numAttribute, indicesSorted
        else:
            return None, None, None
    except Exception as e:
        print('Error in function readDICOM_Image.sortSequenceByTag: ' + str(e))


def getSeriesDicomDataset(imagePathList):
    """This method reads the DICOM files in imagePathList and 
    returns a list where each element is a DICOM Dataset object/class"""
    try:
        datasetList = [getDicomDataset(imagePath) for imagePath in imagePathList if getDicomDataset(imagePath) is not None]
        if datasetList:
            return datasetList
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.getDicomDataset: ' + str(e))


def getDicomDataset(imagePath):
    """This method reads the DICOM file in imagePath and returns the DICOM Dataset object/class"""
    try:
        if os.path.exists(imagePath):
            return pydicom.dcmread(imagePath)
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.getDicomDataset: ' + str(e))


def getPixelArray(dataset):
    """This method reads the DICOM Dataset object/class and returns the Image/Pixel array"""
    try:
        if any(hasattr(dataset, attr) for attr in ['PixelData', 'FloatPixelData', 'DoubleFloatPixelData']):
            if hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
                imageList = list()
                originalArray = dataset.pixel_array.astype(np.float32)
                if len(np.shape(originalArray))==2:
                    slope = float(getattr(dataset.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0], 'RescaleSlope', 1)) * np.ones(originalArray.shape)
                    intercept = float(getattr(dataset.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0], 'RescaleIntercept', 0)) * np.ones(originalArray.shape)
                    pixelArray = np.transpose(originalArray * slope + intercept)
                else:
                    for index in range(np.shape(originalArray)[0]):
                        sliceArray = np.squeeze(originalArray[index, ...])
                        slope = float(getattr(dataset.PerFrameFunctionalGroupsSequence[index].PixelValueTransformationSequence[0], 'RescaleSlope', 1)) * np.ones(sliceArray.shape)
                        intercept = float(getattr(dataset.PerFrameFunctionalGroupsSequence[index].PixelValueTransformationSequence[0], 'RescaleIntercept', 0)) * np.ones(sliceArray.shape)
                        tempArray = np.transpose(sliceArray * slope + intercept)
                        imageList.append(tempArray)
                    pixelArray = np.array(imageList)
                    del sliceArray, tempArray, index
                del originalArray, imageList
            else:
                slope = float(getattr(dataset, 'RescaleSlope', 1)) * np.ones(dataset.pixel_array.shape)
                intercept = float(getattr(dataset, 'RescaleIntercept', 0)) * np.ones(dataset.pixel_array.shape)
                pixelArray = np.transpose(dataset.pixel_array.astype(np.float32) * slope + intercept)
            del slope, intercept
            return pixelArray
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.getPixelArray: ' + str(e))


def getAffineArray(dataset):
    """This method reads the DICOM Dataset object/class and returns the Affine/Orientation matrix"""
    try:
        if any(hasattr(dataset, attr) for attr in ['PixelData', 'FloatPixelData', 'DoubleFloatPixelData']):
            if hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
                affineList = list()
                for index in range(len(dataset.PerFrameFunctionalGroupsSequence)):
                    image_orientation = dataset.PerFrameFunctionalGroupsSequence[index].PlaneOrientationSequence[0].ImageOrientationPatient
                    row_cosine = np.array(image_orientation[:3])
                    column_cosine = np.array(image_orientation[3:])
                    slice_cosine = np.cross(row_cosine, column_cosine)
                    row_spacing, column_spacing = dataset.PerFrameFunctionalGroupsSequence[index].PixelMeasuresSequence[0].PixelSpacing
                    slice_spacing = dataset.PerFrameFunctionalGroupsSequence[index].PixelMeasuresSequence[0].SpacingBetweenSlices

                    affine = np.identity(4, dtype=np.float32)
                    affine[:3, 0] = row_cosine * column_spacing
                    affine[:3, 1] = column_cosine * row_spacing
                    affine[:3, 2] = slice_cosine * slice_spacing
                    affine[:3, 3] = dataset.PerFrameFunctionalGroupsSequence[index].PlanePositionSequence[0].ImagePositionPatient
                    affineList.append(affine)
                affine = np.squeeze(np.array(affineList))
            else:
                image_orientation = dataset.ImageOrientationPatient
                row_cosine = np.array(image_orientation[:3])
                column_cosine = np.array(image_orientation[3:])
                slice_cosine = np.cross(row_cosine, column_cosine)
                row_spacing, column_spacing = dataset.PixelSpacing
                slice_spacing = dataset.SliceThickness

                affine = np.identity(4, dtype=np.float32)
                affine[:3, 0] = row_cosine * column_spacing
                affine[:3, 1] = column_cosine * row_spacing
                affine[:3, 2] = slice_cosine * slice_spacing
                affine[:3, 3] = dataset.ImagePositionPatient
            return affine
        else:
            return None
    except Exception as e:
        print('Error in function readDICOM_Image.getAffineArray: ' + str(e))


def getColourmap(imagePath):
    """This method reads the DICOM file in imagePath and returns the colourmap if there's any"""
    try:
        dataset = getDicomDataset(imagePath)
        if hasattr(dataset, 'ContentLabel'):
            if dataset.PhotometricInterpretation == 'PALETTE COLOR':
                colourmapName = dataset.ContentLabel
            elif 'MONOCHROME' in dataset.PhotometricInterpretation:
                colourmapName = 'gray'
            else:
                colourmapName = None
        else:
            colourmapName = None
        if len(dataset.dir("PaletteColor"))>=3 and dataset.PhotometricInterpretation == 'PALETTE COLOR':
            if colourmapName is None:
                colourmapName = 'custom'
            redColour = list(dataset.RedPaletteColorLookupTableData)
            greenColour = list(dataset.GreenPaletteColorLookupTableData)
            blueColour = list(dataset.BluePaletteColorLookupTableData)
            redLut = list(struct.unpack('<' + ('H' * dataset.RedPaletteColorLookupTableDescriptor[0]), bytearray(redColour)))
            greenLut = list(struct.unpack('<' + ('H' * dataset.GreenPaletteColorLookupTableDescriptor[0]), bytearray(greenColour)))
            blueLut = list(struct.unpack('<' + ('H' * dataset.BluePaletteColorLookupTableDescriptor[0]), bytearray(blueColour)))
            colours = np.transpose([redLut, greenLut, blueLut])
            normaliseFactor = int(np.power(2, dataset.RedPaletteColorLookupTableDescriptor[2]))
            # Fast ColourTable loading
            colourTable = np.around(colours/normaliseFactor, decimals = 2)
            indexes = np.unique(colourTable, axis=0, return_index=True)[1]
            lut = [colourTable[index].tolist() for index in sorted(indexes)]
            # Full / Complete Colourmap - takes 20 seconds to load each image
            # lut = (colours/normaliseFactor).tolist()
        else:
            lut = None
        return colourmapName, lut
    except Exception as e:
        print('Error in function readDICOM_Image.getColourmap: ' + str(e))
    

def checkImageType(dataset):
    """This method reads the DICOM Dataset object/class and returns if it is a Magnitude, Phase, Real or Imaginary image or None"""
    try:
        flagMagnitude = False
        flagPhase = False
        flagReal = False
        flagImaginary = False
        flagMap = False
        mapsList = ['ADC', 'FA', 'B0', 'T1', 'T2', 'T2_STAR', 'B0 MAP', 'T1 MAP', 'T2 MAP', 'T2_STAR MAP', 'MAP', 'FIELD_MAP']
        if hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
            if hasattr(dataset.MRImageFrameTypeSequence[0], 'FrameType'):
                if set(['M', 'MAGNITUDE']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagMagnitude = True
                elif set(['P', 'PHASE', 'B0', 'FIELD_MAP']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagPhase = True
                elif set(['R', 'REAL']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagReal = True
                elif set(['I', 'IMAGINARY']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagImaginary = True
                elif set(mapsList).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagMap = list(set(mapsList).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)))[0]
            elif hasattr(dataset.MRImageFrameTypeSequence[0], 'ComplexImageComponent'):
                if set(['M', 'MAGNITUDE']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagMagnitude = True
                elif set(['P', 'PHASE']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagPhase = True
                elif set(['R', 'REAL']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagReal = True
                elif set(['I', 'IMAGINARY']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagImaginary = True
                elif set(mapsList).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagMap = list(set(mapsList).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)))[0]
        else:
            try: private_ge = dataset[0x0043102f]
            except: private_ge = None
            if private_ge is not None: #MAG = 0; PHASE = 1; REAL = 2; IMAG = 3; # RawDataType_ImageType in GE - '0x0043102f'
                try:
                    if struct.unpack('h', private_ge.value)[0] == 0:
                        flagMagnitude = True
                    elif struct.unpack('h', private_ge.value)[0] == 1:
                        flagPhase = True
                    elif struct.unpack('h', private_ge.value)[0] == 2:
                        flagReal = True
                    elif struct.unpack('h', private_ge.value)[0] == 3:
                        flagImaginary = True
                except:
                    if private_ge.value == 0:
                        flagMagnitude = True
                    elif private_ge.value == 1:
                        flagPhase = True
                    elif private_ge.value == 2:
                        flagReal = True
                    elif private_ge.value == 3:
                        flagImaginary = True
            if hasattr(dataset, 'ImageType'):
                if set(['M', 'MAGNITUDE']).intersection(set(dataset.ImageType)):
                    flagMagnitude = True
                elif set(['P', 'PHASE']).intersection(set(dataset.ImageType)):# or ('B0' in dataset.ImageType) or ('FIELD_MAP' in dataset.ImageType):
                    flagPhase = True
                elif set(['R', 'REAL']).intersection(set(dataset.ImageType)):
                    flagReal = True
                elif set(['I', 'IMAGINARY']).intersection(set(dataset.ImageType)):
                    flagImaginary = True
                elif set(mapsList).intersection(set(dataset.ImageType)):
                    flagMap = list(set(mapsList).intersection(set(dataset.ImageType)))[0]
        return flagMagnitude, flagPhase, flagReal, flagImaginary, flagMap
    except Exception as e:
        print('Error in function readDICOM_Image.checkImageType: ' + str(e))


def checkAcquisitionType(dataset):
    """This method reads the DICOM Dataset object/class and returns if it is a Water, Fat, In-Phase, Out-phase image or None"""
    try:
        flagWater = False
        flagFat = False
        flagInPhase = False
        flagOutPhase = False
        if hasattr(dataset, 'PerFrameFunctionalGroupsSequence'):
            if hasattr(dataset.MRImageFrameTypeSequence[0], 'FrameType'):
                if set(['W', 'WATER']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagWater = True
                elif set(['F', 'FAT']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagFat = True
                elif set(['IP', 'IN_PHASE']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagInPhase = True
                elif set(['OP', 'OUT_PHASE']).intersection(set(dataset.MRImageFrameTypeSequence[0].FrameType)):
                    flagOutPhase = True
            elif hasattr(dataset.MRImageFrameTypeSequence[0], 'ComplexImageComponent'):
                if set(['W', 'WATER']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagWater = True
                elif set(['F', 'FAT']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagFat = True
                elif set(['IP', 'IN_PHASE']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagInPhase = True
                elif set(['OP', 'OUT_PHASE']).intersection(set(dataset.MRImageFrameTypeSequence[0].ComplexImageComponent)):
                    flagOutPhase = True
        else:
            if hasattr(dataset, 'ImageType'):
                if set(['W', 'WATER']).intersection(set(dataset.ImageType)):
                    flagWater = True
                elif set(['F', 'FAT']).intersection(set(dataset.ImageType)):# or ('B0' in dataset.ImageType) or ('FIELD_MAP' in dataset.ImageType):
                    flagFat = True
                elif set(['IP', 'IN_PHASE']).intersection(set(dataset.ImageType)):
                    flagInPhase = True
                elif set(['OP', 'OUT_PHASE']).intersection(set(dataset.ImageType)):
                    flagOutPhase = True
        return flagWater, flagFat, flagInPhase, flagOutPhase
    except Exception as e:
        print('Error in function readDICOM_Image.checkAcquisitionType: ' + str(e))
