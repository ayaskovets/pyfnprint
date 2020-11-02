from glob import glob
from os import path

import cv2 as cv
import numpy as np


class FingerprintImage:
    """
    A fingerprint image wrapper

    Arguments:
        id     - unique fingerprint id
        number - image (scanner) identifier
    """
    def __init__(self,
                 id: int,
                 number: int,
                 fppath: str):
        self.id = id
        self.number = number
        self.fppath = fppath

    def getData(self,
                colorspace: int=cv.IMREAD_GRAYSCALE,
                astype: int=np.uint8):
        return cv.imread(self.fppath, colorspace).astype(astype)

def readOne(fppath: str):
    """
    Read the single fingerprint image

    Arguments:
        fppath - path to the image

    Return FingerprintImage object
    """
    if path.exists(fppath) and path.isfile(fppath):
        spl = path.splitext(path.basename(fppath))[0].split('_')
        while '' in spl:
            spl.remove('')

        if len(spl) < 2:
            raise Exception(fppath + ' must be \'[fingerid]_[imageid]\'.*')

        id, number = int(spl[0]), int(spl[1])

        return FingerprintImage(id, number, fppath)
    else:
        raise Exception(fppath + ' is not found!')


def readFolder(wildcard: str):
    """
    Read folder using the wildcard match

    Arguments:
        wildcard - path pattern to match

    Return list of FingerprintImage objects
    """
    fnprints = []
    for file in glob(wildcard):
        fnprints.append(readOne(file))
    return fnprints
