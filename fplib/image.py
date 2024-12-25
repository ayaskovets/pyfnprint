#
# Copyright (c) 2024, Andrei Yaskovets
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List
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
                 file_path: str):
        self.id = id
        self.number = number
        self.file_path = file_path

    def __eq__(self, other):
        return self.id == other.id and self.number == other.number

    def __lt__(self, other):
        if self.id == other.id:
            return self.number < other.number
        else:
            return self.id < other.id

    def getData(self,
                colorspace: int=cv.IMREAD_GRAYSCALE,
                astype: int=np.uint8):
        return cv.imread(self.file_path, colorspace).astype(astype)

def readOne(file_path: str) -> FingerprintImage:
    """
    Read the single fingerprint image

    Arguments:
        file_path - path to the image

    Return FingerprintImage object
    """
    if path.exists(file_path) and path.isfile(file_path):
        spl = path.splitext(path.basename(file_path))[0].split('_')
        while '' in spl:
            spl.remove('')

        if len(spl) < 2:
            raise Exception(file_path + ' must be \'[fingerid]_[imageid]\'.*')

        id, number = int(spl[0]), int(spl[1])

        return FingerprintImage(id, number, file_path)
    else:
        raise Exception(file_path + ' is not found!')


def readFolder(wildcard: str) -> List[FingerprintImage]:
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
