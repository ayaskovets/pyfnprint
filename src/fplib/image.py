from glob import glob
from os import path

from cv2 import imread


class FingerprintImage:
    """
    A fingerprint image wrapper

    Arguments:
        id     - unique fingerprint id
        hand   - hand identifier
        finger - finger number
        fppath - path to the fingerprint image
    """
    def __init__(self,
                 id: int,
                 hand: int,
                 finger: int,
                 fppath: str):
        self.id = id
        self.hand = hand
        self.finger = finger
        self.fppath = fppath

    def getData(self,
                colorspace: int,
                astype: int):
        return imread(self.fppath, colorspace).astype(astype)

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

        if len(spl) < 1:
            raise Exception(fppath + ' must start with [id_]')

        id = int(spl[0])
        hand = (0 if len(spl) < 2 else int(spl[1]))
        finger = (0 if len(spl) < 3 else int(spl[2]))

        return FingerprintImage(id, hand, finger, fppath)
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
