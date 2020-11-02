import cv2 as cv
import numpy as np


def binarize(image: np.array,
             method: str,
             threshold: int=160,
             blksize: int=9,
             c: int=0):
    """
    Binarize grayscale image

    Arguments:
        image     - image to binarize
        method    - algorithm, one of ['gauss', 'global', 'mean', 'otsu']
        threshold - threshold value for non-adaptive methods. Used in \
['global', 'otsu']
        blksize   - window side in pixels. Used in ['global', 'otsu']
        c         - constant to subtract from mean for adaptive methods. Used \
in ['gauss', 'mean']

    Return the binarized image
    """
    if method == 'gauss':
        return _binarize_gauss(image, blksize, c)
    elif method == 'global':
        return _binarize_global(image, threshold)
    elif method == 'mean':
        return _binarize_mean(image, blksize, c)
    elif method == 'otsu':
        return _binarize_otsu(image, threshold)
    else:
        raise Exception(method + ' binarization method is not supported!')


def _binarize_mean(img: np.array,
                   blksize: int,
                   c: int):
    """ Adaptive mean thresholding """
    bin = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY, blksize, c)
    return bin


def _binarize_gauss(img: np.array,
                    blksize: int,
                    c: int):
    """ Adaptive gaussian thresholding """
    bin = cv.adaptiveThreshold(img, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, blksize, c)
    return bin


def _binarize_global(img: np.array,
                     threshold: int):
    """ Simple global value thresholding """
    _, bin = cv.threshold(img, threshold, 1, cv.THRESH_BINARY)
    return bin


def _binarize_otsu(img: np.array,
                   threshold: int):
    """ The OTSU binarization method """
    _, bin = cv.threshold(img, threshold, 1, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return bin
