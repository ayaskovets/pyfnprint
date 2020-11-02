import functools
import itertools

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage.morphology import binary_hit_or_miss
from scipy.signal import argrelextrema
from skimage.morphology import skeletonize

import fplib.filter as fpfilter


verbose = False


def normalize(image: np.array,
              low: int,
              upp: int,
              dtype=np.float32):
    """
    Normalize an image

    Arguments:
        image - image to normalize
        low   - lower bound of the normalized image
        upp   - upper bound of the normalized image
        dtype - type to convert the resulting image to

    Returns the normalized image
    """
    return cv.normalize(image, None, low, upp, cv.NORM_MINMAX).astype(dtype)


def standartize(image: np.array):
    """
    Apply standartization to the image

    Arguments:
        image - image to standartize

    Return the standartized image
    """
    return (image - np.mean(image)) / np.std(image)


def equalize(image: np.array):
    """
    Equalize histogram of an image

    Arguments:
        image - image to apply histogram equalization method to

    Return the image with equalized histogram
    """
    return cv.equalizeHist(image)


def resize(image: np.array,
           width: int=None,
           height: int=None):
    """
    Change the width of an image preserving the aspect ratio if only a single \
dimension is provided

    Arguments:
        image  - image to resize
        width  - resulting width
        height - resulting height

    Return the resized image
    """
    rows, cols = image.shape

    if width and height:
        return cv.resize(image, (width, height))
    elif width:
        return cv.resize(image, (width, int(width * rows / cols)))
    elif height:
        return cv.resize(image, (int(height * cols / rows), height))
    else:
        return image


def mask(image: np.array,
         blksize: int):
    """
    Create mask separating the image from background

    Arguments:
        image   - image to generate mask for
        blksize - mask atomic block size

    Return the mask that separates the background from the fingerprint
    """
    mask = normalize(image, 0, 255, dtype=np.uint8)

    # M(image)/2 = empirical threshold value
    threshold = np.mean(image) / 2

    for y in range(0, mask.shape[1], blksize):
        for x in range(0, mask.shape[0], blksize):
            blk = mask[x:x + blksize, y:y + blksize]
            mask[x:x + blksize, y:y + blksize] = np.var(blk) > threshold

    # Find the longest contour in the mask and set everything inside it to ones
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.fillPoly(mask, pts=[max(contours, key=cv.contourArea)], color=(1, 1))

    # Make the edges of the mask smoother
    krnsize = blksize + 1 if blksize % 2 == 0 else blksize
    mask = cv.GaussianBlur(mask, (krnsize * 3, krnsize * 3), krnsize * 3)

    return mask


def orientation(image: np.array,
                grdsigma: float,
                blksigma: float,
                smtsigma: float):
    """
    Create orientation matrix of the fingerprint image

    Arguments
        image    - image to generate orientation matrix for
        grdsigma - frequency parameter for gradient filters
        blksigma - frequency parameter for smoothing filters
        smtsigma - frequency parameter for additional post-smoothing filters

    Return the orientation matrix of the image
    """
    # Find the basic gradients
    Gx = fpfilter.filter(image, 'sobelx', int(3 * grdsigma))
    Gy = fpfilter.filter(image, 'sobely', int(3 * grdsigma))

    # Smoothing
    Gxx = fpfilter.filter(Gx * Gx, 'lowpass', int(3 * blksigma), blksigma)
    Gyy = fpfilter.filter(Gy * Gy, 'lowpass', int(3 * blksigma), blksigma)
    Gxy = fpfilter.filter(Gx * Gy, 'lowpass', int(3 * blksigma), blksigma)

    sin = fpfilter.filter(2 * Gxy, 'lowpass', int(3 * smtsigma), smtsigma)
    cos = fpfilter.filter(Gxx - Gyy, 'lowpass', int(3 * smtsigma), smtsigma)

    # Clockwise rotation in radians
    orient = np.arctan2(sin, cos) / 2 + np.pi / 2
    return orient


def angles(orient: np.array,
           blksize: int):
    """
    Acquire the ridge rotation matrix from the orientation image

    Arguments
        orient  - orientation matrix of the image. Can be acquired via \
orientation() function
        blksize - block size of the returned rotation matrix

    Return the ridge rotation matrix
    """
    angles = np.zeros(orient.shape)
    for j in range(0, orient.shape[1], blksize):
        for i in range(0, orient.shape[0], blksize):
            angles[i - blksize:i + blksize, j - blksize:j + blksize] =\
                _orientblk_angle(orient, (i, j), blksize)
    return angles


def poincare(angles: np.array,
             blksize: int):
    """
    Acquire the poincare index for every block of the angles matrix

    Arguments
        angles  - ridge rotation matrix. Can be acquired via angles() function
        blksize - block size of the returned index matrix

    Return the poincare index matrix
    """
    idx = [(2,0), (1,0), (0,0), (0,1), (0,2), (1,2), (2,2), (2,1)]
    fpidx = lambda b: np.sum([b[k2] - b[k1] for k1, k2 in zip(idx, idx[1:])])

    angles_scaled = np.zeros(tuple(int(x / blksize) for x in angles.shape))
    if (blksize > 1):
        for i in range(0, angles_scaled.shape[0]):
            for j in range(0, angles_scaled.shape[1]):
                angles_scaled[i, j] = np.mean(
                    angles[i * blksize:(i + 1) * blksize,
                           j * blksize:(j + 1) * blksize])
    angles_scaled[angles_scaled < 90] += 180
    angles_scaled[angles_scaled > 90] -= 180

    pidx = np.zeros(angles.shape)
    for i in range(1, angles_scaled.shape[0] - 1):
        for j in range(1, angles_scaled.shape[1] - 1):
            blk = angles_scaled[i - 1:i + 2, j - 1:j + 2]
            pidx[(i - 1) * blksize:(i + 2) * blksize,
                 (j - 1) * blksize:(j + 2) * blksize] = fpidx(blk)
    return pidx


def angular_coherence(angles: np.array,
                      blksize: int,
                      wndsize: int):
    """
    Acquire the angular coherence for every block of the angles matrix

    Arguments
        angles  - ridge rotation matrix. Can be acquired via angles() function
        blksize - block size of the returned coherence matrix
        wndsize - size of the window to calculate coherence in

    Return the angular coherence matrix
    """
    angles_scaled = np.zeros(tuple(int(x / blksize) for x in angles.shape))
    h, w = angles_scaled.shape
    if (blksize > 1):
        for i in range(0, h):
            for j in range(0, w):
                angles_scaled[i, j] = np.mean(
                    angles[i * blksize:(i + 1) * blksize,
                           j * blksize:(j + 1) * blksize])

    hlfwnd = int(np.floor(wndsize / 2))

    angcoh = np.zeros(angles.shape)
    for i in range(0, h):
        for j in range(0, w):
            i1, i2 = np.max((0, i - hlfwnd)), np.min((i + hlfwnd + 1, h))
            j1, j2 = np.max((0, j - hlfwnd)), np.min((j + hlfwnd + 1, w))

            blk = angles_scaled[i1:i2, j1:j2]

            cossum = -1
            for val in np.nditer(blk):
                cossum += np.cos(angles_scaled[i, j] - val)

            # plt.figure()
            # print('shape =', blk.shape, 'sum =', cossum)
            # plt.imshow(blk)
            # plt.show()

            angcoh[i * blksize:(i + 1) * blksize,
                   j * blksize:(j + 1) * blksize] = cossum
    return angcoh


def _orientblk_angle(orient: np.array,
                     point: tuple,
                     blksize: int):
    """ Return angle or clockwise rotation of an orientation matrix block """
    i, j = point
    i1, i2 = np.max([i - blksize, 0]), np.min([i + blksize, orient.shape[0]])
    j1, j2 = np.max([j - blksize, 0]), np.min([j + blksize, orient.shape[1]])
    blk = orient[i1:i2, j1:j2]

    sin = np.mean(np.sin(2 * blk))
    cos = np.mean(np.cos(2 * blk))
    return np.round(np.rad2deg(np.arctan2(sin, cos) / 2), 2)


def frequency(image: np.array,
              orient: np.array,
              blksize: int):
    """
    Create frequency matrix for the image

    Arguments:
        image   - image to generate frequency matrix for
        orient  - orientation matrix of the image. Can be acquired via \
orientation() function
        blksize - block size considered by algorithm

    Return frequency estimation of the fingerprint ridges
    """
    # Inverse colors and make black = 1 and white = 0
    freq = 1 - normalize(image, 0, 1)

    for y in range(0, image.shape[1], blksize):
        for x in range(0, image.shape[0], blksize):
            blkor = orient[x:x + blksize, y:y + blksize]
            blkim = freq[x:x + blksize, y:y + blksize]
            freq[x:x + blksize, y:y + blksize] = _freq(blkim, blkor)

    return freq


def _freq(image: np.array,
          orient: np.array):
    """ Frequency estimation of an image block """
    # Average orientation in the block
    sin = np.mean(np.sin(2 * orient))
    cos = np.mean(np.cos(2 * orient))
    angle = np.arctan2(sin, cos) / 2

    # Rotate to align orientation vertically
    rot = rotate(image, np.rad2deg(angle) + 90, reshape=True)

    # Norm by a number of pixels that belong to the image in a column
    nonzero = np.count_nonzero(rot, axis=0)
    nonzero[nonzero == 0] = 1

    # Project the sum of the vertical axis
    prj = np.sum(rot, axis=0) / nonzero

    # Leave only the explicitly identifiable peaks
    prj[prj < np.mean(prj)] = 0

    peaks = argrelextrema(prj, np.greater, order=3)[0]
    n_peaks = len(peaks)

    # PLOT: Image blocks
    if verbose:
        plt.figure()
        plt.imshow(1 - rot, cmap='gray')
        plt.text(0, -3, 'Number of peaks =' + str(n_peaks))

        npts = len(prj)
        plt.plot(np.linspace(0, npts, npts), prj / max(prj) * npts, color='r')
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()
        plt.show()

    if n_peaks == 1:
        return 1 / np.max((peaks[0], rot.shape[0] - peaks[0]))

    # Frequency = number of peaks / distance between the 2 most distant peaks
    return n_peaks / (peaks[n_peaks - 1] - peaks[0]) if n_peaks != 0 else 0


def skeleton(image: np.array):
    """
    Skeletonize the image

    Arguments:
        image - image to skeletonize

    Return the skeletonized image
    """
    return skeletonize(image)


def prune(sklt: np.array,
          windows: np.array,
          iters: int=1):
    """
    Remove iteratively matching points of the skeleton image

    Arguments:
        sklt    - skeleton image to be pruned. Can be acquired via skeleton() \
function
        windows - array of n dimentional windows to match
        iters   - number of times to repeat the pruning procedure

    Return the pruned image
    """
    pruned = np.array(sklt)
    for _ in itertools.repeat(None, iters):
        for wnd in windows:
            pruned[binary_hit_or_miss(pruned, wnd)] = 0

    return pruned


def fillholes(image: np.array,
              hole_value: bool=0):
    """
    Fill one-pixel-sized holes in binary image

    Arguments:
        image - image to be processed

    Return image with the holes filled
    """
    image_filled = np.array(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] == hole_value and\
               not image[i + 1, j] == hole_value and\
               not image[i - 1, j] == hole_value and\
               not image[i, j + 1] == hole_value and\
               not image[i, j - 1] == hole_value:
                image_filled[i, j] = not hole_value
    return image_filled
