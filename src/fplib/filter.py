import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import rotate


verbose = False


def filter(image: np.array,
           method: str=None,
           krnsize: int=11,
           krnsigma: int=2,
           krn: np.array=None):
    """
    Filter an image using the selected kernel or a user-provided kernel

    Arguments:
        image    - image to filter
        method   - algorithm, one of ['gaussx', 'gaussy', 'highpass', \
'laplace', 'log', 'lowpass', 'sobelx', 'sobely']
        krnsize  - kernel size, used in all methods but ['laplacian']
        krnsigma - kernel frequency parameter, used in all methods but \
['laplace', 'sobelx', 'sobely']
        krn      - if provided and method=None, use the provided kernel

    Return the filtered image
    """
    if method is None and krn is not None:
        return _filter_custom(image, krn)

    if method == 'gaussx':
        return _filter_gauss(image, 'x', krnsize, krnsigma)
    elif method == 'gaussy':
        return _filter_gauss(image, 'y', krnsize, krnsigma)
    elif method == 'highpass':
        return _filter_highpass(image, krnsize, krnsigma)
    elif method == 'laplace':
        return _filter_laplacian(image)
    elif method == 'log':
        return _filter_log(image, krnsize, krnsigma)
    elif method == 'lowpass':
        return _filter_lowpass(image, krnsize, krnsigma)
    elif method == 'sobelx':
        return _filter_sobel(image, 'x', krnsize)
    elif method == 'sobely':
        return _filter_sobel(image, 'y', krnsize)
    else:
        raise Exception(method + ' filtering method is not supported!')


def _filter_custom(image: np.array,
                   krn: np.array):
    """ User-provided kernel filtering """
    assert(krn.shape[0] == krn.shape[1])
    return convolve2d(image.astype(np.float32), krn, mode='same')


def _filter_log(image: np.array,
                krnsize: np.array,
                krnsigma: float):
    """ LoG filtering """
    side = np.floor(krnsize / 2)
    x, y = np.meshgrid(np.arange(-side, side), np.arange(-side, side))

    krn = -(np.pi * krnsigma**4) *\
        (1 - ((x**2 + y**2) / (2 * krnsigma**2))) *\
        np.exp(-(x**2 + y**2) / (2 * krnsigma**2))

    return _filter_custom(image, krn)


def _filter_laplacian(image: np.array):
    """ Laplacian filtering """
    return cv.Laplacian(image.astype(np.float32), cv.CV_32F)


def _filter_sobel(image: np.array,
                  axis: str,
                  krnsize: int):
    """ The Sobel kernel filtering """
    assert(axis == 'x' or axis == 'y')
    if axis == 'x':
        return cv.Sobel(image.astype(np.float32), cv.CV_32F, 1, 0,
                        ksize=krnsize)
    elif axis == 'y':
        return cv.Sobel(image.astype(np.float32), cv.CV_32F, 0, 1,
                        ksize=krnsize)


def _filter_gauss(image: np.array,
                  axis: str,
                  krnsize: int,
                  krnsigma: float):
    """ One of gradients of the gaussian """
    assert(axis == 'x' or axis == 'y')
    krn = cv.getGaussianKernel(krnsize, krnsigma)
    krn = krn * krn.T
    krny, krnx = np.gradient(krn)

    if axis == 'x':
        return _filter_custom(image, krnx)
    elif axis == 'y':
        return _filter_custom(image, krny)


def _filter_lowpass(image: np.array,
                    krnsize: int,
                    krnsigma: float):
    """ Gaussian filtering """
    krn = cv.getGaussianKernel(krnsize, krnsigma)
    krn = krn * krn.T
    return _filter_custom(image, krn)


def _filter_highpass(image: np.array,
                     krnsize: int,
                     krnsigma: float):
    """ Inverse gaussian filtering """
    krn = cv.getGaussianKernel(krnsize, krnsigma)
    krn = krn * krn.T
    return image.astype(np.float32) - _filter_custom(image, krn)


def medgabor(image: np.array,
             orient: np.array,
             freq: np.array):
    """
    Gabor-filtering based on the median frequency estimation

    Arguments:
        image  - an image to filter
        orient - orientation matrix of the image
        freq   - frequency estimation matrix of the image

    Return the filtered image
    """
    medfreq = np.round(np.median(freq[freq > 0]), 2)
    sigma_x = 0.66 / medfreq
    sigma_y = 0.66 / medfreq

    # Original gabor filter
    krnsize = int(np.round(3 * np.max([sigma_x, sigma_y])))
    x, y = np.meshgrid(np.linspace(-krnsize, krnsize, (2 * krnsize + 1)),
                       np.linspace(-krnsize, krnsize, (2 * krnsize + 1)))

    original_gab = np.exp(-(x**2 / sigma_x**2 + y**2 / sigma_y**2)) *\
        np.cos(2 * np.pi * medfreq * x)

    d_angle = 3
    n_filters = int(180 / d_angle)

    # Generate rotated filters
    fbank = np.zeros((n_filters, original_gab.shape[0], original_gab.shape[1]))
    for o in range(0, n_filters):
        fbank[o] = rotate(original_gab, -(o * d_angle + 90), reshape=False)

    # Find indices of matrix points greater than maxsze from the image boundary
    validr, validc = np.array(np.where(freq != 0))
    padimage = np.pad(image, krnsize, constant_values=0)

    # Convert orientation matrix values from radians to a filter index
    gab_idx = np.round(orient * n_filters / np.pi)
    gab_idx[gab_idx < 1] += n_filters
    gab_idx[gab_idx > n_filters] -= n_filters

    flt = np.zeros(image.shape, dtype=np.float32)

    # Do the filtering
    for k in range(0, len(validr)):
        r = validr[k] + krnsize
        c = validc[k] + krnsize

        gab = fbank[int(gab_idx[r - krnsize][c - krnsize] - 1)]
        blk = padimage[r - krnsize:r + krnsize + 1,
            c - krnsize:c + krnsize + 1]
        flt[r - krnsize][c - krnsize] = np.sum(blk * gab)

        # PLOT: Gabor filter along with the corresponding image block
        if verbose:
            plt.figure()
            plt.imshow(np.hstack((blk, gab)), cmap='gray')
            plt.show()
    return flt


"""
def gabor(image, orient, freq):
    flt = np.zeros(image.shape)
    freq = np.round(freq, 2)

    krnsize = 20
    offset = int(np.floor(krnsize / 2))
    step = int(np.floor(offset))

    thresh = np.median(freq)
    validr, validc = np.array(np.where(freq != 0))
    valididx = np.array(np.where(
        (validr > krnsize) & (validr < image.shape[0] - krnsize) &
        (validc > krnsize) & (validc < image.shape[1] - krnsize)))

    for k in valididx[0]:
        i = validr[k]
        j = validc[k]

        sin = np.mean(np.sin(2 * orient[i:i + offset, j:j + offset]))
        cos = np.mean(np.cos(2 * orient[i:i + offset, j:j + offset]))

        sigma = np.round(np.mean(freq[i:i + offset, j:j + offset]), 2)
        angle = np.round(np.arctan2(sin, cos) / 2, 1)

        if sigma < thresh:
            flt[i - offset:i + offset + 1, j - offset:j + offset + 1] = 255
            continue

        x, y = np.meshgrid(np.linspace(-offset, offset, (2 * offset + 1)),
                            np.linspace(-offset, offset, (2 * offset + 1)))

        x_theta = x * np.cos(angle + np.pi / 2) + y * np.sin(angle + np.pi / 2)
        y_theta = y * np.cos(angle + np.pi / 2) - x * np.sin(angle + np.pi / 2)

        sigma_x = 0.66 / (abs(np.cos(angle)) + 0.01) / sigma
        sigma_y = 0.66 / (abs(np.sin(angle)) + 0.01) / sigma
        lmbd = 1 / sigma

        gab = np.exp(-(x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) *\
            np.cos(2 * np.pi / lmbd * x_theta)

        imgblk = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
        flt[i][j] = np.sum(imgblk * gab)

        # PLOT: Gabor filter along with the corresponding image block
        # fpplot.plotstack((imgblk, gab), 'x')

    return flt
"""
