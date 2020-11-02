import math

import matplotlib.pyplot as plt
import numpy as np


from fplib.minutae import MnType
from fplib.preprocess import _orientblk_angle


def plotimage(image: np.array):
    """
    Plot an image using the grayscale colormap

    Arguments:
        image - image to display
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.show()


def plotstack(images: tuple,
              axis: str):
    """
    Plot multiple images as one

    Arguments:
        images - tuple of images to stack and display
        axis   - axis to stack images around which. One of ['x', 'y']
    """
    assert(axis == 'x' or axis == 'y')
    if axis == 'x':
        plotimage(np.hstack(images))
    elif axis == 'y':
        plotimage(np.vstack(images))


def plotorient(image: np.array,
               orient: np.array,
               blksize: int=15):
    """
    Plot the orientation image

    Arguments:
        image   - image to plot
        orient  - orientation matrix of the image. Can be acquired via \
orientation() function
        blksize - quiver plot block size
    """
    rows, cols = image.shape

    sin = np.sin(orient[blksize:rows:blksize, blksize:cols:blksize])
    cos = np.cos(orient[blksize:rows:blksize, blksize:cols:blksize])

    x, y = np.meshgrid(
        np.arange(blksize, cols, step=blksize),
        np.arange(blksize, rows, step=blksize))

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, cos, -sin, color='red', pivot='middle')

    plt.xticks(np.arange(blksize / 2, cols, blksize))
    plt.yticks(np.arange(blksize / 2, rows, blksize))
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.grid(True)
    plt.show()
    plt.grid(False)


def _plotangles(image: np.array,
               angles: np.array,
               blksize: int):
    """
    Plot the skeleton image with minutae overlapped

    Arguments:
        image   - image to be used as a background
        angles  - ridge rotation matrix. Can be acquired via angles() function
        blksize - block size of angles
    """
    ls = int(np.floor(blksize / 2))
    col = 'red'

    tans = np.tan(np.deg2rad(angles))

    plt.figure()
    plt.imshow(image, cmap='gray')
    for j in range(ls, angles.shape[1], blksize):
        for i in range(ls, angles.shape[0], blksize):
            k = tans[i, j]
            if np.abs(k) > 1:
                x = lambda y: y / k
                plt.plot([j + x(-ls), j + x(ls)], [i - ls, i + ls], color=col)
            else:
                y = lambda x: x * k
                plt.plot([j - ls, j + ls], [i + y(-ls), i + y(ls)], color=col)
    plt.show()


def plotminutae(sklt: np.array,
                minutae: np.array):
    """
    Plot the skeleton image with minutae overlapped

    Arguments:
        sklt    - skeleton image. Can be acquired via skeleton() function
        minutae - minutae list consisting of tuples (row, col, type, angle). \
Can be acquired via minutae() function
    """
    fig, ax = plt.subplots()
    plt.imshow(sklt, cmap='gray')

    for point in minutae:
        i, j, t, a = point if len(point) == 4 else point + (None,)
        clr = None
        if t == MnType.Termination:
            clr = 'red'
            ax.add_artist(plt.Circle((j, i), radius=2, color=clr, fill=False))
        elif t == MnType.Bifurcation:
            clr = 'blue'
            ax.add_artist(plt.Circle((j, i), radius=2, color=clr, fill=False))
        elif t == MnType.Core:
            clr = 'lime'
            ax.add_artist(plt.Circle((j, i), radius=5, color=clr, fill=True))

        if a is not None:
            k = math.tan(np.deg2rad(a))
            if np.abs(k) > 1:
                x = lambda y: y / k
                plt.plot([j + x(-5), j + x(5)], [i - 5, i + 5], color=clr)
            else:
                y = lambda x: x * k
                plt.plot([j - 5, j + 5], [i + y(-5), i + y(5)], color=clr)

    plt.show()
