import matplotlib.pyplot as plt
import numpy as np


from fplib.minutae import Type


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
        image   - 
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


def plotminutae(sklt: np.array,
                minutae: np.array):
    """
    Plot the skeleton image with minutae overlapped

    Arguments:
        sklt    - skeleton image. Can be acquired via skeleton() function
        minutae - minutae list consisting of tuples (x, y, minutae_type). \
Can be acquired via minutae() function
    """
    fig, ax = plt.subplots()
    plt.imshow(sklt, cmap='gray')

    for i, j, t in minutae:
        if t == Type.Termination:
            ax.add_artist(plt.Circle((j, i), radius=2, color='red',
                fill=False))
        elif t == Type.Bifurcation:
            ax.add_artist(plt.Circle((j, i), radius=2, color='yellow',
                fill=False))

    plt.show()
