#
# Copyright (c) 2020-present, Andrei Yaskovets
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from enum import Enum
from itertools import combinations
from math import tan

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import fplib.preprocess as fppreprocess


verbose = False


class MnType(Enum):
    Core = 1
    Bifurcation = 2
    Termination = 3


import fplib.plot as fpplot


def minutae(sklt: np.array,
            orient: np.array=None,
            ridge_value: bool=1,
            remove_invalid: bool=1):
    """
    Extract minutae points from the skeleton image

    Arguments:
        sklt           - skeleton image
        orient         - orientation matrix. Can be acquired via the \
orientation() function
        ridge_value    - value of ridge pixels
        remove_invalid - whether to remove invalid minutae

    Return array of tuples with minutae coordinates (row, col, type, angle)
    """
    idx = [(0,0), (1,0), (2,0), (2,1), (2,2), (1,2), (0,2), (0,1), (0,0)]
    points = []
    for i in range(1, sklt.shape[0] - 1):
        for j in range(1, sklt.shape[1] - 1):
            if not sklt[i][j] == ridge_value:
                continue

            blk = sklt[i - 1:i + 2, j - 1:j + 2]

            # pfunc value is equal to the number of ridges that come into block
            psums = [blk[k1] * (1 - blk[k2]) for k1, k2 in zip(idx, idx[1:])]
            pfunc = np.sum(psums)

            angle = None
            if pfunc == 1:
                if orient is not None:
                    angle = fppreprocess._orientblk_angle(orient, (i, j), 3)
                points.append((i, j, MnType.Termination, angle))
            elif pfunc > 2:
                if orient is not None:
                    angle = fppreprocess._orientblk_angle(orient, (i, j), 3)
                points.append((i, j, MnType.Bifurcation, angle))

            if verbose:
                # PLOT: image block 3x3
                plt.figure()
                plt.imshow(blk, cmap='gray')
                plt.text(-0.5, -0.6, 'pfunc = ' + str(pfunc), color='black')
                plt.show()

    if remove_invalid:
        # Remove border terminations
        points = _remove_border_points(sklt, points, ridge_value)

    mnte = np.empty(len(points), dtype=tuple)
    mnte[:] = points
    return mnte


def _remove_border_points(sklt: np.array,
                          minutiae: tuple,
                          ridge_value: bool=1):
    """ Remove border minutae points """
    result = []
    for point in minutiae:
        i, j, _, _ = point
        # Count as a border if there is a direction with only ridge 0 pixels
        if (ridge_value in sklt[i, :j]) and\
           (ridge_value in sklt[:i, j]) and\
           (ridge_value in sklt[i, j + 1:]) and\
           (ridge_value in sklt[i + 1:, j]):
           result.append(point)
    return result


def core(ornt: np.array,
         mask: np.array=None,
         blksize: int=None):
    """
    Extract core point from an orientation image

    Arguments:
        ornt    - orientation matrix. Can be acquired via the \
orientation() function
        blksize - block size for calculations
        mask    - fingerprint mask. Can be acquired via the mask() function

    Returns core point tuple (row, col, type=MnType.Core)
    """
    if blksize is None:
        blksize = int(min(ornt.shape) / 25)

    # Acquire plain angles and square them
    angl = fppreprocess.angles(ornt, blksize)

    """# Poor results with angular characteristics
    rads = np.deg2rad(angl)
    rads = rads * rads

    # Create a map of combined poincare index and angular coherence
    pncr = fppreprocess.poincare(rads, blksize)
    angc = fppreprocess.angular_coherence(rads, blksize, blksize)
    pnac = fppreprocess.normalize((1 - fppreprocess.normalize(angc, 0, 1)) +
        fppreprocess.normalize(pncr, 0, 1), 0, 1)

    pnac[pnac.shape[0] - (pnac.shape[0]%blksize):,:] = 0
    pnac[:,pnac.shape[1] - (pnac.shape[1]%blksize):] = 0
    if mask is not None:
        pnac *= mask
    """

    # Create mask to search for core orientations clockwise
    crcl_5x5 = np.array([
        [-45, -30, 0, 30, 45],
        [-50, -45, 90, 45, 50],
        [-70, 90, 90, 90, 70],
        [50, 45, 90, -45, -50],
        [45, 30, 0, -30, -45]
    ])
    arch_5x5 = np.array([
        [-45, -30, 0, 30, 45],
        [-50, -45, 90, 45, 50],
        [-70, 90, 90, 90, 70],
        [90, 90, 90, 90, 90],
        [90, 90, 90, 90, 90]
    ])

    coremask_side = 5

    padw = int(np.floor(coremask_side / 2))
    sclangl = angl[::blksize, ::blksize]
    padangl = np.pad(sclangl, padw, constant_values=0)

    arch, crcl = np.zeros(sclangl.shape), np.zeros(sclangl.shape)
    for i in range(0, sclangl.shape[0]):
        for j in range(0, sclangl.shape[1]):
            r = i + padw
            c = j + padw

            d = np.abs(padangl[r-padw:r+padw+1, c-padw:c+padw+1] - arch_5x5)
            d[d > 90] -= 180
            arch[i, j] = np.sum(np.abs(d))

            d = np.abs(padangl[r-padw:r+padw+1, c-padw:c+padw+1] - crcl_5x5)
            d[d > 90] -= 180
            crcl[i, j] = np.sum(np.abs(d))

    arch = (1 - fppreprocess.normalize(arch, 0, 1))
    crcl = (1 - fppreprocess.normalize(crcl, 0, 1))
    if mask is not None:
        arch *= mask[::blksize, ::blksize]
        crcl *= mask[::blksize, ::blksize]

    mask_match = fppreprocess.normalize(arch * crcl, 0, 1)
    mask_maxes = np.where(mask_match >= np.max(mask_match) - 0.1)
    row = mask_maxes[0][0] * blksize + blksize / 2
    col = mask_maxes[1][0] * blksize + blksize / 2

    """ Poor results with angular characteristics
    # Look through N most probable points and find the most conforming one
    N = 7

    pnacmax = np.sort(np.unique(pnac[::blksize, ::blksize].ravel()))[-N:][::-1]
    crclmax = np.sort(np.unique(crcl.ravel()))[-N:][::-1]

    # Find the most probable point
    sclpnac = pnac[::blksize, ::blksize]

    row, col, coef = 0, 0, 0
    for pval in pnacmax:
        pmatches = np.transpose(np.where(sclpnac == pval))
        for pmatch in pmatches:
            for cval in crclmax:
                cmatches = np.transpose(np.where(crcl == cval))
                for cmatch in cmatches:
                    dist = np.linalg.norm(pmatch - cmatch)
                    newcoef = (pval + cval) / (dist if dist != 0 else 1)
                    if (newcoef > coef):
                        coef = newcoef
                        row = int((cmatch[0] + pmatch[0] + 1) * blksize / 2)
                        col = int((cmatch[1] + pmatch[1] + 1) * blksize / 2)
    """

    if verbose:
        # PLOT: angles and core probability maps
        fpplot.plotstack((pnac[::blksize, ::blksize], crcl), axis='x')
        fpplot._plotangles(crcl.repeat(blksize, axis=0).repeat(blksize, axis=1),
            angl, blksize)

    return (row, col, MnType.Core)
