import time as tm

import numpy as np
import cv2 as cv

import fplib.binarize   as fpbinarize
import fplib.filter     as fpfilter
import fplib.image      as fpimage
import fplib.minutae    as fpminutae
import fplib.plot       as fpplot
import fplib.preprocess as fppreprocess


# begin_time = tm.clock_gettime(tm.CLOCK_MONOTONIC)
# end_time = tm.clock_gettime(tm.CLOCK_MONOTONIC)
# print(end_time - begin_time, '= Time')

def prepare(path):
    fnp = fpimage.readOne(path)
    img = fnp.getData(colorspace=cv.IMREAD_GRAYSCALE, astype=np.uint8)
    img = fppreprocess.resize(img, width=400, height=500)

    # initial preprocessing
    img = fppreprocess.normalize(img, low=0, upp=255)
    mask = fppreprocess.mask(img, blksize=20)
    nimg = fppreprocess.standartize(img)

    # orientation matrix
    ornt = fppreprocess.orientation(nimg, grdsigma=3, blksigma=3, smtsigma=3)

    # frequency matrix
    freq = fppreprocess.frequency(nimg, ornt, blksize=50)
    freq = freq * mask

    # gabor filtering
    prep = fpfilter.medgabor(nimg, ornt, freq)

    # binarization
    prep = 255 - fppreprocess.normalize(prep, 0, 255, np.uint8)
    prep = fpbinarize.binarize(prep, 'otsu')
    prep = fppreprocess.fillholes(prep)

    # skeletization
    sklt = fppreprocess.skeleton(prep)
    sklt = fppreprocess.prune(sklt,
        np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]], np.bool))

    # minutae extraction
    mnte = fpminutae.minutae(sklt, ornt, remove_invalid=1)

    return nimg, mask, sklt, mnte, ornt


nimg1, mask1, sklt1, mnte1, ornt1 = prepare('./test/PNG/7_4.png')
# nimg2, mask2, sklt2, mnte2, ornt2 = prepare('./test/PNG/1_2.png')

blksize = 15
angl1 = fppreprocess.angles(ornt1, blksize)

anglmod = np.deg2rad(angl1)
anglmod = anglmod * anglmod

# fpplot.plotimage(anglmod)

pncr1 = fppreprocess.angular_coherence(anglmod, blksize, 11)
fpplot.plotangles(pncr1, angl1, blksize)
