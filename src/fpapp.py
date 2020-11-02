import time as tm

import numpy as np
import cv2 as cv

import fplib.binarize   as fpbinarize
import fplib.filter     as fpfilter
import fplib.image      as fpimage
import fplib.minutae    as fpminutae
import fplib.plot       as fpplot
import fplib.preprocess as fppreprocess


begin_time = tm.clock_gettime(tm.CLOCK_MONOTONIC)


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


nimg1, mask1, sklt1, mnte1, ornt1 = prepare('./test/PNG/1_2.png')
# _, sklt2, mnte2, _ = prepare('./test/PNG/1_4.png')

angl1 = fppreprocess.angles(ornt1, 10)
# fpplot.plotangles(sklt1, angl1, 10)

end_time = tm.clock_gettime(tm.CLOCK_MONOTONIC)
print(end_time - begin_time, '= Time')

# fpplot.plotorient(nimg1, ornt1)
# fpplot.plotimage(1 - sklt1)
# fpplot.plotminutae(1 - sklt1, mnte1)
