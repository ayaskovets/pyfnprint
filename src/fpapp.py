import numpy as np

import fplib.binarize   as fpbinarize
import fplib.filter     as fpfilter
import fplib.image      as fpimage
import fplib.minutae    as fpminutae
import fplib.plot       as fpplot
import fplib.preprocess as fppreprocess


def prepare(path):
    # read image
    fnp = fpimage.readOne(path)
    img = fnp.getData()
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

    # morphologic transformations
    sklt = fppreprocess.prune(sklt,
        np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
        ]), 5)
    sklt = fppreprocess.prune(sklt,
        np.array([
            [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
        ]), 1)
    sklt = fppreprocess.prune(sklt,
        np.array([
            [[1, 1, 1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 1, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 1]],
            [[1, 0, 0], [1, 1, 0], [1, 0, 0]],
        ]), 1)

    # minutae extraction
    mnte = fpminutae.minutae(sklt, ornt, remove_invalid=1)

    return nimg, mask, sklt, mnte, ornt


nimg, mask, sklt, mnte, ornt = prepare('./test/FVC/2000/DB1_B/101_1.tif')

blksize = 11
angl = fppreprocess.angles(ornt, blksize)
rads = np.deg2rad(angl)
rads = rads * rads
# angc = fppreprocess.poincare(rads, blksize)
angc = fppreprocess.angular_coherence(rads, blksize, blksize / 3)

# fpplot.plotimage(nimg)
# fpplot.plotorient(nimg, ornt, blksize)
# fpplot.plotminutae(sklt, mnte)
fpplot.plotangles(angc, angl, blksize)