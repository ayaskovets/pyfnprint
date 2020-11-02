import numpy as np

import fplib.binarize   as fpbinarize
import fplib.feature    as fpfeature
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
        ]), 8)
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

    # core point detection
    mnte = np.resize(mnte, (mnte.shape[0] + 1,))
    mnte[mnte.shape[0] - 1] = fpminutae.core(ornt, mask)

    # feature vector creation
    feat_c = fpfeature.extract(mnte, method='circular', bucketsize=15)
    feat_r = fpfeature.extract(mnte, method='radial', bucketsize=10)

    return fnp, nimg, mask, ornt, sklt, mnte, feat_c, feat_r


path = './test/FVC/2000/DB1_B/101_8.tif'
#path = './test/PNG/1_4.png'
fnp, nimg, mask, ornt, sklt, mnte, feat_c, feat_r = prepare(path)

# fpplot.plotimage(nimg * mask)
# fpplot.plotorient(nimg, ornt, blksize)
fpplot.plotminutae(sklt, mnte)
