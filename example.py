from os import path, listdir
import csv

import numpy as np

import fplib.binarize   as fpbinarize
import fplib.feature    as fpfeature
import fplib.filter     as fpfilter
import fplib.image      as fpimage
import fplib.minutae    as fpminutae
import fplib.plot       as fpplot
import fplib.preprocess as fppreprocess


verbose = False


def _prepare(fnp: fpimage.FingerprintImage):
    # read image
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
    feat_r = fpfeature.extract(mnte, method='radial', bucketsize=36)
    feat_c = fpfeature.extract(mnte, method='circular', bucketsize=30)

    # visualizations
    if verbose:
        fpplot.plotimage(nimg * mask)
        fpplot.plotminutae(sklt, mnte)

    return nimg, mask, ornt, sklt, mnte, feat_r, feat_c


def enroll(fnp: fpimage.FingerprintImage,
           folder: str):
    nimg, mask, ornt, sklt, mnte, feat_r, feat_c = _prepare(fnp)

    template_path = path.join(folder, str(fnp.id) + '_' + str(fnp.number))
    np.save(template_path + '_r', feat_r, allow_pickle=True)
    np.save(template_path + '_c', feat_c, allow_pickle=True)


def load_templates(folder: str):
    templates = []
    for template in listdir(folder):
        template_path = path.join(folder, template)
        id = template.split('_')[0]
        templates.append((id, np.load(template_path, allow_pickle=True)))
    return templates


def identify(fnp: fpimage.FingerprintImage,
             templates: list):
    nimg, mask, ornt, sklt, mnte, feat_r, feat_c = _prepare(fnp)

    distances = {}
    for template in templates:
        if template[0] not in distances.keys():
            distances[template[0]] = 0

        if template[1][1] == feat_r[1]:
            distances[template[0]] += fpfeature.distance(feat_r, template[1])
        if template[1][1] == feat_c[1]:
            distances[template[0]] += fpfeature.distance(feat_c, template[1])

    print(min(distances, key=distances.get), distances)
    return min(distances, key=distances.get)


if __name__ == "__main__":
    # verbose = True
    root = 'data'
    template_storage = path.join(root, 'templates')

    # create templates
    train_fnps = sorted(fpimage.readFolder(path.join(root, 'train', '*')))
    for i in range(0, len(train_fnps)):
        print('[', i + 1, '/', len(train_fnps), '] Enrolling ',
            train_fnps[i].fppath, '...', sep='')

        enroll(train_fnps[i], template_storage)

    # write the prediction file
    with open(path.join(root, 'prediction.csv'), 'w') as testfile:
        predictionwriter = csv.writer(testfile, delimiter=',')
        predictionwriter.writerow(['name', 'id'])

        # load templates
        templates = load_templates(template_storage)

        # make predictions
        test_fnps = sorted(fpimage.readFolder(path.join(root, 'test', '*')))
        for i in range(0, len(test_fnps)):
            print('[', i + 1, '/', len(test_fnps), '] Identifying ',
                test_fnps[i].fppath, '...', sep='', end='')

            name = path.basename(test_fnps[i].fppath)
            id = identify(test_fnps[i], templates)
            predictionwriter.writerow([name, str(id)])
