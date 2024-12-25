#!/usr/bin/env python3

#
# Copyright (c) 2024, Andrei Yaskovets
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from os import path, listdir, makedirs
import csv, getopt, sys

import numpy as np

import fplib.binarize   as fpbinarize
import fplib.feature    as fpfeature
import fplib.filter     as fpfilter
import fplib.image      as fpimage
import fplib.minutae    as fpminutae
import fplib.plot       as fpplot
import fplib.preprocess as fppreprocess


def _prepare(fnp: fpimage.FingerprintImage, verbose: bool):
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
           folder: str,
           verbose: bool):
    nimg, mask, ornt, sklt, mnte, feat_r, feat_c = _prepare(fnp, verbose)

    template_path = path.join(folder, f'{fnp.id}_{fnp.number}')
    np.save(f'{template_path}_{feat_r[1]}', feat_r[0], allow_pickle=True)
    np.save(f'{template_path}_{feat_c[1]}', feat_c[0], allow_pickle=True)


def load_templates(folder: str):
    templates = []
    for template in listdir(folder):
        template_path = path.join(folder, template)
        id = template.split('_')[0]
        method = template.split('_')[-1].split('.')[0]
        templates.append((id, method, np.load(template_path, allow_pickle=True)))
    return templates


def identify(fnp: fpimage.FingerprintImage,
             templates: list,
             thresh_mean_multiplier: int,
             verbose: bool):
    nimg, mask, ornt, sklt, mnte, feat_r, feat_c = _prepare(fnp, verbose)

    distances = {}
    for template in templates:
        id, method, feat = template

        if id not in distances.keys():
            distances[id] = 0

        if method == feat_r[1]:
            distances[template[0]] += fpfeature.distance(feat_r, (feat, method))
        elif method == feat_c[1]:
            distances[template[0]] += fpfeature.distance(feat_c, (feat, method))

    minid = min(distances, key=distances.get)
    thresh = np.mean(list(distances.values())) * thresh_mean_multiplier

    return minid if (distances[minid] < thresh) else None


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'vept:')
        if len(args) != 2:
            raise getopt.GetoptError('')
    except getopt.GetoptError:
        print('usage: python3 ', sys.argv[0], ' [OPTIONS] [data_path] [templates_path]',
            '\n\tRun predictions on prepared data'
            '\n'
            '\n\t[OPTIONS]:',
            '\n\t-v               - enable verbose visualizations',
            '\n\t-e               - enroll mode',
            '\n\t-p               - prediction mode',
            '\n\t-t [multiplier]  - mean distance to all templates is multiplied by this value to get prediction threshold',
            '\n\t[data_path]      - path to the data directory that contains train and test data',
            '\n\t[templates_path] - path to templates directory to store processed fingerprints to',
            sep='')
        sys.exit(1)

    verbose = False
    enroll_mode = False
    prediction_mode = False
    thresh_mean_multiplier = 0.667
    data_path = args[0]
    templates_path = args[1]

    if not path.exists(data_path) or not path.isdir(data_path):
        print('[data_path] is not a directory')
        sys.exit(1)

    if not path.exists(templates_path):
        makedirs(templates_path)
    elif not path.isdir(templates_path):
        raise Exception('[templates_path] exists and is not a directory')

    for opt, arg in opts:
        if opt == '-v':
            verbose = True
        if opt == '-e':
            enroll_mode = True
        if opt == '-p':
            prediction_mode = True
        if opt == '-t':
            thresh_mean_multiplier = float(arg)

    if not enroll_mode and not prediction_mode:
        print('Either -e or -p must be specified')
        sys.exit(1)

    # Create templates
    if enroll_mode:
        train_fnps = sorted(fpimage.readFolder(path.join(data_path, 'train', '*')))
        for i, train_fnp in enumerate(train_fnps, start=1):
            print(f'[{i}/{len(train_fnps)}] Enrolling {train_fnp.file_path}...')

            enroll(train_fnp, templates_path, verbose)

    # Write the prediction file
    if prediction_mode:
        with open(path.join(data_path, 'prediction.csv'), 'w') as testfile:
            predictionwriter = csv.writer(testfile, delimiter=',')
            predictionwriter.writerow(['name', 'predicted_id'])

            # Load templates
            templates = load_templates(templates_path)

            # Make predictions
            test_fnps = sorted(fpimage.readFolder(path.join(data_path, 'test', '*')))
            for i, test_fnp in enumerate(test_fnps, start=1):
                print(f'[{i}/{len(test_fnps)}] Identifying {test_fnp.file_path}...')

                name = path.basename(test_fnp.file_path)
                id = identify(test_fnp, templates, thresh_mean_multiplier, verbose)
                predictionwriter.writerow([name, id])
