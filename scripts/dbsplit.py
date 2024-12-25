#!/usr/bin/env python3

#
# Copyright (c) 2024, Andrei Yaskovets
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
from datetime import datetime
from os import listdir, makedirs, path
from shutil import copyfile
from typing import List, Dict, Tuple
import csv, getopt, random, sys

import fplib.image as fpimage


def split(input_paths: List[str],
          share_of_fingers_in_train: int,
          share_of_samples_per_finger_in_train: int) -> Tuple[
              Dict[str, List[fpimage.FingerprintImage]],
              Dict[str, List[fpimage.FingerprintImage]]]:
    """
    Combine the contents of the passed directories into a single db and split it
    into train & test

    Arguments:
        input_paths                          - list of paths to fingerprint containing directories
        share_of_fingers_in_train            - percent of users to add to train
        share_of_samples_per_finger_in_train - percent of images per user to add to train

    Returns train, test
    """
    # Read all files into [{id: [fingerprint_image]}]
    input_paths_images = []
    for input_path in input_paths:
        if not path.isdir(input_path):
            print(f'Input {input_path} is not a directory')
            sys.exit(1)

        path_images = defaultdict(list)
        for input_file in listdir(input_path):
            fingerprint_image = fpimage.readOne(path.join(input_path, input_file))
            path_images[fingerprint_image.id].append(fingerprint_image)
        input_paths_images.append(path_images)

        total_fingers = len(path_images)
        total_images = sum(map(len, path_images.values()))

        print(f'- read {total_images} images of {total_fingers} fingers from {input_path}')

    # Shuffle the images and restructure into [[fingerprint_image]] where each
    # sublist shares the same id
    db = []
    global_id = 1
    for input_path_images in input_paths_images:
        ids = list(input_path_images.keys())
        random.shuffle(ids)

        # Take images from a single directory in a random order
        for id in ids:
            # Shuffle images within the same id
            random.shuffle(input_path_images[id])

            for i in range(0, len(input_path_images[id])):
                input_path_images[id][i].id = global_id
                input_path_images[id][i].number = i + 1
            global_id += 1

            db.append(input_path_images[id])

    # Split into train/test
    train = defaultdict(list)
    test = defaultdict(list)

    total_ids_in_train = (share_of_fingers_in_train * len(db)) // 100
    total_ids_in_test = len(db) - total_ids_in_train
    total_images_in_train = 0
    total_images_in_test = 0

    for id in range(total_ids_in_train):
        total_images_of_id_in_train = (share_of_samples_per_finger_in_train * len(db[id]) // 100)
        total_images_of_id_in_test = len(db[id]) - total_images_of_id_in_train

        train[id + 1] = db[id][:total_images_of_id_in_train]
        test[id + 1] = db[id][total_images_of_id_in_train:]

        total_images_in_train += total_images_of_id_in_train
        total_images_in_test += total_images_of_id_in_test

    for id in range(total_ids_in_train, len(db)):
        test[id + 1] = db[id]

        total_images_in_test += len(db[id])

    print(f'- adding {total_images_in_train} images of {total_ids_in_train} fingers to train')
    print(f'- adding {total_images_in_test} images of {total_ids_in_test} fingers to test')

    return train, test


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:f:i:')
        if len(args) != 2:
            raise getopt.GetoptError('')
    except getopt.GetoptError:
        print('usage: python3 ', sys.argv[0], ' [OPTIONS] [input_paths] [output]',
            '\n\tSplit fingerprint data into train and test and create'
            '\n\ttest csv descriptor for validation of predictions'
            '\n'
            '\n\t[OPTIONS]:',
            '\n\t-s [number] - seed for RNG',
            '\n\t-f [number] - share (percents) of fingers in train (100 by default)',
            '\n\t-i [number] - share (percents) of images in train per user (50 by default)',
            '\n\t[input_paths]: one or more directories where every filename conforms to "[id]_[number].[ext]"',
            '\n\t[input_path]: directory to write the outputs to',
            sep='')
        sys.exit(1)

    input_paths = args[:-1]
    input_path = args[-1]
    share_of_fingers_in_train = 100
    share_of_samples_per_finger_in_train = 50
    seed = int(datetime.now().strftime('%Y%m%d%H%M%S'))

    for opt, arg in opts:
        if opt == '-f':
            value = int(arg)
            if value > 100 or value < 0:
                raise Exception('-f option must be within [0, 100] range')
            share_of_fingers_in_train = value
        elif opt == '-i':
            value = int(arg)
            if value > 100 or value < 0:
                raise Exception('-i option must be within [0, 100] range')
            share_of_samples_per_finger_in_train = value
        elif opt == '-s':
            seed = int(arg)

    if not path.exists(input_path):
        makedirs(input_path)
    elif not path.isdir(input_path):
        raise Exception('[input_path] exists and is not a directory')

    # Split data
    random.seed(seed)
    train, test = split(
        input_paths,
        share_of_fingers_in_train,
        share_of_samples_per_finger_in_train)

    # Create train & test directories
    train_path = path.join(input_path, 'train')
    if not path.exists(train_path):
        makedirs(train_path)

    test_path = path.join(input_path, 'test')
    if not path.exists(test_path):
        makedirs(test_path)

    # Write train as is
    train_finger_ids = train.keys()
    for id, images in train.items():
        for image in images:
            name = f'{id}_{image.number}.{image.file_path.split(".")[-1]}'
            copyfile(image.file_path, path.join(train_path, name))

    # Write test shuffled
    test_images = []
    for id, images in test.items():
        for image in images:
            test_images.append(image)
    random.shuffle(test_images)

    with open(path.join(input_path, 'test.csv'), 'w') as testfile:
        testwriter = csv.writer(testfile, delimiter=',')
        testwriter.writerows([
            ['seed', seed],
            ['paths', input_paths],
            ['share_of_fingers_in_train', share_of_fingers_in_train],
            ['share_of_samples_per_finger_in_train', share_of_samples_per_finger_in_train],
            ['enrolled', len(train_finger_ids)],
            ['---', '---'],
            ['name', 'true_id'],
        ])

        for i, test_image in enumerate(test_images, start=1):
            # Make all test fingers with mock id '0' for compatibility with fplib.FingerprintImage
            name = f'0_{i}.{test_image.file_path.split(".")[-1]}'
            copyfile(test_image.file_path, path.join(test_path, name))

            # Store none in test.csv for fingers that are not in train at all
            id = test_image.id if test_image.id in train_finger_ids else None
            testwriter.writerow([name, id])
