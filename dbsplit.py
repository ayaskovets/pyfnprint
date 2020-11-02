from collections import defaultdict
from datetime import datetime
from enum import Enum
from os import listdir, makedirs, path
from shutil import copyfile
from typing import List
import csv, getopt, random, sys

import fplib.image as fpimage


def split(paths: List[str],
          p_users: int,
          p_images: int):
    """
    Combine the contents of the passed folders into a single db and split it
    into train & test

    Arguments:
        paths    - list of paths to fingerprint containing folders
        p_users  - percent of users to add to train
        p_images - percent of images per user to add to train

    Returns train, test
    """
    # Read all files into dict(folder) of dicts(id) of lists(files)
    folders = {}
    for p in paths:
        d = path.dirname(p)
        if d not in folders.keys():
            folders[d] = defaultdict(list)

        if path.isdir(p):
            for f in listdir(p):
                fnp = fpimage.readOne(path.join(p, f))
                folders[d][fnp.id].append(fnp)
        else:
            fnp = fpimage.readOne(p)
            folders[d][fnp.id].append(fnp)
    # Shuffle the files
    db = []
    i = 1
    for folder, fnprints in folders.items():
        ids = list(fnprints.keys())
        random.shuffle(ids)
        for id in ids:
            fnps = list(fnprints[id])
            random.shuffle(fnps)
            for n in range(0, len(fnps)):
                fnps[n].number = n + 1
                fnps[n].id = i
            db.append(fnps)
            i += 1
    # Split to train and test
    train = defaultdict(list)
    test = defaultdict(list)

    for id in range(0, (p_users * len(db)) // 100):
        train[id + 1] = db[id][0:(p_images * len(db[id]) // 100)]
        test[id + 1] = db[id][(p_images * len(db[id]) // 100):]

    for id in range((p_users * len(db)) // 100, len(db)):
        test[id + 1] = db[id]

    return train, test


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:u:n:')
        if len(args) < 2:
            raise getopt.GetoptError('')
    except getopt.GetoptError:
        print('usage: python3 ', sys.argv[0], ' [OPTIONS] [paths] [out]',
            '\n\t[OPTIONS]:',
            '\n\t-s [number] - seed for random generator',
            '\n\t-u [number] - %% of train users (default 100)',
            '\n\t-n [number] - %% of train images per user (default 50)',
            '\n\t[paths]: multiple of folder (../) or wildcard (../*.png)',
            '\n\t[out]: folder to write the outputs to',
            sep='')
        sys.exit(1)

    if not (path.exists(args[-1]) and path.isdir(args[-1])):
        raise Exception('[out] is not provided')

    p_users = 100
    p_images = 50
    seed = int(datetime.now().strftime('%Y%m%d%H%M%S'))

    for opt, arg in opts:
        if opt == '-u':
            p_users = int(arg)
            if p_users > 100 or p_users < 0:
                raise Exception('-u option must be within [0, 100] range')
        elif opt == '-n':
            p_images = int(arg)
            if p_images > 100 or p_images < 0:
                raise Exception('-n option must be within [0, 100] range')
        elif opt == '-s':
            seed = int(arg)

    random.seed(seed)

    train, test = split(args[:-1], p_users, p_images)

    # Create train & test folders
    p_train = path.join(args[-1], 'train')
    if not path.exists(p_train):
        makedirs(p_train)
    p_test = path.join(args[-1], 'test')
    if not path.exists(p_test):
        makedirs(p_test)

    # Write train as it is
    for id, fnps in train.items():
        for fnp in fnps:
            name = str(id) + '_' + str(fnp.number) + '.' +\
                fnp.fppath.split('.')[-1]
            copyfile(fnp.fppath, path.join(p_train, name))

    # Write test shuffled
    testfnps = []
    for id, fnps in test.items():
        for fnp in fnps:
            testfnps.append(fnp)
    random.shuffle(testfnps)

    with open(path.join(args[-1], 'test.csv'), 'w') as testfile:
        testwriter = csv.writer(testfile, delimiter=',')
        testwriter.writerow(['seed', str(seed)])
        for a in args[:-1]:
            testwriter.writerow(['path', a])

        testwriter.writerow(['name', 'id'])
        for i in range(0, len(testfnps)):
            name = str(i + 1) + '_1.' + testfnps[i].fppath.split('.')[-1]
            copyfile(testfnps[i].fppath, path.join(p_test, name))
            testwriter.writerow([name, str(testfnps[i].id)])
