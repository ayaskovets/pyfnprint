#!/usr/bin/env python3

#
# Copyright (c) 2020-present, Andrei Yaskovets
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from os import path
import csv, getopt, sys

from numpy import unique

from third_party import confusion_matrix_pretty_print


def evaluate(test_csv_path: str,
             prediction_csv_path: str):

    # Read the real test ids
    enrolled = None
    test = {}
    with open(test_csv_path, 'r') as testcsv:
        testreader = csv.reader(testcsv, delimiter=',')

        testdata = False
        for row in testreader:
            if testdata:
                test[int(row[0].split('_')[0])] = int(row[1])
            if row[0] == 'enrolled':
                enrolled = int(row[1])
            if row[0] == 'name' and row[1] == 'id':
                testdata = True

    if enrolled is None:
        raise Exception('Error: no \'enrolled,[value]\' row in' + sys.argv[1])

    # Read the prediction file
    prediction = {}
    with open(prediction_csv_path, 'r') as predictioncsv:
        predictionreader = csv.reader(predictioncsv, delimiter=',')

        for i, row in enumerate(predictionreader):
            if i == 0:
                if (row[0] != 'name' or row[1] != 'id'):
                    raise Exception('Error: ' + sys.argv[2] + 'must have\
                        [name, id] header row')
            else:
                prediction[int(row[0].split('_')[0])] = int(row[1])

    columns = list(range(1, enrolled + 1))
    if enrolled < len(unique(list(test.values()))):
        columns = list(range(1, enrolled + 1)) + [0]

    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(
        list(test.values()), list(prediction.values()),
        columns=columns, fz=8, show_null_values=True)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], '')
        if len(args) < 2:
            raise getopt.GetoptError('')
    except getopt.GetoptError:
        print('usage: python3 ', sys.argv[0], ' [test.csv] [prediction.csv]',
            '\n\t[test.csv]:       path to test.csv file made with dbsplit.py',
            '\n\t[prediction.csv]: prediction csv file with [name, id] cols',
            sep='')
        sys.exit(1)

    if not (path.exists(args[0]) and path.isfile(args[0])):
        raise Exception(args[0] + ' is invalid path')
    if not (path.exists(args[1]) and path.isfile(args[1])):
        raise Exception(args[1] + ' is invalid path')

    evaluate(args[0], args[1])
