import csv
import os

import numpy


def load_data_from_file(path):
    data = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            temp_arr = []
            with open(path + filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    temp_arr.append(row)
                data.append(temp_arr)
        data = numpy.asarray(data, 'float32')

        data = numpy.nan_to_num(data)

        print("spectrograms loaded")

    return data


def read_features_from_file(path, omit_first_column=False):
    with open(path, newline='') as f:
        features = []
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            features.append(row)
    if (omit_first_column):
        for row in features:
            del row[0]
    features = numpy.asarray(features, dtype="float32")
    return features


def read_valence_arousal(plot):
    path = "/home/michal/PycharmProjects/Datasets/1000_songs_dataset/annotations/static_annotations_a_v_only.csv"
    with open(path, newline='') as f:
        features = []
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            features.append(row)
    features = numpy.asarray(features, dtype="float32")

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(features[:, 0], features[:, 1], 'g.')
        plt.ylabel('arousal')
        plt.xlabel('valence')
        plt.savefig("/home/michal/PycharmProjects/AudioFeatureExtraction/charts/data_vis.png")
        plt.show()

    return features
