import csv
import numpy
import math
import os


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
        data = numpy.asarray(data)
        data = numpy.asarray(data, 'float32')

        #check for infinities or nans
        sp_num = 0
        for sp in data:
            sp_row_num = 0
            for sp_row in sp:
                sp_el_num = 0
                for sp_el in sp_row:
                    if math.isnan(sp_el):
                        data[sp_num, sp_row_num, sp_el_num] = 0.0
                    if math.isinf(sp_el):
                        if (sp_el < 0):
                            data[sp_num, sp_row_num, sp_el_num] = 0.0
                        else:
                            data[sp_num, sp_row_num, sp_el_num] = 1.0
                    sp_el_num += 1
                sp_row_num += 1
            sp_num += 1

        print("spectrograms loaded")
    return data



def read_features_from_file(path):
    with open(path, newline='') as f:
        features = []
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            features.append(row)
    features = numpy.asarray(features, dtype="float32")
    features *= 10000
    print(numpy.min(features))
    print(numpy.max(features))
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

        plt.ylabel('arousal')
        plt.xlabel('valence')
        plt.plot(features[:,0], features[:,1], 'g.')
        plt.savefig("/home/michal/PycharmProjects/AudioFeatureExtraction/charts/data_vis.png")
        plt.show()

    return features

def get_features_from_files_in_path(path, features = ()):
    from pydub import AudioSegment
    import os
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if str(filename).endswith(".wav"):
                sound = AudioSegment.from_wav(path + filename)
    #TODO
