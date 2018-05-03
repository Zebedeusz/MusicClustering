def save_pkl_with_feature_for_dataset(path, feature):
    from sklearn.externals import joblib
    import os

    extracted_features = get_features_of_files_in_path(path, feature)
    if not os.path.isdir(path + "/features_dumps"):
        os.mkdir(path + "/features_dumps")
    joblib.dump(extracted_features, path + "/features_dumps/" + feature[0].value + ".pkl")


def save_npy_with_feature_for_dataset(path, feature):
    import os
    import numpy

    extracted_features = get_features_of_files_in_path(path, feature)
    if not os.path.isdir(path + "/features_dumps"):
        os.mkdir(path + "/features_dumps")
    numpy.save(path + "/features_dumps/" + feature[0].value, extracted_features)
    numpy.savetxt(path + "/features_dumps/" + feature[0].value + ".csv", extracted_features, delimiter=",")

def read_and_save_features_from_files_in_path(path, features, csvSavePath):
    import os

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if str(filename).endswith(".wav"):
                extracted_features = get_features_of_file(path + "/" + filename, features)

                csv_filename = str(filename).replace(".wav", ".csv")

                if csvSavePath is not "":
                    with open(csvSavePath + "/" + csv_filename, 'w') as csvfile:
                        for row in extracted_features:
                            for el in row:
                                csvfile.write(str(el))
                                csvfile.write(",")
                            csvfile.write("\n")
                        csvfile.close()
                print(features + " extracted from " + filename + " and saved as " + csv_filename)


def get_features_of_files_in_path(path, features):
    import os
    import numpy
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA

    extracted_features = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        files_qnt = len(filenames)
        qnt = 1
        if files_qnt > 0:
            percentage = qnt // files_qnt
            print("{} %".format(percentage))
        for filename in filenames:
            if str(filename).endswith(".wav"):
                extracted_features.append(get_features_of_file(path + "/" + filename, features))
                qnt += 1
                temp_percentage = qnt * 100 // files_qnt
                if temp_percentage is not percentage:
                    percentage = temp_percentage
                    print("{} %".format(percentage))

    extracted_features = normalize(numpy.array(extracted_features), axis=0)
    if (extracted_features.shape[1] > 40):
        pca = PCA(n_components=40)
        extracted_features = pca.fit_transform(extracted_features)
        extracted_features = normalize(extracted_features, axis=0)

    return extracted_features


def get_features_of_file(filepath, features):
    def add_feature_to_list(feature):
        import collections

        if isinstance(feature, collections.Iterable):
            extracted_features.extend(feature)
        else:
            extracted_features.append(feature)

    import scipy.io.wavfile as wav
    import numpy
    from feature_extraction.FeaturesFacade import get_feature
    import datetime

    print("{} : working with file: {}".format(datetime.datetime.now().time(), filepath))
    extracted_features = []

    try:
        print("Reading wave content")
        f, sound = wav.read(filepath)
    except FileNotFoundError:
        return extracted_features

    if not isinstance(features, list):
        raise Exception("Unsupported data type as function parameter")

    for feature in features:
        print("Reading " + feature.value)
        add_feature_to_list(get_feature(feature, sound, filepath))

    return numpy.asarray(extracted_features)


def get_gmms_from_mfccs_of_filepath(filepath):
    import numpy
    import os
    from sklearn.mixture import GaussianMixture

    gmms = numpy.array([])

    for (dirpath, dirnames, filenames) in os.walk(filepath):
        print("Extracting MFCC GMMs from files in " + filepath)
        files_qnt = len(filenames)
        qnt = 1
        percentage = qnt // files_qnt
        print("{} %".format(percentage))
        for filename in filenames:
            # print("Extracting from file " + filename)
            mfccs = get_features_of_file(filepath + "/" + filename, "mfcc")
            gmms = numpy.append(gmms, GaussianMixture(n_components=100).fit(mfccs))
            qnt += 1
            temp_percentage = qnt * 100 // files_qnt
            if temp_percentage is not percentage:
                percentage = temp_percentage
                print("{} %".format(percentage))

    return gmms


def save_gmms_from_mfccs_of_filepath(filepath):
    import os
    from sklearn.mixture import GaussianMixture
    from sklearn.externals import joblib

    for (dirpath, dirnames, filenames) in os.walk(filepath):
        print("Extracting MFCC GMMs from files in " + filepath)
        files_qnt = len(filenames)
        qnt = 1
        percentage = qnt * 100 // files_qnt
        print("{} %".format(percentage))
        for filename in filenames:
            # print("Extracting from file " + filename)
            mfccs = get_features_of_file(filepath + "/" + filename, "mfcc")
            if len(mfccs) > 0 and mfccs is not None:
                gmm = GaussianMixture(n_components=100).fit(mfccs)
                joblib.dump(gmm, filepath + "/gmm_pkl/" + filename.replace(".wav", ".pkl"))

                qnt += 1
                temp_percentage = qnt * 100 // files_qnt
                if temp_percentage is not percentage:
                    percentage = temp_percentage
                    print("{} %".format(percentage))


def get_gmms_samples_from_path(pkl_filepath):
    import os
    from sklearn.externals import joblib
    import numpy

    samples = []

    for (dirpath, dirnames, filenames) in os.walk(pkl_filepath):
        print("Reading .pkl GMM models from " + pkl_filepath)
        for filename in filenames:
            if str(filename).__contains__(".pkl"):
                samples.append(joblib.load(pkl_filepath + "/" + filename).sample()[0])

    samples = numpy.array(samples)

    samples = samples.reshape([samples.shape[0], samples.shape[2]])

    return samples


def load_fps_from_path(catalogue):
    import os
    from IO import read_features_from_file
    import numpy

    root_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/Fluctuation Patterns/"

    fps = []
    for (dirpath, dirnames, filenames) in os.walk(root_path + catalogue):
        for filename in filenames:
            if str(filename).__contains__(".csv"):
                fps.extend(read_features_from_file(root_path + catalogue + "/" + filename, True))

    return numpy.array(fps)
