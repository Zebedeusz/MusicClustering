from enum import Enum
class Feature(Enum):
    MFCC = "MFCC"
    MFCC40 = "MFCC40"

    COMPRESS_FEATURE = "COMPRESSIBILITY FEATURE"
    MEDIAN_SPECTRAL_BAND_ENERGY = "MEDIAN SPECTRAL BAND ENERGY"
    SPECTRAL_CENTROID = "SPECTRAL CENTROID"
    
    SPECTRAL_PATTERN = "SPECTRAL PATTERN"
    DELTA_SPECTRAL_PATTERN = "DELTA SPECTRAL PATTERN"
    VARIANCE_DELTA_SPECTRAL_PATTERN = "VARIANCE DELTA SPECTRAL PATTERN"
    LOGARITHMIC_FLUCTUATION_PATTERN = "LOGARITHMIC FLUCTUATION PATTERN"
    CORRELATION_PATTERN = "CORRELATION PATTERN"
    SPECTRAL_CONTRAST_PATTERN = "SPECTRAL CONTRAST PATTERN"


def read_and_save_features_from_files_in_path(path, features, csvSavePath):
    import os

    extracted_features = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
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


def get_features_of_file(filepath, features):
    import scipy.io.wavfile as wav
    import numpy

    extracted_features = []

    try:
        f, sound = wav.read(filepath)
    except FileNotFoundError:
        return extracted_features

    if not isinstance(features, list):
        raise Exception("Unsupported data type as function parameter")

    for feature in features:
        if features is Feature.MFCC:
            from librosa.feature import mfcc
            mfcc = mfcc(sound)
            mfcc = numpy.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))
            extracted_features.extend(mfcc)
        #elif str(features).__contains__("mfcc") and str(features).replace("mfcc","").isdigit():
        elif feature is Feature.MFCC40:
            from librosa.feature import mfcc
            #mfcc = (mfcc(sound, n_mfcc=int(features.replace("mfcc",""))))
            mfcc = (mfcc(sound, n_mfcc=40))
            mfcc = numpy.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))
            extracted_features.extend(mfcc)
        elif  feature is Feature.COMPRESS_FEATURE:
            from Features import compressibility_feature
            extracted_features.extend(compressibility_feature(filepath))
        elif  feature is Feature.MEDIAN_SPECTRAL_BAND_ENERGY:
            from Features import median_spectral_band_energy
            extracted_features.extend(median_spectral_band_energy(sound, ))
        elif feature is Feature.SPECTRAL_CENTROID:
            from Features import spectral_centroid
            extracted_features.extend(spectral_centroid(sound, ))
        else:
            #preprocess
            pass

    return extracted_features

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
            #print("Extracting from file " + filename)
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
            #print("Extracting from file " + filename)
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

def get_features_from_csv(filepath, omit_first_column):
    pass
