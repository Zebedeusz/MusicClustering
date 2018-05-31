filepath = "/media/michal/HDD1/Music Emotion Datasets/Decoded/1000songs/2.wav"
filepath2 = "/media/michal/HDD1/Music Emotion Datasets/Decoded/1000songs"

aljanaki = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Aljanaki/"

##
songs1k = "/media/michal/HDD1/Music Emotion Datasets/Decoded/1000songs"

eerola = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola"

aljanaki_classical = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Aljanaki/classical"
aljanaki_electronic = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Aljanaki/electronic"
aljanaki_pop = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Aljanaki/pop"
aljanaki_rock = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Aljanaki/rock"

genres_blues = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/blues"
genres_classical = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/classical"
genres_country = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/country"
genres_disco = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/disco"
genres_hiphop = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/hiphop"
genres_jazz = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/jazz"
genres_metal = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/metal"
genres_pop = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/pop"
genres_reggae = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/reggae"
genres_rock = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres/rock"

ismir_angry = "/media/michal/HDD1/Music Emotion Datasets/Decoded/ISMIR2012/angry"
ismir_happy = "/media/michal/HDD1/Music Emotion Datasets/Decoded/ISMIR2012/happy"
ismir_relax = "/media/michal/HDD1/Music Emotion Datasets/Decoded/ISMIR2012/relax"
ismir_sad = "/media/michal/HDD1/Music Emotion Datasets/Decoded/ISMIR2012/sad"

vuos = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Vuoskoski/"
##
genres = "/media/michal/HDD1/Music Emotion Datasets/Decoded/genres"

eerola_pkl = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola/gmm_pkl"
eerola_data_stats_fp = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola/data_statistics/variance_distribution_from_fp.png"
eerola_data_stats_gmm = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola/data_statistics/variance_distribution_of_variances_from_gmms.png"
eerola_fp = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Fluctuation Patterns/Eerola.csv"

test_file = "resources/test_file.wav"

from feature_extraction.FeaturesFacade import Feature
# fs = get_features_of_file(test_file, [Feature.COMPRESS_FEATURE])
# print(fs)
from feature_extraction.helper import save_npy_with_feature_for_dataset

features = [Feature.COMPRESS_FEATURE, Feature.MEDIAN_SPECTRAL_BAND_ENERGY, Feature.SPECTRAL_CENTROID,
            Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN,
            Feature.CORRELATION_PATTERN, Feature.SPECTRAL_CONTRAST_PATTERN]

features = [Feature.COMPRESS_FEATURE,
            Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN,
            Feature.CORRELATION_PATTERN, Feature.SPECTRAL_CONTRAST_PATTERN]

# , Feature.MEDIAN_SPECTRAL_BAND_ENERGY left for better times
datasets = [ismir_happy, ismir_angry]
datasets_with_no_msbe = [aljanaki_rock, aljanaki_pop,
                         aljanaki_electronic, aljanaki_classical]
genres = [genres_blues, genres_classical, genres_country, genres_disco, genres_hiphop, genres_jazz, genres_metal,
          genres_pop, genres_reggae, genres_rock]

datasets_with_no_msbe.extend(genres)


# datasets.extend(datasets_with_no_msbe)
# datasets.extend(genres)

def isFeatureSavedForDataset(path, feature):
    import os.path
    feature_dumps_path = "/features_dumps/"
    return os.path.exists(path + feature_dumps_path + feature.value + ".npy")

for dataset in datasets:
    print(dataset.split("/")[-1])
    for f in features:
        # if not isFeatureSavedForDataset(dataset, f):
        print(f)
        save_npy_with_feature_for_dataset(dataset, [f])

        # fs = get_features_of_files_in_path(eerola, [Feature.CORRELATION_PATTERN])
# read_and_save_features_from_files_in_path(filepath2, "mfcc", filepath2)

# gmms = get_gmms_from_mfccs_of_filepath(filepath2)

# cluster_som(fs)
# cluster_hierarchical(fs)
# from Statistics import variance_distribution
# from IO import read_features_from_file
# fps = read_features_from_file(eerola_fp, True)
# variance_distribution(fps, True, eerola_data_stats_fp,"",1)
# from feature_extraction.helper import get_features_of_file, Features
# ee = get_features_of_file(filepath, [Feature.COMPRESS_FEATURE])
# print(ee)
