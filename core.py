import numpy
from sklearn.mixture import GaussianMixture

from FeatureExtraction import get_features_of_file
from FeatureExtraction import read_and_save_features_from_files_in_path
from FeatureExtraction import get_gmms_from_mfccs_of_filepath, save_gmms_from_mfccs_of_filepath, get_gmms_samples_from_path

from Clustering import cluster_som, cluster_hierarchical

def variance_distribution_of_variances_from_gmms(data_path, gmm_samples, plot_save_path):
    variances = []
    for i in range(gmm_samples):
        print(str(i + 1)+ " / " + str(gmm_samples))
        samples = get_gmms_samples_from_path(data_path)

        from Statistics import variance_for_every_column
        variances.append(variance_for_every_column(samples))
    from Statistics import variance_distribution
    variance_distribution(variances, True, plot_save_path)

filepath = "/media/michal/HDD/Music Emotion Datasets/Decoded/1000songs/2.wav"
filepath2 = "/media/michal/HDD/Music Emotion Datasets/Decoded/1000songs"

aljanaki = "/media/michal/HDD/Music Emotion Datasets/Decoded/Aljanaki/"
vuos = "/media/michal/HDD/Music Emotion Datasets/Decoded/Vuoskoski"
eerola = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola"
ismir_angry = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/angry"
ismir_happy = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/happy"
ismir_relax = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/relax"
ismir_sad = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/sad"
genres = "/media/michal/HDD/Music Emotion Datasets/Decoded/genres"

eerola_pkl = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/gmm_pkl"
eerola_data_stats = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/data_statistics/variance_distribution_of_variances_from_gmms.png"
#read_and_save_features_from_files_in_path(filepath2, "mfcc", filepath2)

#gmms = get_gmms_from_mfccs_of_filepath(filepath2)

# cluster_som(samples)
# cluster_hierarchical(samples)

variance_distribution_of_variances_from_gmms(eerola_pkl, 100, eerola_data_stats)




