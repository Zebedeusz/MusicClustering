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
eerola_data_stats_fp = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/data_statistics/variance_distribution_from_fp.png"
eerola_data_stats_gmm = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/data_statistics/variance_distribution_of_variances_from_gmms.png"
eerola_fp = "/media/michal/HDD/Music Emotion Datasets/Decoded/Fluctuation Patterns/Eerola.csv"
#read_and_save_features_from_files_in_path(filepath2, "mfcc", filepath2)

#gmms = get_gmms_from_mfccs_of_filepath(filepath2)

# cluster_som(samples)
# cluster_hierarchical(samples)
# from Statistics import variance_distribution
# from IO import read_features_from_file
# fps = read_features_from_file(eerola_fp, True)
# variance_distribution(fps, True, eerola_data_stats_fp,"",1)
from feature_extraction.helper import get_features_of_file, Feature
ee = get_features_of_file(filepath, [Feature.COMPRESS_FEATURE])
print(ee)