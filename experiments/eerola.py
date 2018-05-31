root_path = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola/"
dumps_path = "features_dumps/"
dumps_path_local = "../feature_dumps/eerola"
annotations_path = "/media/michal/HDD1/Music Emotion Datasets/Decoded/data_statistics/eerola_simple.csv"
annotations_path_local = "../annotations/eerola_simple.csv"

# parameters
cluster_sizes = [4, 6, 8, 10]
eps_values = [0.01, 0.1, 0.5, 1, 2, 85, 100, 120, 150, 175, 200, 220, 250, 280, 300]
min_samples_qnts = [2, 5, 10, 20]
clustering_methods = ["k_means", "dbscan"]

if __name__ == '__main__':
    from feature_extraction.FeaturesFacade import Feature
    from experiments.utilities import load_feature_npys
    import numpy

    # MFCC
    from feature_extraction.helper import get_gmms_samples_from_path

    gmm_pkl_path = "/media/michal/HDD1/Music Emotion Datasets/Decoded/Eerola/gmm_pkl"
    mfccs = get_gmms_samples_from_path(gmm_pkl_path)
    mfccs = numpy.delete(mfccs, 6, axis=0)

    # FP
    # from feature_extraction.helper import load_fps_from_path
    # fps_catalogue = "eerola"
    # fps = load_fps_from_path(fps_catalogue)

    # 1st art
    features1 = [Feature.COMPRESS_FEATURE, Feature.MEDIAN_SPECTRAL_BAND_ENERGY, Feature.SPECTRAL_CENTROID]
    features1_dumps = load_feature_npys(root_path, [features1[2]])
    features1_dumps = numpy.delete(features1_dumps, 6, axis=0)
    features1_dumps = numpy.hstack((features1_dumps, load_feature_npys(root_path, features1[0:2])))

    # 2nd art
    features2 = [Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN,
                 Feature.CORRELATION_PATTERN, Feature.SPECTRAL_CONTRAST_PATTERN]
    features2_dumps = load_feature_npys(root_path, features2)

    fs = {"mfcc": mfccs, "group1": features1_dumps, "group2": features2_dumps}

    from experiments.utilities import conduct_experiments

    conduct_experiments(root_path, annotations_path_local, fs, clustering_methods, cluster_sizes, eps_values,
                        min_samples_qnts)
