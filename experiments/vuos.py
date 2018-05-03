root_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/Vuoskoski/"
dumps_path = "features_dumps/"
dumps_path_local = "../feature_dumps/vuos"
annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/vuos_simple.csv"
annotations_path_local = "../annotations/vuos_simple.csv"

if __name__ == '__main__':
    from experiments.utilities import load_annotations, analyse_clustering_results
    from Clustering import cluster_k_means
    from feature_extraction.FeaturesFacade import Feature
    from experiments.utilities import load_feature_npys

    features = [Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN]
    f_dumps = load_feature_npys(root_path, features)
    anns = load_annotations(annotations_path_local)
    # labels = cluster_som(f_dumps)
    labels = cluster_k_means(f_dumps, 8, False)
    analyse_clustering_results(8, labels, anns)
