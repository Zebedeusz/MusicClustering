root_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/"
dumps_path = "features_dumps/"
dumps_path_local = "../feature_dumps/eerola"
annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/eerola_simple.csv"
annotations_path_local = "../annotations/eerola_simple.csv"

if __name__ == '__main__':
    from experiments.utilities import load_annotations, analyse_clustering_results
    from Clustering import cluster_som, cluster_k_means, cluster_dbscan
    from feature_extraction.FeaturesFacade import Feature
    from experiments.utilities import load_feature_npys
    import numpy

    # TODO
    # create list to be saved as csv structured as on x.csv on desktop and save with numpy as array with dtype str
    # numpy.empty(shape=(), dtype="str")

    # parameters
    cluster_sizes = [4, 6, 8, 10]
    eps_values = [0.01, 0.1, 0.5, 1, 5, 10, 100]
    min_saples_qnts = [2, 5, 10, 20]
    clustering_methods = {"k_means": cluster_k_means, "dbscan": cluster_dbscan, "som": cluster_som}

    anns = load_annotations(annotations_path_local)

    # MFCC
    from feature_extraction.helper import get_gmms_samples_from_path

    gmm_pkl_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/gmm_pkl"
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

    for method_name, method in clustering_methods.items():
        stats_save_path_method = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/experiments/" + method_name

        if method_name == "k_means":
            for n_clusters in cluster_sizes:
                stats_save_path_n_clus = stats_save_path_method + "_" + str(n_clusters)

                for f_name, f in fs.items():
                    stats_save_path_final = stats_save_path_n_clus + "_" + f_name

                    labels = cluster_k_means(f, n_clusters,
                                             stats_save_path=stats_save_path_final + "_clustering_stats.csv")
                    analyse_clustering_results(n_clusters, labels, anns,
                                               stats_save_path_final + "_groups_emotions.csv")

        elif method_name == "dbscan":
            for eps in eps_values:
                stats_save_path_eps = stats_save_path_method + "_eps_" + str(eps)
                for min_samples in min_saples_qnts:
                    stats_save_path_min_s = stats_save_path_eps + "_min_samples_" + str(min_samples)

                    for f_name, f in fs.items():
                        stats_save_path_final = stats_save_path_min_s + "_" + f_name

                        labels = cluster_dbscan(f, eps, min_samples,
                                                stats_save_path=stats_save_path_final + "_clustering_stats.csv")
                        analyse_clustering_results(len(set(labels)), labels, anns,
                                                   stats_save_path_final + "_groups_emotions.csv")

        elif method_name == "som":
            for n_clusters in cluster_sizes:
                stats_save_path_n_clus = stats_save_path_method + "_" + str(n_clusters)

                for f_name, f in fs.items():
                    stats_save_path_f = stats_save_path_n_clus + "_" + f_name

                    labels = cluster_som(f, n_clusters, stats_save_path=stats_save_path_f + "_clustering_stats.csv")
                    analyse_clustering_results(n_clusters, labels, anns,
                                               stats_save_path_f + "_groups_emotions.csv")
