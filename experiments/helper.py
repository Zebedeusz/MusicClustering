root_datasets_path = "/media/michal/HDD/Music Emotion Datasets/Decoded"
dumps_catalogue = "features_dumps"
gmm_pkl_catalogue = "gmm_pkl"
experiments_catalogue = "experiments"
annotations_catalogue = "data_statistics"

clustering_methods = ["k_means", "dbscan"]

# annotations table constants
group_sizes_col = 4
sil_col = 5
dunn_col = 6
davies_col = 7
emotions_purity_col = 8

def conduct_experiments(dataset_name, cluster_sizes, eps_values, min_samples_qnts):
    import numpy
    from Clustering import cluster_som, cluster_k_means, cluster_dbscan
    from experiments.utilities import load_annotations, get_results_array_template, get_free_results_filepath, \
        analyse_clustering_results

    from feature_extraction.FeaturesFacade import Feature
    from experiments.utilities import load_feature_npys

    # MFCC
    from feature_extraction.helper import get_gmms_samples_from_path
    mfccs = get_gmms_samples_from_path(root_datasets_path + "/" + dataset_name + "/" + gmm_pkl_catalogue)

    # FP
    from feature_extraction.helper import load_fps_from_path
    fps = load_fps_from_path(dataset_name)

    # 1st art
    features1 = [Feature.COMPRESS_FEATURE, Feature.MEDIAN_SPECTRAL_BAND_ENERGY, Feature.SPECTRAL_CENTROID]
    features1_dumps = load_feature_npys(root_datasets_path + "/" + dataset_name + "/", features1)

    # 2nd art
    features2 = [Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN,
                 Feature.CORRELATION_PATTERN, Feature.SPECTRAL_CONTRAST_PATTERN]
    features2_dumps = load_feature_npys(root_datasets_path + "/" + dataset_name + "/", features2)

    fs = {"mfcc": mfccs, "fp": fps, "group1": features1_dumps, "group2": features2_dumps}

    anns = load_annotations(root_datasets_path + "/" + annotations_catalogue + "/" + dataset_name + ".csv")

    for f_name, f in fs.items():
        try:
            results_array = get_results_array_template(f_name, cluster_sizes, eps_values, min_samples_qnts)

            for method_name in clustering_methods:
                if method_name == "k_means":
                    row = 2
                    for n_clusters in cluster_sizes:
                        labels, group_sizes, sil, dunn, davisb = cluster_k_means(f, n_clusters)
                        emotions_purity = analyse_clustering_results(n_clusters, labels, anns,
                                                                     save_path=root_datasets_path + "/" + dataset_name
                                                                          + "/experiments/" + f_name + "/" + method_name
                                                                          + "_" + str(n_clusters) + ".csv")

                        group_sizes = group_sizes.replace(" ", "")
                        group_sizes = group_sizes.split(",")
                        group_sizes = ";".join([str(s) for s in group_sizes])
                        results_array[row, group_sizes_col] = group_sizes
                        results_array[row, sil_col] = sil
                        results_array[row, dunn_col] = dunn
                        results_array[row, davies_col] = davisb
                        results_array[row, emotions_purity_col] = emotions_purity
                        row += 1

                elif method_name == "dbscan":
                    row = 3 + 2 * len(cluster_sizes)
                    for eps in eps_values:
                        for min_samples in min_samples_qnts:
                            labels, group_sizes, sil, dunn = cluster_dbscan(f, eps, min_samples)
                            emotions_purity = analyse_clustering_results(len(set(labels)), labels, anns,
                                                                         save_path=root_datasets_path + "/" + dataset_name
                                                                              + "/experiments/" + f_name + "/" + method_name
                                                                              + "_" + str(eps) + "_" + str(
                                                                        min_samples) + ".csv")

                            group_sizes = group_sizes.replace(" ", "")
                            group_sizes = group_sizes.split(",")
                            group_sizes = ";".join([str(s) for s in group_sizes])
                            results_array[row, group_sizes_col] = group_sizes
                            results_array[row, sil_col] = sil
                            results_array[row, dunn_col] = dunn
                            results_array[row, emotions_purity_col] = emotions_purity
                            row += 1

                elif method_name == "som":
                    row = 2 + len(cluster_sizes)
                    for n_clusters in cluster_sizes:
                        labels, group_sizes, sil = cluster_som(f, n_clusters)
                        emotions_purity = analyse_clustering_results(n_clusters, labels, anns,
                                                                     save_path=root_datasets_path + "/" + dataset_name
                                                                          + "/experiments/" + f_name + "/"
                                                                          + method_name + "_" + str(
                                                                    n_clusters) + ".csv")

                        group_sizes = group_sizes.replace(" ", "")
                        group_sizes = group_sizes.split(",")
                        group_sizes = ";".join([str(s) for s in group_sizes])
                        results_array[row, group_sizes_col] = group_sizes
                        results_array[row, sil_col] = sil
                        results_array[row, emotions_purity_col] = emotions_purity
                        row += 1
        finally:
            numpy.savetxt(get_free_results_filepath(root_datasets_path + "/" + dataset_name + "/", f_name),
                          results_array, delimiter=",", fmt="%s")  # , encoding="utf_8")
