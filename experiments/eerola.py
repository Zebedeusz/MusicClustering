root_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/Eerola/"
dumps_path = "features_dumps/"
dumps_path_local = "../feature_dumps/eerola"
annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/eerola_simple.csv"
annotations_path_local = "../annotations/eerola_simple.csv"

# parameters
cluster_sizes = [4, 6, 8, 10]
eps_values = [0.01, 0.1, 0.5, 1, 2, 85, 100, 120, 150, 175, 200, 220, 250, 280, 300]
min_samples_qnts = [2, 5, 10, 20]
clustering_methods = ["k_means", "dbscan"]


def get_free_results_filepath(results_filename):
    import os

    results_dirpath = root_path + "experiments/"
    results_filepath = results_dirpath + results_filename + ".csv"
    file_no = 1
    while os.path.exists(results_filepath):
        results_filepath = results_dirpath + results_filename + "_{}.csv".format(file_no)
        file_no += 1

    return results_filepath


def get_results_array_template(feature):
    import numpy

    results_array = numpy.zeros(shape=(3 + len(cluster_sizes) * 2 + (len(eps_values) * len(min_samples_qnts)), 7),
                                dtype="S30")

    results_array[0] = ['cechy', "metoda grupowania", "parametry", "", "rozmiary grup", "silhuette", "sum war"]
    results_array[1] = ["", "", "liczebnosc grup", "", "", "", ""]
    results_array[2, 0:2] = [feature, "k-srednich"]
    results_array[2 + len(cluster_sizes), 1] = "SOM"
    results_array[2 + 2 * len(cluster_sizes), 2:4] = ["eps", "min_samples"]
    results_array[3 + 2 * len(cluster_sizes), 1] = "DBSCAN"

    last_som_rom_no = 2 + 2 * len(cluster_sizes)
    results_array[2:last_som_rom_no, 2] = numpy.asarray(cluster_sizes + cluster_sizes, dtype="S2")

    first_dbscan_row_no = 3 + 2 * len(cluster_sizes)
    last_dbscan_row_no = first_dbscan_row_no + (len(eps_values) * len(min_samples_qnts))
    results_array[first_dbscan_row_no:last_dbscan_row_no, 2] = sorted(eps_values * len(min_samples_qnts))
    results_array[first_dbscan_row_no:last_dbscan_row_no, 3] = min_samples_qnts * len(eps_values)

    return results_array

if __name__ == '__main__':
    from experiments.utilities import load_annotations, analyse_clustering_results
    from Clustering import cluster_som, cluster_k_means, cluster_dbscan
    from feature_extraction.FeaturesFacade import Feature
    from experiments.utilities import load_feature_npys
    import numpy

    get_results_array_template("x")

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

    group_sizes_col = 4
    sil_col = 5
    sum_war_col = 6
    for f_name, f in fs.items():
        try:
            results_array = get_results_array_template(f_name)

            for method_name in clustering_methods:
                if method_name == "k_means":
                    row = 2
                    for n_clusters in cluster_sizes:
                        labels, group_sizes, sil = cluster_k_means(f, n_clusters)
                        summed_var = analyse_clustering_results(n_clusters, labels, anns,
                                                                save_path=root_path + "/experiments/" + f_name + "/" + method_name + "_" + str(
                                                                    n_clusters) + ".csv")

                        group_sizes = group_sizes.replace(" ", "")
                        group_sizes = group_sizes.split(",")
                        group_sizes = ";".join([str(s) for s in group_sizes])
                        results_array[row, group_sizes_col] = group_sizes
                        results_array[row, sil_col] = sil
                        results_array[row, sum_war_col] = summed_var
                        row += 1

                elif method_name == "dbscan":
                    row = 3 + 2 * len(cluster_sizes)
                    for eps in eps_values:
                        for min_samples in min_samples_qnts:
                            labels, group_sizes, sil = cluster_dbscan(f, eps, min_samples)
                            summed_var = analyse_clustering_results(len(set(labels)), labels, anns,
                                                                    save_path=root_path + "/experiments/" + f_name + "/" + method_name + "_" + str(
                                                                        eps) + "_" + str(min_samples) + ".csv")

                            group_sizes = group_sizes.replace(" ", "")
                            group_sizes = group_sizes.split(",")
                            group_sizes = ";".join([str(s) for s in group_sizes])
                            results_array[row, group_sizes_col] = group_sizes
                            results_array[row, sil_col] = sil
                            results_array[row, sum_war_col] = summed_var
                            row += 1

                elif method_name == "som":
                    row = 2 + len(cluster_sizes)
                    for n_clusters in cluster_sizes:
                        labels, group_sizes, sil = cluster_som(f, n_clusters)
                        summed_var = analyse_clustering_results(n_clusters, labels, anns,
                                                                save_path=root_path + "/experiments/" + f_name + "/" + method_name + "_" + str(
                                                                    n_clusters) + ".csv")

                        group_sizes = group_sizes.replace(" ", "")
                        group_sizes = group_sizes.split(",")
                        group_sizes = ";".join([str(s) for s in group_sizes])
                        results_array[row, group_sizes_col] = group_sizes
                        results_array[row, sil_col] = sil
                        results_array[row, sum_war_col] = summed_var
                        row += 1
        finally:
            numpy.savetxt(get_free_results_filepath(f_name), results_array, delimiter=",", fmt="%s")
