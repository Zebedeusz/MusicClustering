dumps_path = "features_dumps/"


# loads dumps with arrays with chosen features from directories specified in parameter 'data_classes_dirs'
# annotates them according to names of those dirs
# returns dictionary with keys - feature vectors and values - annotatios
# usable for datasets such as ismir
def load_and_annotate_feature_dumps(data_root_path, data_classes_dirs, features):
    import joblib
    import numpy

    annotated_features = {}
    for data_class_path in data_classes_dirs:
        class_features = []
        for feature in features:
            class_features = numpy.hstack((class_features,
                                           joblib.load(data_root_path + data_class_path + dumps_path + feature[
                                                    0].value + ".pkl")))
        data_class_dict = {k: data_class_path.replace("/", "") for k in class_features}
        annotated_features.update(data_class_dict)

    return annotated_features


# loads dumps with arrays with chosen features from root dataset path
def load_feature_dumps(data_root_path, features):
    import joblib
    import numpy

    class_features = []
    for feature in features:
        path = data_root_path + dumps_path + feature.value + ".pkl"
        class_features = numpy.hstack((class_features, joblib.load(path)))
    return class_features


# loads numpy binaries with arrays with chosen features from root dataset path
def load_feature_npys(data_root_path, features):
    import numpy
    import os

    npys_paths = []
    if (os.path.isdir(data_root_path + dumps_path)):
        npys_paths.append(data_root_path + dumps_path)
    else:
        all_dirs_in_path = [x[0] for x in os.walk(data_root_path)]
        for dir in all_dirs_in_path:
            if str(dir).__contains__("features_dumps"):
                npys_paths.append(dir)

    all_features = []
    for dir in npys_paths:
        class_features = []
        for feature in features:
            path = dir + "/" + feature.value + ".npy"
            if len(class_features) == 0:
                class_features = numpy.load(path)
            else:
                to_stack = numpy.load(path)
                class_features = numpy.hstack((class_features, to_stack))
        all_features.extend(class_features)
    all_features = numpy.array(all_features)
    return all_features


# loads annotations saved in .csv file
# the file has to contain only annotation - column with ids and headline have to be removed
def load_annotations(annotations_path):
    import numpy
    raw_data = open(annotations_path, 'rt')
    data = numpy.loadtxt(raw_data, delimiter=",")
    return data


# given 1d arrays with labeled elements in dataset - associacions to groups - 'labels'
# and number of groups
# and 2d array with emotions values for every element
# calculates mean emotions values for elements in every group
def analyse_clustering_results(groups_qnt, labels, annotations, save_path=False):
    import numpy

    groups_with_annotated_elements = []
    for i in range(groups_qnt):
        groups_with_annotated_elements.append([])
    for i in range(len(labels)):
        groups_with_annotated_elements[labels[i]].append(numpy.array(annotations[i]))
    groups_with_annotated_elements = numpy.array(groups_with_annotated_elements)

    groups_annotated = []
    for group in groups_with_annotated_elements:
        groups_annotated.append(numpy.mean(group, axis=0))
    groups_annotated = numpy.array(groups_annotated)

    emotions_per_group = []
    for group in groups_annotated:
        max_el = numpy.max(group)
        emotions_per_group.append(max_el - numpy.mean(numpy.delete(group, max_el)))

    emotions_purity = numpy.mean(emotions_per_group)

    if save_path:
        numpy.savetxt(save_path, groups_annotated, delimiter=",", fmt='%.2e')

    return emotions_purity


def get_free_results_filepath(dataset_root_path, results_filename):
    import os

    results_dirpath = dataset_root_path + "experiments/"
    results_filepath = results_dirpath + results_filename + ".csv"
    file_no = 1
    while os.path.exists(results_filepath):
        results_filepath = results_dirpath + results_filename + "_{}.csv".format(file_no)
        file_no += 1

    return results_filepath


def get_results_array_template(feature, cluster_sizes, eps_values, min_samples_qnts):
    import numpy

    results_array = numpy.zeros(shape=(3 + len(cluster_sizes) * 2 + (len(eps_values) * len(min_samples_qnts)), 9),
                                dtype="S30")

    results_array[0] = ['cechy', "metoda grupowania", "parametry", "", "rozmiary grup", "silhuette", "dunn",
                        "davies-bouldin", "purity emocji"]
    results_array[1] = ["", "", "liczebnosc grup", "", "", "", "", "", ""]
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
