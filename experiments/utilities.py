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
        data_class_features = []
        for feature in features:
            data_class_features = numpy.hstack((data_class_features,
                                                joblib.load(data_root_path + data_class_path + dumps_path + feature[
                                                    0].value + ".pkl")))
        data_class_dict = {k: data_class_path.replace("/", "") for k in data_class_features}
        annotated_features.update(data_class_dict)

    return annotated_features


# loads dumps with arrays with chosen features from root dataset path
def load_feature_dumps(data_root_path, features):
    import joblib
    import numpy

    data_class_features = []
    for feature in features:
        path = data_root_path + dumps_path + feature.value + ".pkl"
        data_class_features = numpy.hstack((data_class_features, joblib.load(path)))
    return data_class_features


# loads numpy binaries with arrays with chosen features from root dataset path
def load_feature_npys(data_root_path, features):
    import numpy

    data_class_features = []
    for feature in features:
        path = data_root_path + dumps_path + feature.value + ".npy"
        if len(data_class_features) == 0:
            data_class_features = numpy.load(path)
        else:
            data_class_features = numpy.hstack((data_class_features, numpy.load(path)))
    return data_class_features


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

    if save_path:
        numpy.savetxt(save_path, groups_annotated, delimiter=",")

    return groups_annotated
