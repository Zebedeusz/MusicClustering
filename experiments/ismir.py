root_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/"
angry_path = "angry/"
happy_path = "happy/"
relax_path = "relax/"
sad_path = "sad/"
data_class_paths = [angry_path, happy_path, relax_path, sad_path]

# parameters
cluster_sizes = [3, 4, 6, 8]
eps_values = [0.01, 0.1, 0.5, 1, 2, 85, 100, 120, 150, 175, 200, 220, 250, 280, 300]
min_samples_qnts = [2, 5, 10, 20]


def generate_annotations_file():
    import numpy

    angry_files = 637
    happy_files = 0
    relax_files = 750
    sad_files = 764

    annotations_list = []
    for i in range(angry_files):
        annotations_list.append([1, 0, 0, 0])
    for i in range(happy_files):
        annotations_list.append([0, 1, 0, 0])
    for i in range(relax_files):
        annotations_list.append([0, 0, 1, 0])
    for i in range(sad_files):
        annotations_list.append([0, 0, 0, 1])

    annotations_list = numpy.array(annotations_list)

    numpy.savetxt("/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/ISMIR2012.csv",
                  annotations_list, delimiter=",", fmt="%d")  # , encoding="utf_8")

if __name__ == '__main__':
    # features = []
    # load_and_annotate_feature_dumps(root_path, data_class_paths, features)
    # generate_annotations_file()

    from experiments.helper import conduct_experiments

    conduct_experiments("ISMIR2012", cluster_sizes, eps_values, min_samples_qnts)
