annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_simple.csv"
annotations_path_local = "../annotations/aljanaki_1.csv"

# parameters
cluster_sizes = [6, 8, 10, 12, 16]
eps_values = [0.01, 0.1, 0.5, 1, 2, 85, 100, 120, 150, 175, 200, 220, 250, 280, 300]
min_samples_qnts = [2, 5, 10, 20]


def simplify_annotations_file():
    anns_filepath = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_1.csv"
    elements_qnt = 400

    import numpy
    from experiments.utilities import load_annotations

    entire_anns = load_annotations(annotations_path_local)

    simplified_annotations = []
    for i in range(elements_qnt):
        anns_for_id = []
        for row in entire_anns:
            if not int(row[0]) == i + 1:
                continue
            anns_for_id.append(row[1:])
        if len(anns_for_id) > 0:
            simplified_annotations.append(numpy.mean(anns_for_id, axis=0))
    simplified_annotations = numpy.array(simplified_annotations)

    numpy.savetxt(annotations_path, simplified_annotations, delimiter=",")


if __name__ == '__main__':
    # simplify_annotations_file()
    from experiments.helper import conduct_experiments

    conduct_experiments("Aljanaki", cluster_sizes, eps_values, min_samples_qnts)
