annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_simple.csv"
annotations_path_local = "../annotations/aljanaki_1.csv"


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
    simplify_annotations_file()
