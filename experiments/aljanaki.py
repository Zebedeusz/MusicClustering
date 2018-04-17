annotations_path = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_simple.csv"


def simplify_annotations_file():
    anns_filepath = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_1.csv"
    elements_qnt = 400

    import numpy
    from experiments.utilities import load_annotations

    entire_anns = load_annotations(anns_filepath)

    simplified_annotations = []
    for i in range(elements_qnt):
        anns_for_id = []
        for row in entire_anns:
            if row[0] is not i:
                break
            anns_for_id.append(row[1:])
        simplified_annotations.append(numpy.mean(anns_for_id, axis=0))
    simplified_annotations = numpy.array(simplified_annotations)

    numpy.savetxt(annotations_path, simplified_annotations, delimiter=",")


if __name__ == '__main__':
    simplify_annotations_file()
