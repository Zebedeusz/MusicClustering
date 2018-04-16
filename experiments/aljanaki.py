def simplify_annotations_file():
    anns_filepath = "/media/michal/HDD/Music Emotion Datasets/Decoded/data_statistics/aljanaki_1.csv"
    elements_qnt = 400

    from experiments.utilities import load_annotations
    entire_anns = load_annotations(anns_filepath)
    print('x')


if __name__ == '__main__':
    simplify_annotations_file()
