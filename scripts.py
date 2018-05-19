def list_shapes_of_features_in_ismir():
    from feature_extraction.FeaturesFacade import Feature
    import numpy

    ismir_angry = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/angry/features_dumps/"
    ismir_happy = "/media/michal/HDD/Music Emotion Datasets/Decoded/ISMIR2012/happy/features_dumps/"

    features = [Feature.COMPRESS_FEATURE, Feature.MEDIAN_SPECTRAL_BAND_ENERGY, Feature.SPECTRAL_CENTROID,
                Feature.SPECTRAL_PATTERN, Feature.DELTA_SPECTRAL_PATTERN, Feature.VARIANCE_DELTA_SPECTRAL_PATTERN,
                Feature.CORRELATION_PATTERN, Feature.SPECTRAL_CONTRAST_PATTERN]

    for ismir_part in [ismir_angry, ismir_happy]:
        print(ismir_part)
        for feature in features:
            path = ismir_part + feature.value + ".npy"
            loaded_fs = numpy.load(path)
            print(feature.value)
            print(loaded_fs.shape)


if __name__ == '__main__':
    list_shapes_of_features_in_ismir()
