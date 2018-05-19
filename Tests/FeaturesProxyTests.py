import unittest


class FeaturesProxyTests(unittest.TestCase):
    test_file = "../resources/test_file.wav"

    # def test_preprocessToKeplerUniFeatures(self):
    #     import scipy.io.wavfile as wav
    #     from feature_extraction.FeaturesProxy import preprocessToKeplerUniFeatures
    #
    #     f, sound = wav.read(self.test_file)
    #
    #     soundP = preprocessToKeplerUniFeatures(sound)

    def test_spectral_base(self):
        from feature_extraction.FeaturesFacade import get_feature, Feature
        import scipy.io.wavfile as wav

        f, sound = wav.read(self.test_file)

        get_feature(Feature.SPECTRAL_PATTERN, sound, "")

        # soundP = preprocessToKeplerUniFeatures(sound)
        # s = spectral_pattern_base(soundP, 3, 10, 5, False, 0.1)
        # spectral_pattern(soundP)
        # correlation_pattern(soundP)
        # spectral_contrast_pattern(sound)
        # print(s)


if __name__ == '__main__':
    unittest.main()
