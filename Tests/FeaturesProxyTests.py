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
        import scipy.io.wavfile as wav
        from feature_extraction.FeaturesProxy import preprocessToKeplerUniFeatures
        from feature_extraction.Features import spectral_pattern

        f, sound = wav.read(self.test_file)

        soundP = preprocessToKeplerUniFeatures(sound)
        # s = spectral_pattern_base(soundP, 3, 0,0,0,0)
        spectral_pattern(soundP)



if __name__ == '__main__':
    unittest.main()