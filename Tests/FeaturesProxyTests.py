import unittest

class FeaturesProxyTests(unittest.TestCase):

    filepath = "/media/michal/HDD/Music Emotion Datasets/Decoded/1000songs/2.wav"

    def test_preprocessToKeplerUniFeatures(self):
        import scipy.io.wavfile as wav
        from feature_extraction.FeaturesProxy import preprocessToKeplerUniFeatures

        f, sound = wav.read(self.filepath)

        preprocessToKeplerUniFeatures(sound)


if __name__ == '__main__':
    unittest.main()