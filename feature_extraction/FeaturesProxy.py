from enum import Enum
class Feature(Enum):
    MFCC = "MFCC"
    MFCC40 = "MFCC40"

    COMPRESS_FEATURE = "COMPRESSIBILITY FEATURE"
    MEDIAN_SPECTRAL_BAND_ENERGY = "MEDIAN SPECTRAL BAND ENERGY"
    SPECTRAL_CENTROID = "SPECTRAL CENTROID"

    SPECTRAL_PATTERN = "SPECTRAL PATTERN"
    DELTA_SPECTRAL_PATTERN = "DELTA SPECTRAL PATTERN"
    VARIANCE_DELTA_SPECTRAL_PATTERN = "VARIANCE DELTA SPECTRAL PATTERN"
    LOGARITHMIC_FLUCTUATION_PATTERN = "LOGARITHMIC FLUCTUATION PATTERN"
    CORRELATION_PATTERN = "CORRELATION PATTERN"
    SPECTRAL_CONTRAST_PATTERN = "SPECTRAL CONTRAST PATTERN"

x = []
features_dict = {}
features_dict[Feature.MFCC] = import feature_extraction.Features

def get_feature(feature, sound, filepath):

    if not isinstance(feature, Feature):
        raise Exception("Unsupported data type as function parameter")


    if feature is Feature.MFCC:
        from feature_extraction.Features import mfcc
        return mfcc(sound)

    elif feature is Feature.MFCC40:
        from feature_extraction.Features import mfcc40
        return mfcc40(sound)

    elif feature is Feature.COMPRESS_FEATURE:
        from feature_extraction.Features import compressibility_feature
        return compressibility_feature(filepath)

    elif feature is Feature.MEDIAN_SPECTRAL_BAND_ENERGY:
        from feature_extraction.Features import median_spectral_band_energy
        return median_spectral_band_energy(sound)

    elif feature is Feature.SPECTRAL_CENTROID:
        from feature_extraction.Features import spectral_centroid
        return spectral_centroid(sound)

    else:
        # preprocess
        if feature is Feature.SPECTRAL_PATTERN:
            pass
        elif feature is Feature.DELTA_SPECTRAL_PATTERN:
            pass
        elif feature is Feature.VARIANCE_DELTA_SPECTRAL_PATTERN:
            pass
        elif feature is Feature.LOGARITHMIC_FLUCTUATION_PATTERN:
            pass
        elif feature is Feature.CORRELATION_PATTERN:
            pass
        elif feature is Feature.SPECTRAL_CONTRAST_PATTERN:
            pass
