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
        sound = preprocessToKeplerUniFeatures(sound)

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


# returns array of shape (f,t) e.g. (1025,1295)
def preprocessToKeplerUniFeatures(sound):
    from scipy.signal import stft
    import numpy

    f = 22050

    #samplerate to 22kHz
    #sound = sound.set_frame_rate(f)

    #STFT - window size 2048 samples, hop 512 samples, Hanning
    f, t, sound_stft = stft(sound, fs=1.0, window="hann", nperseg=2048, noverlap=512)

    #magnitude spectrum
    sound_stft_sole = abs(sound_stft)

    #linear resolution to logarithmic Cent scale
    cent_const = 440*(pow(2,-57/12))
    sound_cent_scale = 1200*numpy.log2(sound_stft_sole/cent_const)
    sound_cent_scale = numpy.nan_to_num(sound_cent_scale)

    #to logarithmic scale again
    sound_log_cent_scale = 20*numpy.log10(sound_cent_scale)
    sound_log_cent_scale = numpy.nan_to_num(sound_log_cent_scale)

    #normalization
    #switched off as better image without it
    # row_sums = sound_log_cent_scale.sum(axis=0)
    # sound_log_cent_scale = sound_log_cent_scale / row_sums[numpy.newaxis, :]

    # import matplotlib.pyplot as plt
    #
    # plt.pcolormesh(t, f, sound_log_cent_scale)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    return sound_log_cent_scale