
def median_spectral_band_energy(wavedata):
    import numpy
    from numpy.fft import fftpack

    #magnitude_spectrum = stft(wavedata)
    fft = fftpack.fft(wavedata)
    power_spectrum = numpy.abs(fft)**2
    return numpy.median(power_spectrum)

#compresses wav file to flac and returns size of the flac file as feature value in bytes
def compressibility_feature(filepath_wav):
    import audiotools
    import os

    filepath_flac = filepath_wav.replace(".wav", ".flac")
    audiotools.open(filepath_wav).convert(filepath_flac,
                                          audiotools.FlacAudio)
    filesize = os.stat(filepath_flac).st_size
    os.remove(filepath_flac)
    return filesize

def spectral_centroid(x):
    import numpy as np

    samplerate = 22050

    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0 / samplerate)[:length // 2 + 1])  # positive frequencies
    return np.sum(magnitudes * freqs) / np.sum(magnitudes)  # return weighted mean

def mfcc(sound):
    from librosa.feature import mfcc
    import numpy
    mfcc = mfcc(sound)
    mfcc = numpy.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))

    return mfcc

def mfcc40(sound):
    from librosa.feature import mfcc
    import numpy
    mfcc = (mfcc(sound, n_mfcc=40))
    mfcc = numpy.reshape(mfcc, (mfcc.shape[1], mfcc.shape[0]))
    return mfcc