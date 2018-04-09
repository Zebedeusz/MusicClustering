def spectral_contrast_pattern(wavedata):
    from librosa.feature import spectral_contrast

    block_size = 40
    hop_size = 20
    freq_bands = 20
    percentile = 0.1

    # wavedata_with_reduced_freq_bands = reduce_frequency_bands(wavedata_preprocessed, freq_bands)

    scp = spectral_contrast(y=wavedata)

    print(scp)


def correlation_pattern(wavedata_preprocessed):
    import numpy

    block_size = 256
    hop_size = 128
    freq_bands = 52
    percentile = 0.5

    wavedata_with_reduced_freq_bands = reduce_frequency_bands(wavedata_preprocessed, freq_bands)

    # calculating Pearson Correlation for every time block
    from scipy.stats import pearsonr
    all_coeffs = []
    for i in range(0, wavedata_with_reduced_freq_bands[1].size, hop_size):
        sound_block = wavedata_with_reduced_freq_bands[:, i:(i + block_size)]
        coeffs = numpy.zeros(shape=(freq_bands, freq_bands))
        for j in range(freq_bands):
            for k in range(freq_bands):
                if j == k:
                    coeffs[j, k] = 1
                elif j > k:
                    coeffs[j, k] = coeffs[k, j]
                else:
                    coeffs[j, k] = \
                        pearsonr(numpy.array(sound_block[j], dtype=numpy.float64),
                                 numpy.array(sound_block[k], dtype=numpy.float64))[0]
        coeffs = numpy.nan_to_num(coeffs)
        all_coeffs.append(coeffs)
    all_coeffs = numpy.array(all_coeffs)

    # percentile as 2d-array with highest sum of it's elements
    import math
    summed_all_coeffs = []
    for coeffs in all_coeffs:
        summed_all_coeffs.append(numpy.sum(coeffs))
    sorted_summed_all_coeffs = sorted(summed_all_coeffs)
    perc_index = math.ceil(percentile * len(sorted_summed_all_coeffs))
    perc = all_coeffs[perc_index]

    return perc


def reduce_frequency_bands(wavedata, freq_bands):
    import numpy

    freq_band_size = len(wavedata) // freq_bands
    wavedata_with_reduced_freq_bands = numpy.zeros(shape=(freq_bands, wavedata.shape[1]))

    for freq_band_index in range(freq_bands - 1):
        wavedata_with_reduced_freq_bands[freq_band_index] = \
            numpy.sum(
                wavedata[freq_band_index * freq_band_size:(freq_band_index + 1) * freq_band_size, :], axis=0) \
            / freq_band_size

    return wavedata_with_reduced_freq_bands

def variance_delta_spectral_pattern(wavedata_preprocessed):
    return spectral_pattern_base(wavedata_preprocessed,
                                 3,
                                 25,
                                 5,
                                 True,
                                 False)

def delta_spectral_pattern(wavedata_preprocessed):
    return spectral_pattern_base(wavedata_preprocessed,
                                 3,
                                 25,
                                 5,
                                 False,
                                 0.9)


def spectral_pattern(wavedata_preprocessed):
    return spectral_pattern_base(wavedata_preprocessed,
                                 False,
                                 10,
                                 5,
                                 False,
                                 0.9)


def spectral_pattern_base(wavedata_preprocessed,
                          delta,
                          block_size,
                          hop_size,
                          variance_summarization,
                          percentile_summarization):
    import numpy
    import math

    soundP = numpy.array(wavedata_preprocessed)

    if (delta):
        soundToSub = soundP[delta:]
        for i in range(delta):
            soundToSub = numpy.vstack((soundToSub, numpy.zeros_like(soundP[0])))
        soundP = numpy.abs(soundP - soundToSub)

    sound_blocks_with_sorted_freq_bands = numpy.zeros(shape=(soundP[1].size // hop_size + 1, len(soundP), block_size))
    k = 0
    for i in range(0, soundP[1].size, hop_size):
        sound_block = soundP[:, i:(i + block_size)]
        sorted_sound_block = numpy.zeros(shape=(len(soundP), block_size))
        for j in range(len(sound_block)):
            sorted_sound_block[j, 0:len(sound_block[j])] = sorted(sound_block[j])
        sound_blocks_with_sorted_freq_bands[k] = sorted_sound_block
        k += 1

    if (variance_summarization):
        sound_blocks_with_sorted_freq_bands = numpy.array(sound_blocks_with_sorted_freq_bands)
        variance_of_time_blocks = varianceblock(sound_blocks_with_sorted_freq_bands)
        return variance_of_time_blocks

    if (percentile_summarization):
        summed_sound_blocks_with_sorted_freq_bands = []
        for sound_block in sound_blocks_with_sorted_freq_bands:
            summed_freq_bands = []
            for freq_band in sound_block:
                summed_freq_bands.append(numpy.sum(freq_band))
            summed_sound_blocks_with_sorted_freq_bands.append(numpy.sum(summed_freq_bands))

        sorted_summed_sound_blocks_with_sorted_freq_bands = sorted(summed_sound_blocks_with_sorted_freq_bands)
        perc_index = math.ceil(percentile_summarization * len(sorted_summed_sound_blocks_with_sorted_freq_bands))
        perc = sound_blocks_with_sorted_freq_bands[perc_index]

        return perc


def varianceblock(data):
    import numpy

    mean_block = meanblock(data)
    time_blocks = len(data)

    variance_init = 0

    for time_block_index in range(time_blocks):
        variance_init += numpy.power(data[time_block_index] - mean_block, 2)

    return variance_init / time_blocks

def meanblock(data):
    import numpy
    time_blocks = len(data)

    summed_block = numpy.zeros((len(data[0]), len(data[0][0])))
    for time_block in range(time_blocks - 1):
        summed_block = sum_2d_arrays(summed_block, data[time_block])

    mean_block = summed_block / time_blocks

    return mean_block


def sum_2d_arrays(arr1, arr2):
    for i in range(len(arr2)):
        for j in range(len(arr2[0])):
            arr1[i][j] = arr1[i][j] + arr2[i][j]

    return arr1

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