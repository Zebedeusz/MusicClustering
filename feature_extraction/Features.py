def delta_spectral_pattern(wavedata_preprocessed):
    pass


def spectral_pattern(wavedata_preprocessed):
    import numpy
    import math

    soundP = numpy.array(wavedata_preprocessed)

    block_size = 10
    hop_size = 5

    sound_blocks_with_sorted_freq_bands = []
    for i in range(0, soundP[0].size, hop_size):
        sound_block = soundP[:, i:(i + block_size)]
        sorted_sound_block = []
        for freq_band in sound_block:
            sorted_sound_block.append(sorted(freq_band))
        sound_blocks_with_sorted_freq_bands.append(sorted_sound_block)

    summed_sound_blocks_with_sorted_freq_bands = []
    for sound_block in sound_blocks_with_sorted_freq_bands:
        summed_freq_bands = []
        for freq_band in sound_block:
            summed_freq_bands.append(numpy.sum(freq_band))
        summed_sound_blocks_with_sorted_freq_bands.append(numpy.sum(summed_freq_bands))

    sorted_summed_sound_blocks_with_sorted_freq_bands = sorted(summed_sound_blocks_with_sorted_freq_bands)
    perc_index = math.ceil(0.9 * len(sorted_summed_sound_blocks_with_sorted_freq_bands))
    perc = sound_blocks_with_sorted_freq_bands[perc_index]

    return perc


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
        time_block = data[time_block_index]
        # converting every time_block from ndarray with list of lists to ndarray with list of ndarrays
        for i in range(len(time_block)):
            time_block[i] = numpy.array(time_block[i])
        time_block = numpy.array(data[time_block_index])

        time_block = numpy.reshape(time_block, mean_block.shape)
        variance_init += numpy.power(time_block - mean_block, 2)

    return variance_init / time_blocks

def meanblock(data):
    import numpy
    time_blocks = len(data)

    summed_block = numpy.zeros((len(data[0]), len(data[0][0])))
    for time_block in range(time_blocks - 1):
        summed_block = sum_two_arrays(summed_block, data[time_block])

    mean_block = summed_block / time_blocks

    return mean_block


def sum_two_arrays(arr1, arr2):
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