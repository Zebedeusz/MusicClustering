# returns array of shape (f,t) e.g. (1025,1295)
def preprocessToKeplerUniFeatures(sound):
    from scipy.signal import stft
    import numpy

    f = 22050

    # samplerate to 22kHz
    # sound = sound.set_frame_rate(f)

    # STFT - window size 2048 samples, hop 512 samples, Hanning
    f, t, sound_stft = stft(sound, fs=1.0, window="hann", nperseg=2048, noverlap=512)

    # magnitude spectrum
    sound_stft_sole = abs(sound_stft)

    # linear resolution to logarithmic Cent scale
    cent_const = 440 * (pow(2, -57 / 12))
    sound_cent_scale = 1200 * numpy.log2(sound_stft_sole / cent_const)
    sound_cent_scale = numpy.nan_to_num(sound_cent_scale)

    # to logarithmic scale again
    sound_log_cent_scale = 20 * numpy.log10(sound_cent_scale)
    sound_log_cent_scale = numpy.nan_to_num(sound_log_cent_scale)

    # normalization
    # switched off as better image without it
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


def percentile_element_wise_for_3d_array(array, percentile):
    # percentile as 2d-array calculated element-wise
    import math
    import numpy

    perc = numpy.asarray(array[0])
    for i in range(perc.shape[0]):
        for j in range(perc.shape[1]):
            sorted_i_j_array = numpy.sort(array[:, i, j])
            perc[i, j] = sorted_i_j_array[math.floor(percentile * len(sorted_i_j_array))]

    return perc


def sort_freq_bands_in_blocks(array, block_size, hop_size):
    import numpy

    blocks_with_sorted_freq_bands = numpy.zeros(shape=(array[1].size // hop_size + 1, len(array), block_size))

    k = 0
    for i in range(0, array[1].size, hop_size):
        block = array[:, i:(i + block_size)]
        sorted_sblock = numpy.zeros(shape=(len(array), block_size))
        for j in range(len(block)):
            sorted_sblock[j, 0:len(block[j])] = sorted(block[j])
        blocks_with_sorted_freq_bands[k] = sorted_sblock
        k += 1

    return blocks_with_sorted_freq_bands


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
