
#check return type and analyse code
def spectral_centroid(wavedata, window_size, sample_rate):
    import numpy as np
    from scipy.signal import stft

    magnitude_spectrum = stft(wavedata, window_size)

    timebins, freqbins = np.shape(magnitude_spectrum)

    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0 ,timebins - 1) * (timebins / float(sample_rate)))

    sc = []

    for t in range( timebins -1):

        power_spectrum = np.abs(magnitude_spectrum[t] ) **2

        sc_t = np.sum(power_spectrum * np.arange(1 , freqbins +1)) / np.sum(power_spectrum)

        sc.append(sc_t)


    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)

    return sc, np.asarray(timestamps)

def median_spectral_band_energy(wavedata, window_size):
    import numpy
    from scipy.signal import stft

    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = numpy.shape(magnitude_spectrum)

    #check if it's pwr spectrum for time bands or freq bands, how many bands
    power_spectrum = numpy.abs(magnitude_spectrum)**2

    return numpy.median(power_spectrum)

#compresses wav file to flac and returns size of the flac file as feature value
def compressibility_feature(filepath_wav):
    #wav -> flac converison
    import audiotools
    filepath_flac = filepath_wav.replace(".wav", ".flac")
    audiotools.open(filepath_wav).convert(filepath_flac,
                                          audiotools.FlacAudio, compression_quality)

    #return size of flac file

