
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

