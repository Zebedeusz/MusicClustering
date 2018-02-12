import numpy

def variance_for_every_column(data_array):
    data_array = numpy.array(data_array)
    data_array = data_array.transpose()
    variances = []
    for column in data_array:
        variances.append(numpy.var(column))
    return variances

def variance_distribution(data_path, plot = False, save_path = "", title="", markersize=3):
    from FeatureExtraction import get_gmms_samples_from_path

    samples = get_gmms_samples_from_path(data_path)
    variances = variance_for_every_column(samples)
    print(variances)

    if(plot or save_path):
        import matplotlib.pyplot as plt

        plt.plot(numpy.arange(1, variances.__len__() + 1, 1), variances ,'bo', markersize=markersize)
        plt.ylabel('variance')
        plt.xlabel('index')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        if plot:
            plt.show()

def variance_distribution_of_variances_from_gmms(data_path, gmm_samples, plot_save_path):
    from FeatureExtraction import get_gmms_samples_from_path

    variances = []
    for i in range(gmm_samples):
        print(str(i + 1)+ " / " + str(gmm_samples))
        samples = get_gmms_samples_from_path(data_path)

        variances.append(variance_for_every_column(samples))
    variance_distribution(variances, False, plot_save_path)


