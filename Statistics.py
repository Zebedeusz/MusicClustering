import numpy

def variance_for_every_column(data_array):
    data_array = numpy.array(data_array)
    data_array = data_array.transpose()
    variances = []
    for column in data_array:
        variances.append(numpy.var(column))
    return variances

def variance_distribution(data_array, plot = False, save_path = ""):
    import matplotlib.pyplot as plt

    variances = variance_for_every_column(data_array)
    print(variances)

    if(plot or save_path):
        plt.plot(numpy.arange(1, variances.__len__() + 1, 1), variances ,'bo')
        plt.ylabel('variance')
        plt.xlabel('index')
        if save_path:
            plt.savefig(save_path)
        if plot:
            plt.show()



