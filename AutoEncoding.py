import numpy
import csv

from ConvAutoEncoder import ConvAutoEncoder
from IO import load_data_from_file


def train_test_net_and_save_features(load_path, save_path, result_file_name, num_features, optimizer, loss_function, normalize_batch):
    data = load_data_from_file(load_path)
    data = numpy.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))

    train_data = data[0:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):len(data)]

    sfr = ConvAutoEncoder()

    sfr.initialiseNet(data.shape[1], data.shape[2], False, normalize_batch, optimizer, num_features, loss_function)
    sfr.trainNet(train_data)
    sfr.visualise_history(save_path, result_file_name)

    result = sfr.test_net(test_data)
    f = open(save_path + optimizer + result_file_name + ".txt", "w")
    for e in result:
        f.write(e.astype('str'))
        f.write("\n")
    f.close()

    features = sfr.get_features(data)
    with open(save_path + result_file_name + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerows(features)

def cross_validate_net():
    data = load_data_from_file()
    data = numpy.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
    sfr = ConvAutoEncoder()
    sfr.cross_validate_net(data, 10)


"""
numpy.random.seed(42)
save_path = "/home/michal/PycharmProjects/AudioFeatureExtraction/charts/random_tests/"


train_test_net_and_save_features(
                load_path="/home/michal/PycharmProjects/Datasets/1000_songs_dataset/clips_45seconds_spectrograms_float_100/",
                save_path=save_path,
                #result_file_name=str(num_specs) + "specs_" + str(num_features) + "f",
                result_file_name="1",
                num_features=100,
                optimizer="rmsprop",
                loss_function="hinge",
                normalize_batch = False)
clustering_checks(features_path=save_path,
                  features_file_name="1",
                  save_path=save_path + "1" + "_clustering_result")
"""