
def get_parent_directory(dir):
    slash_last_index = dir.rfind("/")
    if slash_last_index == dir.__len__() - 1:
        slash_last_index = dir.rfind("/", 0, dir.__len__() - 2)
    return str(dir).replace(dir[slash_last_index:], "")

# creates data_statistics folder based on last folder in path if it was not already created, returns it's path
def provide_data_statistics_path(path):
    stats_path = path + "/data_statistics"
    from pathlib import Path
    if not Path(stats_path).is_dir():
        Path(stats_path).mkdir()
    return stats_path

# finds all catalogues named "gmm_pkl" in root_catalogue path and saves them in a list - returns the list
def find_gmm_pkl_catalogues(root_catalogue):
    import os
    pkl_catalogues = []
    for (dirpath, dirnames, filenames) in os.walk(root_catalogue):
        for dirname in dirnames:
            if str(dirname).__eq__('gmm_pkl'):
                pkl_catalogues.append(str(root_catalogue + '/' + dirname))
            else:
                pkl_catalogues.extend(find_gmm_pkl_catalogues(str(root_catalogue + '/' + dirname)))
    pkl_catalogues = list(set(pkl_catalogues))
    pkl_catalogues.sort()
    return pkl_catalogues

#looks for catalouges named "gmm_pkl" in given directory,
# samples pkls in those catalouges and analyzes variance after one sampling
#results are saved at data_statstics folder
def analyse_mfcc_gmm_sampling_variance(catalogue):
    from Miscellaneous.Statistics import variance_distribution
    gmm_pkl_catalogues = find_gmm_pkl_catalogues(catalogue)
    for pkl_catalogue in gmm_pkl_catalogues:
        variance_distribution(
            pkl_catalogue,
            False,
            provide_data_statistics_path(get_parent_directory(pkl_catalogue)) + "/variance_distribution_from_gmms.png")

#looks for catalouges named "gmm_pkl" in given directory,
# samples pkls in those catalouges and analyzes variance after multiple sampling
#results are saved at data_statstics folder
def analyse_multiple_mfcc_gmm_sampling_variance(catalogue):
    from Miscellaneous.Statistics import variance_distribution_of_variances_from_gmms
    gmm_pkl_catalogues = find_gmm_pkl_catalogues(catalogue)
    for pkl_catalogue in gmm_pkl_catalogues:
        variance_distribution_of_variances_from_gmms(
            pkl_catalogue,
            100,
            provide_data_statistics_path(get_parent_directory(pkl_catalogue)) + "/variance_distribution_of_variances_from_gmms.png")

def analyse_fps_variance(catalogue):
    from Miscellaneous.Statistics import variance_distribution
    from IO import read_features_from_file
    import os

    fps_data_path = provide_data_statistics_path(catalogue)

    for (dirpath, dirnames, filenames) in os.walk(catalogue):
        for filename in filenames:
            if str(filename).__contains__(".csv"):
                fps = read_features_from_file(catalogue + "/" + filename, True)
                variance_distribution(fps, False, fps_data_path + "/" + str(filename ).replace(".csv", "") + "_variance_distribution_from_fp.png", "", 1)

# provided catalogue to .csv files with fp features and name of a dataset e.g. Aljanaki
# the function caclulates variance of fps for every file of provided dataset name in catalogue
# and saves .png in data_statistics folder
def analyse_fps_variance_for_specified_datasets(catalogue, dataset_name):
    from Miscellaneous.Statistics import variance_distribution
    from IO import read_features_from_file
    import os

    fps_data_path = provide_data_statistics_path(catalogue)

    dataset_csv_fp_files = []

    for (dirpath, dirnames, filenames) in os.walk(catalogue):
        for filename in filenames:
            if filename.__contains__(dataset_name):
                dataset_csv_fp_files.append(filename)

    fps = []
    for csv_fp_filename in dataset_csv_fp_files:
        if str(csv_fp_filename).__contains__(".csv"):
            fps.extend(read_features_from_file(catalogue + "/" + csv_fp_filename, True))

    variance_distribution(fps, False, fps_data_path + "/" + dataset_name + "_variance_distribution_from_fp.png", "", 1)


if __name__ == '__main__':
    #analyse_mfcc_gmm_sampling_variance('/media/michal/HDD/Music Emotion Datasets/Decoded')
    analyse_multiple_mfcc_gmm_sampling_variance('/media/michal/HDD/Music Emotion Datasets/Decoded')
    #analyse_fps_variance("/media/michal/HDD/Music Emotion Datasets/Decoded/Fluctuation Patterns")
    #analyse_fps_variance_for_specified_datasets("/media/michal/HDD/Music Emotion Datasets/Decoded/Fluctuation Patterns", "ISMIR2012")