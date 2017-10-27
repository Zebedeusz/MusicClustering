import numpy

from IO import read_features_from_file, read_valence_arousal

def clustering_checks(features_path, features_file_name, save_path):
    numpy.random.seed(42)

    features = read_features_from_file(path=features_path + features_file_name  + ".csv")
    labels_encoded = cluster_k_means(features,5, False)

    val_ar = read_valence_arousal(False)
    labels_gt = cluster_k_means(val_ar,5, False)
    from sklearn.metrics import silhouette_score
    with open(save_path + ".txt", 'w') as f:
        f.write("VA number of clusters: " + cluster_sizes(labels_gt)+"\n")
        f.write("VA silhuette: " + str(silhouette_score(val_ar, labels_gt)) +"\n")
        f.write("features number of clusters: " + cluster_sizes(labels_encoded)+"\n")
        f.write("features silhuette: " + str(silhouette_score(features, labels_encoded))+"\n")
        f.close()

def cluster_sizes(labels):
    cluster_sizes = []
    for i in range(len(set(labels))):
        cluster_sizes.append(0)
    for l in labels:
        cluster_sizes[l] += 1
    cluster_sizes_str = ""
    for e in cluster_sizes:
        cluster_sizes_str += str(e) + ", "
    return cluster_sizes_str

def cluster_k_means(data,n_clusters, plot):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(data)

    print("cluster sizes: {}".format(cluster_sizes(k_means.labels_)))
    print("silhuette: {}".format(silhouette_score(data, k_means.labels_)))

    if plot and len(data[0]) == 2:
        import matplotlib.pyplot as plt

        k_m_l = numpy.asarray(k_means.labels_, dtype="float32")
        k_m_l = numpy.reshape(k_m_l, (-1,1))
        data_labaled = numpy.hstack((data, k_m_l))
        colour = "r."

        for d in data_labaled:
            if d[2] == 0:
                colour = "g."
            elif d[2] == 1:
                colour = "b."
            elif d[2] == 2:
                colour = "y."
            elif d[2] == 3:
                colour = "m."
            elif d[2] == 4:
                colour = "c."
            else:
                colour = "r."

            plt.plot(d[0],d[1],colour)

        plt.ylabel('arousal')
        plt.xlabel('valence')
        plt.savefig("/home/michal/PycharmProjects/AudioFeatureExtraction/charts/data_k_means_vis.png")
        plt.show()

    return k_means.labels_
    #return cluster_sizes(k_means.labels_), silhouette_score(data, k_means.labels_)

def cluster_dbscan(data, eps, min_samples):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    print("number of clusters: {}".format(len(set(dbscan.labels_))))
    print("cluster sizes: {}".format(cluster_sizes(dbscan.labels_)))
    print("silhuette: {}".format(silhouette_score(data, dbscan.labels_)))

    if len(data[0]) == 2:
        import matplotlib.pyplot as plt

        k_m_l = numpy.asarray(dbscan.labels_, dtype="float32")
        k_m_l = numpy.reshape(k_m_l, (-1,1))
        data_labaled = numpy.hstack((data, k_m_l))
        colour = "r."

        for d in data_labaled:
            if d[2] == 0:
                colour = "g."
            elif d[2] == 1:
                colour = "b."
            elif d[2] == 2:
                colour = "y."
            elif d[2] == 3:
                colour = "m."
            elif d[2] == 4:
                colour = "c."
            else:
                colour = "r."

            plt.plot(d[0],d[1],colour)

        plt.show()