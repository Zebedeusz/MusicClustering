import numpy

from IO import read_features_from_file, read_valence_arousal


def clustering_checks(features_path, features_file_name, save_path):
    numpy.random.seed(42)

    features = read_features_from_file(path=features_path + features_file_name + ".csv")
    labels_encoded = cluster_k_means(features, 5, False)

    val_ar = read_valence_arousal(False)
    labels_gt = cluster_k_means(val_ar, 5, False)
    from sklearn.metrics import silhouette_score
    with open(save_path + ".txt", 'w') as f:
        f.write("VA number of clusters: " + cluster_sizes(labels_gt) + "\n")
        f.write("VA silhuette: " + str(silhouette_score(val_ar, labels_gt)) + "\n")
        f.write("features number of clusters: " + cluster_sizes(labels_encoded) + "\n")
        f.write("features silhuette: " + str(silhouette_score(features, labels_encoded)) + "\n")
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


def clustered_groups(data, labels):
    clusters = []
    for i in range(len(set(labels))):
        clusters.append([])

    for i in range(len(labels)):
        clusters[labels[i]].append(data[i])

    return clusters

def cluster_k_means(data, n_clusters, plot=False, stats_save_path=False):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from jqmcvi.base import davisbouldin, dunn_fast
    import numpy

    data = numpy.nan_to_num(data)
    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(data)

    if len(k_means.labels_) > 1:
        sill = silhouette_score(data, k_means.labels_)
        dunn = dunn_fast(data, k_means.labels_)
        davisb = davisbouldin(clustered_groups(data, k_means.labels_), k_means.cluster_centers_)
    else:
        sill = 0
        dunn = 0
        davisb = 0

    print("cluster sizes: {}".format(cluster_sizes(k_means.labels_)))
    print("silhuette: {}".format(sill))
    print("dunn: {}".format(dunn))
    print("davies_bouldin: {}".format(davisb))

    if stats_save_path:
        import csv
        with open(stats_save_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows([["cluster sizes", cluster_sizes(k_means.labels_)],
                              ["silhuette", sill]])

    if plot and len(data[0]) == 2:
        import matplotlib.pyplot as plt

        k_m_l = numpy.asarray(k_means.labels_, dtype="float32")
        k_m_l = numpy.reshape(k_m_l, (-1, 1))
        data_labaled = numpy.hstack((data, k_m_l))

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

            plt.plot(d[0], d[1], colour)

        plt.ylabel('arousal')
        plt.xlabel('valence')
        plt.savefig("/home/michal/PycharmProjects/AudioFeatureExtraction/charts/data_k_means_vis.png")
        plt.show()

    return k_means.labels_, cluster_sizes(k_means.labels_), sill, dunn, davisb
    # return cluster_sizes(k_means.labels_), silhouette_score(data, k_means.labels_)


def cluster_dbscan(data, eps, min_samples, stats_save_path=False):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    from jqmcvi.base import dunn_fast
    import numpy

    data = numpy.nan_to_num(data)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    if len(set(dbscan.labels_)) > 1:
        print(len(dbscan.labels_))
        sil = silhouette_score(data, dbscan.labels_)
        dunn = dunn_fast(data, dbscan.labels_)
    else:
        sil = 0
        dunn = 0

    print("number of clusters: {}".format(len(set(dbscan.labels_))))
    print("cluster sizes: {}".format(cluster_sizes(dbscan.labels_)))
    print("silhuette: {}".format(sil))
    print("dunn: {}".format(dunn))

    if stats_save_path:
        import csv
        with open(stats_save_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows([["cluster sizes", cluster_sizes(dbscan.labels_)],
                              ["silhuette", sil]])

    if len(data[0]) == 2:
        import matplotlib.pyplot as plt

        k_m_l = numpy.asarray(dbscan.labels_, dtype="float32")
        k_m_l = numpy.reshape(k_m_l, (-1, 1))
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

            plt.plot(d[0], d[1], colour)

        plt.show()

    return dbscan.labels_, cluster_sizes(dbscan.labels_), sil, dunn


def cluster_som(data, n_clusters, stats_save_path=False):
    import sompy
    from sklearn.metrics import silhouette_score

    mapsize = [40, 40]
    som = sompy.SOMFactory.build(data, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var',
                                 initialization='pca', neighborhood='gaussian', training='batch',
                                 name='sompy')
    som.train(n_job=1, verbose='info')

    # v = sompy.mapview.View2DPacked(50, 50, title="")
    # v.show(som, what='codebook', which_dim=[0, 1], cmap=None, col_sz=6)

    # som.component_names = ['1', '2']
    # v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6)

    # v = sompy.mapview.View2DPacked(2, 2)
    cl = som.cluster(n_clusters=n_clusters)

    labels = getattr(som, 'cluster_labels')

    if len(labels) > 1:
        sil = silhouette_score(data, labels)
    else:
        sil = 0

    if stats_save_path:
        import csv
        with open(stats_save_path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerows([["cluster sizes", cluster_sizes(labels)],
                              ["silhuette", sil]])

    return labels, cluster_sizes(labels), sil


def cluster_hierarchical(data):
    import numpy
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    numpy.random.seed(0)

    Z = linkage(data, "ward")
    c, coph_dists = cophenet(Z, pdist(data))

    # dendrogram(
    #     Z,
    #     leaf_rotation=90.,  # rotates the x axis labels
    #     leaf_font_size=8.,  # font size for the x axis labels
    # )

    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=30,  # show only the last p merged clusters
        show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    plt.show()
