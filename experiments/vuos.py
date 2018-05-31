# parameters
cluster_sizes = [3, 5, 8, 10]
eps_values = [0.01, 0.1, 0.5, 1, 2, 85, 100, 120, 150, 175, 200, 220, 250, 280, 300]
min_samples_qnts = [2, 5, 10, 20]

if __name__ == '__main__':
    from experiments.helper import conduct_experiments

    conduct_experiments("Vuoskoski", cluster_sizes, eps_values, min_samples_qnts)
