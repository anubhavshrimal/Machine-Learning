import numpy as np
from collections import defaultdict
from copy import deepcopy


class RepresentativeClustering:
    def __init__(self, random=True, max_itr=200, tolerance=0.001):
        """
        :param max_itr: Maximum number of iterations if convergence is not reached. default = 200
        :param tolerance: Percent change allowed in updated centroids and old centroids. default = 0.1%
        :param random: Tells whether to initialize seeds randomly or manually. default True
        """
        self.max_itr = max_itr
        self.tolerance = tolerance
        self.random = random

    def clustering(self, dataset, k, type):
        """
        Creates k clusters in a dataset
        :param dataset: list of data points
        :param k: number of clusters
        :param type: type of clustering algorithm to be applied (1: "k-means", 2: "k-medians")
        :return: set of k clusters found in the dataset
        """

        centroids = self.initialize_cluster_representatives(k, dataset)
        clusters = None

        for i in range(self.max_itr):
            clusters = defaultdict(list)
            for data_point in dataset:
                # calculate distances of data point from all centroids
                distances = self.distance_function(centroids, data_point, type)

                # assign data_point to the closes centroid
                closest_index = distances.index(min(distances))
                clusters[closest_index].append(data_point)

            prev_centroids = deepcopy(centroids)
            print(f"Iteration {i+1}: ")
            print_instance(centroids, clusters)
            # find the new centroid
            self.update_centroids(centroids, clusters, type)

            # check if convergence is reached
            if self.converged(prev_centroids, centroids):
                break

        return centroids, clusters

    def distance_function(self, centroids, data_point, type):
        """
        Apply distance function over data point and centroids
        :param centroids: set of centroids
        :param data_point: one data point from the dataset
        :param type: algorithm applied (k-means or k-medians)
        :return: returns the distance array of data-point from respective centroids
        """
        if type == 1:
            return self.euclidean_distance(data_point, centroids)
        if type == 2:
            return self.manhattan_distance(data_point, centroids)

    @staticmethod
    def update_centroids(centroids, clusters, type):
        """
        Updates the centroids based on the clusters formed
        :param centroids: set of centroid
        :param clusters: set of clusters belonging to each centroids
        :param type: algorithm applied (k-means or k-medians)
        """
        for c_indx in clusters:
            if type == 1:
                centroids[c_indx] = np.average(clusters[c_indx], axis=0)
            if type == 2:
                centroids[c_indx] = np.median(clusters[c_indx], axis=0)

    def converged(self, prev_centroids, centroids):
        """
        Check if the % change of previous centroids & new centroids is greater than the tolerance
        :param prev_centroids:
        :param centroids:
        :return:
        """
        converge = True
        for c in centroids:
            new_centroid = centroids[c]
            old_centroid = prev_centroids[c]
            if np.sum(np.abs((new_centroid - old_centroid) / old_centroid)) * 100 > self.tolerance:
                converge = False
                break
        return converge

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return [np.linalg.norm(data_point - centroids[c]) for c in centroids]

    @staticmethod
    def manhattan_distance(data_point, centroids):
        return [np.sum(np.abs(data_point - centroids[c]), axis=0) for c in centroids]

    def initialize_cluster_representatives(self, k, dataset):
        """
        Gets the initial cluster representatives as input
        :param k: Number of cluster representatives
        :param dataset: list of data points
        :return: k initial cluster representatives
        """
        centroids = {}
        if not self.random:
            print(f"Enter {k} cluster representatives [Separated by newline, with "
                  f"{len(dataset[0])} dimensions (, separated)]: ")
            for i in range(k):
                centroids[i] = np.array(list(map(int, input().strip().split(","))))
        else:
            # Randomly initialized centroids from data points
            for i, data_index in enumerate(np.random.randint(0, len(dataset), k)):
                centroids[i] = deepcopy(dataset[data_index])

            # Randomly initialized centroids from data space
            """
            for j in range(k):
                temp = []
                for i in range(len(dataset[0])):
                    temp.append(np.random.randint(np.min(dataset[:, i]), np.max(dataset[:, i]+1)))
                centroids[j] = np.array(temp)
            """

        print("Initial Centroids: ")
        for j in centroids:
            print(f'Centroid {j+1}: {centroids[j]}')
        print()

        return centroids


def print_instance(centroids, clusters):
    print()
    for c in clusters:
        print(f'Cluster {c+1}: ')
        print('\tCentroid:', centroids[c])
        print('\tCluster:', *clusters[c])
    print("--------------------------------------")


file = input("Enter input file name: ")
dataset = []

try:
    with open(file) as f:
        for line in f:
            row = list(map(float, line.strip().split(",")))
            dataset.append(row)
except FileNotFoundError:
    print("Incorrect file name")
    exit(0)

dataset = np.array(dataset)

# Make an object of RepresentativeClustering class
rc = RepresentativeClustering()

k = int(input("Enter number of clusters to be created: "))

choice = int(input("Select the clustering algorithm you want to use:\n 1. K-Means 2. K-Medians\n"))
if 1 <= choice <= 2:
    centroids, clusters = rc.clustering(dataset, k, choice)

    print("Final Clusters found:")
    print_instance(centroids, clusters)

else:
    print("Invalid choice")
