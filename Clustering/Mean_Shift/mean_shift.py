import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.datasets.samples_generator import make_blobs
from collections import defaultdict
style.use('ggplot')

# n_samples = Number of data points
# centers = Number of groups or classifications
# n_features = Number of features in each sample
X, y = make_blobs(n_samples=30, centers=4, n_features=2)

colors = ['r', 'g', 'b', 'y', 'c']*10


class Mean_Shift:
    # TODO: Bandwith should be calculated dynamically rather than hardcoded
    def __init__(self, bandwidth=4):
        self.bandwidth = bandwidth

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for feature in data:
                    if np.linalg.norm(feature - centroid) < self.bandwidth:
                        in_bandwidth.append(feature)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break

            if optimized:
                break

        self.centroids = centroids

        # Find to which centroid is the feature closest to and append it in that centroid's classification
        self.classifications = defaultdict(list)
        for feature in data:
            distances = [np.linalg.norm(feature - self.centroids[c]) for c in self.centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(feature)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        classification = distances.index(min(distances))

        return classification


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for features in clf.classifications[classification]:
        plt.scatter(features[0], features[1], marker='o', color=color, linewidths=5)

for i in centroids:
    plt.scatter(centroids[i][0], centroids[i][1], marker='x', color='k', s=150)

plt.show()