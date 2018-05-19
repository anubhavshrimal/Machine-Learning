from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

style.use('ggplot')


# Create random data points whose centers are the following
centers = [[20, 0, 0], [0, 20, 0], [0, 0, 20], [0, 0, 0]]
X, _ = make_blobs(n_samples=200, centers=centers, cluster_std=2)

# Fit the data into MeanShift classifier with search bandwidth = 10
clf = MeanShift(bandwidth=10)
clf.fit(X)

# Get the labels of each data point
# and cluster centers of the number of clusters formed
labels = clf.labels_
cluster_centers = clf.cluster_centers_
print(cluster_centers)
n_clusters = len(cluster_centers)
print('Number of clusters found:', n_clusters)

# Plot the data points with their clusters and centers on a 3d graph
colors = 10*['r', 'g', 'b', 'y', 'c']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
           marker='x', s=150, linewidth=5, zorder=10, color='k')

plt.show()
