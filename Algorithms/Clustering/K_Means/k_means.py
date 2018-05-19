import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from collections import defaultdict
from sklearn import preprocessing


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = 2
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        # Take first k points in data as centroids
        # can take random points as well
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = defaultdict(list)

            # Find to which centroid is the feature closest to and append it in that centroid's classification
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[c]) for c in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(features)

            prev_centroids = dict(self.centroids)

            # Calculate the average centroid point for each classification
            # by averaging the features in that classification
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # If the desired tolerance is achieved i.e. the centroids are not changing values
            # more than tolerance % then simply break the loop
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                new_centroid = self.centroids[c]
                if np.sum((new_centroid - original_centroid)/original_centroid) * 100.0 > self.tol:
                    optimized = False
            if optimized:
                break

    # used for predicting in which classification does the new data point or feature lie
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
        classification = distances.index(min(distances))

        return classification


df = pd.read_excel('../../datasets/titanic.xls')

df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)


def convert_non_numeric_data(df):
    columns = df.columns.values
    for col in columns:
        text_digits = {}

        def convert_to_int(val):
            return text_digits[val]

        if df[col].dtype != np.int64 or df[col].dtype != np.float64:
            col_contents = df[col].values.tolist()
            unique_elements = set(col_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_digits:
                    text_digits[unique] = x
                    x += 1
            df[col] = list(map(convert_to_int, df[col]))

    return df

df = convert_non_numeric_data(df)

X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):
    feature = np.array(X[i].astype(float))
    prediction = clf.predict(feature)
    if prediction == y[i]:
        correct += 1

print(correct/len(X))

'''
style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

colors = ['r', 'g', 'b']*10

clf = K_Means()
clf.fit(X)

for c in clf.centroids:
    plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for features in clf.classifications[classification]:
        plt.scatter(features[0], features[1], marker='x', s=150, color=color, linewidths=5)

unknowns = np.array([[1, 3],
              [3, 5],
              [3, 7],
              [-3, -1],
              [0, 0],
              [8, 9]])

for u in unknowns:
    classification = clf.predict(u)
    plt.scatter(u[0], u[1], marker='*', color=colors[classification], s=150, linewidths=5)
plt.show()
'''