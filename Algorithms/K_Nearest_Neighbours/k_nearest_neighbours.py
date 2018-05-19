import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import style
from collections import Counter
import warnings
import random
import pandas as pd


def k_nearest_neighbors(data, predict, k=5):
    # If k is greater than total number of points in the dataset
    if k > sum(len(v) for v in data.values()):
        warnings.warn('K is set to a value more than total data points!')
    distances = []

    for label in data:
        for features in data[label]:
            # Calculate the eucledian distance between all the features and prediction points
            # euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            # Append the distance with the label of the features
            distances.append([euclidean_distance, label])

    votes = [i[1] for i in sorted(distances)[:k]]
    # Find the most common label from the k closest points
    vote_results = Counter(votes).most_common(1)[0][0]

    # Find the confidence of prediction
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_results, confidence


# Load the data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Replace the unassigned values with -99999 and drop the id column
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Extract the data without the table headers in float format
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

# Divide the data set into 80% training and 20% testing data
test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

# Append the training data into the dict train_set without the last column of labels
for i in train_data:
    train_set[i[-1]].append(i[:-1])

# Append the testing data into the dict test_set without the last column of labels
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for label in test_set:
    for predict in test_set[label]:
        vote, confidence = k_nearest_neighbors(train_set, predict, k=5)

        # If prediction is correct
        if vote == label:
            correct += 1
        # else print the confidence of the wrong prediction
        else:
            print(confidence)

        total += 1

print('Accuracy:', correct/total)

"""
# style.use('fivethirtyeight')

# {Label: [[data-point], [data-point], ...]}
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

prediction, confidence = k_nearest_neighbors(dataset, new_features, k=3)
print(prediction, confidence)

# Visualising dataset and the feature which we want to predict
[[plt.scatter(point[0], point[1], s=100, color=label) for point in dataset[label]] for label in dataset]
plt.scatter(new_features[0], new_features[1], color=prediction)
plt.show()
"""