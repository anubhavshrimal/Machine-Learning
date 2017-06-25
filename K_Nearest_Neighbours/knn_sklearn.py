import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
# Load the data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')

# Replace the unassigned values with -99999 and drop the id column
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Get the features and labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load Classifier of K Nearest Neighbours
pickle_file_name = 'knn_sklearn.pickle'

# If pre-trained pickle exists load it
if os.path.isfile('./' + pickle_file_name):
    # Load the classifier from the pre-trained pickle
    pickle_file = open(pickle_file_name, 'rb')
    clf = pickle.load(pickle_file)
# Otherwise train the classifier and save it in a pickle
else:
    # Default k = 5
    clf = neighbors.KNeighborsClassifier()

    # train the model on training data
    clf.fit(X_train, y_train)

    # save the pickle
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(clf, f)

# Get the accuracy on the testing data
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Make predictions on some random values
example_dataset = np.array([[8,3,3,1,2,3,4,4,2], [3,3,2,3,1,3,5,4,1], [5,3,3,1,2,2,4,3,1]])
# example_dataset = example_dataset.reshape(len(example_dataset), -1)
prediction = clf.predict(example_dataset)
print(prediction)
