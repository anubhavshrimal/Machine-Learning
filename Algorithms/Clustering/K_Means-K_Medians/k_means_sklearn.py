import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
Titanic Dataset column details:-

Pclass - Passanger Class (1 = 1st class; 2 = 2nd class; 3 = 3rd class)
survival - Survived (0 = No; 1 = Yes)
name - Name of the passanger
sex - Gender of the passanger
sibsp - Number of siblings/spouses on board
parch - Number of parents/children on board
ticket - ticket number
fare - Passanger fare (british pounds)
cabin - Cabin
embarked - Port of embarkation (C = Cherbourg; Q = Queenstow; S = Southhampton)
boat - Lifeboat number
body - Body Identification Number
home.dest - Home or Destination
'''

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

clf = KMeans(n_clusters=2)
clf.fit(X)
labels = clf.labels_

correct = 0
for i in range(len(y)):
    if y[i] == labels[i]:
        correct += 1

print('Accuracy of classifying dead or alive:', correct/len(y))

'''
style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['r.', 'g.', 'b.', 'y.']*10

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=25)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=100)
plt.show()
'''