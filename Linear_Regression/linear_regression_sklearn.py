import quandl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

"""
# Needed if more than 50 request/day
quandl.ApiConfig.api_key = "Quandl_API_KEY"
"""
# Get the data set from quandl
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Refine the data set to our needs
df['HL_Percent'] = ((df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']) * 100
df['Percent_Change'] = ((df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']) * 100

# Extract only the relevant features
df = df[['Adj. Close', 'HL_Percent', 'Percent_Change', 'Adj. Volume']]

# prediction column
forecast_col = 'Adj. Close'

# Replace null values in the dataset with a very small value so it has the least impact
df.fillna(-99999, inplace=True)

# Number of days in future that we want to predict the price for
future_days = 10

# define the label as Adj. Close future_days ahead in time
# shift Adj. Close column future_days rows up i.e. future prediction
df['label'] = df[forecast_col].shift(-future_days)

# Get the features array in X
X = np.array(df.drop(['label'], 1))

# Regularize the data set across all the features for better training
X = preprocessing.scale(X)

# Extract the last future_days rows for prediction as they don't have the values due to the shift
predict_X = X[-future_days:]

# Get the data for training and testing
X = X[:-future_days]

# Drop the last future_days rows as there is no label for them because we shifted the column up
df.dropna(inplace=True)

# Get the labels in y
y = np.array(df['label'])

# Shuffle the data and get Training and Testing data
# Testing data = 20% of total data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load Classifier of linear regression
pickle_file_name = 'linear_regression_sklearn.pickle'

# If pre-trained pickle exists load it
if os.path.isfile('./' + pickle_file_name):
    # Load the classifier from the pre-trained pickle
    pickle_file = open(pickle_file_name, 'rb')
    clf = pickle.load(pickle_file)
# Otherwise train the classifier and save it in a pickle
else:
    # n_jobs = -1 means training the model parallely, as many jobs as possible
    clf = LinearRegression(n_jobs=-1)

    # train the model on training data
    clf.fit(X_train, y_train)

    # save the pickle
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(clf, f)


# Test the accuracy of the data on the testing data set
# How well is the model predicting the future prices
accuracy = clf.score(X_test, y_test)
print('Accuracy on test data:', accuracy)

predictions = clf.predict(predict_X)
print('Predictions for next 10 days: ')
print(predictions)







