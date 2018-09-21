import pandas as pd
import quandl
import pickle
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "Nfj9bb-tke8UTyjhSz1H"

# df = quandl.get('WIKI/GOOGL')
# pickle.dump(df, open('google_data.pickle', 'wb'))

df = pickle.load(open("google_data.pickle", "rb"))

# only keep usefull features
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

# Adj.open is the price at opening
# Adj.close is the price at closing
# A meaningfull feature would be adj.open - adj.close  (the volatility of the price / daily percent change):

df['PC_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# same here for the percentage of change btw the highest value and the close value
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

# again, keep only the meaningfull features
df = df [['Adj. Close', 'HL_PCT','PC_change', 'Adj. Volume']]

# we will try to predict the price in the futur
forecast_col = 'Adj. Close'
# if one of the data column is missing, replace it with -99999 so it should be considered as an outlier, and tho ignored
df.fillna(-99999, inplace=True)

# predict the next 10% of the available days
forecast_out = int(math.ceil(0.01 * len(df)))
# print(forecast_out)
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

df.dropna(inplace=True)
X = np.array(df.drop(["label"], 1))
y = np.array(df["label"])
X = preprocessing.scale(X)
y = np.array(df["label"])

# print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.1)

clf = LinearRegression()
# clf = svm.SVR()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
