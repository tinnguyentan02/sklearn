from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml
import ssl  # secure sockets layer or TLS (Transport Layer Security)
# boston = datasets.load_boston()
ssl._create_default_https_context = ssl._create_unverified_context
boston = fetch_openml(name="boston", version = 1, as_frame = True)

# features / labels
X = boston.data
y = boston.target

#algorithm
l_reg = linear_model.LinearRegression() 


# plt.scatter(X.iloc[:, 0], y)  # [:, 0] choose all rows of first column
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# train
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
# print("Predictions: ", predictions)
# print("R^2 value: ", l_reg.score(X, y))
# print("coedd: ", l_reg.coef_)
# print("intercept: ", l_reg.intercept_)


