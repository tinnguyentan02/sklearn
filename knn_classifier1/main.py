import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("knn_classifier1/car.data")

X = data[[
    "buying",
    "paint",
    "safety"
]].values
y = data[['class']]
# print(X, y)

# converting the data

#x
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print(X)

#y
# print(np.unique(y.flatten()))

label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
print(np.array(y))

knn = neighbors.KNeighborsClassifier(n_neighbors = 25, weights = 'uniform')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print('Accuracy: ', accuracy)
print(prediction)
print(y_test)

