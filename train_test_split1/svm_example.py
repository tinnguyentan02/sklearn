from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
# split it in features and labels

X = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']
print(X.shape)
print(y.shape)

# hours of study vs good/bad grades
# 10 different students 
# train with 8 students and predict with the remaining 2
# level of accuracy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = svm.SVC()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape) 


