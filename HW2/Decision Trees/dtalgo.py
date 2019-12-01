import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

dataset = pd.read_csv('page-blocks.data',sep="\s+", header= None)

X = dataset.iloc[:, :10]
y = dataset.iloc[:, 10]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
accuracy_arr = []

kfold = KFold(n_splits=10, shuffle=True)
for train, test in kfold.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=30, splitter="best")
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_arr.append(accuracy)

print accuracy_arr
print(str(np.mean(accuracy_arr) * 100) + "% accuracy with entropy criterion")


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)
# clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=30, splitter="best")
# clf = clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

accuracy_arr2 = []
kfold = KFold(n_splits=10, shuffle=True)
for train, test in kfold.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_split=30, splitter="best")
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_arr2.append(accuracy)

print accuracy_arr2
print(str(np.mean(accuracy_arr2) * 100) + "% accuracy with gini criterion")



