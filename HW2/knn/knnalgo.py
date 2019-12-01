import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

name = ["height", "length", "area", "eccen", "p_black", "p_and", "mean_tr", "blackpix", "blackand", "wbtrans", "Classification"]
df = pd.read_csv('page-blocks.data', sep = "\s+",header = None, names=name)
#preprocessing
X = df.iloc[:, :-1]
y = df.iloc[:, 10]

#scaling the initial dataset
from sklearn.preprocessing  import StandardScaler
scaler  = StandardScaler()
scaler.fit(df.drop('Classification', axis=1))
scaled_feature = scaler.transform(df.drop('Classification', axis=1)) #transform: perform standardization by centuring and scaling
df_feat =  pd.DataFrame(scaled_feature, columns=df.columns[:-1])

#splitting the initial data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_feature, y, test_size=0.30)


#USE KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

#calculate the error rate of k values
error_rate = []
#
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
"""for i in range(len(error_rate)):
    print(error_rate[i])"""
plt.plot(range(1,40), error_rate, color='blue', linestyle= 'dashed', marker='o',markerfacecolor='red',markersize=10)
plt.title("Error Rate vs K Value")
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show() #show the error plot

knn = KNeighborsClassifier(n_neighbors=3) #3 since it provides the best accuracy but with relatively high bias
#K Fold Cross Validation

kfold = KFold(n_splits=10, shuffle=True)
crossvalScore = []
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
for train, test in kfold.split(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    sc = MinMaxScaler() #for normalization
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    model = knn.fit(X_train, y_train)
    prediction = knn.predict(X_test)
    crossvalScore.append(model.score(X_test, y_test))
    # print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))

#score of each model in k fold cross validation and the average accuracy rate of the models
print("Score from each iteration: ", crossvalScore)
print("Average K Fold Score: ", np.mean(crossvalScore))


