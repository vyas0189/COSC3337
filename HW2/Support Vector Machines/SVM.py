import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
# from sklearn.preprocessing  import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(font_scale=1.2)

df = pd.read_csv('page-blocks.data', header = None, delim_whitespace=True)
df.columns = ['Height', 'Length', 'Area', 'Eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans', 'Type']

# Separating into Features and Classfication
features = df.iloc[:, :-1]
target = df.iloc[:, 10]

# Normalizing Our features
scaler = MinMaxScaler() 
df_feat = scaler.fit_transform(features)

# Spliting into a training and testing data set
Xtrain, Xtest, ytrain, ytest = train_test_split(df_feat, np.ravel(target), test_size=0.3, random_state=109) 
clf = svm.SVC(kernel='linear', C=2**2)
clf.fit(Xtrain, ytrain)

# Prediction
ypred = clf.predict(Xtest)
print("Accuracy: ", metrics.accuracy_score(ytest, ypred))

# K Fold Cross Validation
kfold = KFold(n_splits=10, shuffle=True)
crossvalScore = []
for train, test in kfold.split(features, target):

    clf = svm.SVC(kernel='linear', C=2**2)

    Xtrain, Xtest = features.iloc[train], features.iloc[test]
    ytrain, ytest = target[train], target[test]

    sc = MinMaxScaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.fit_transform(Xtest)

    model = clf.fit(Xtrain, ytrain)
    prediction = clf.predict(Xtest)
    crossvalScore.append(model.score(Xtest, ytest))
print("Score from each iteration: ")
for n, i in enumerate(crossvalScore):
    print('\t'*3,n+1,': ', i, sep='')

print("\nAverage K Fold Score: ", np.mean(crossvalScore))

# Standardizing our features
# scaler  = StandardScaler()
# scaler.fit(df.drop('Type', axis=1))
# scaled_feature = scaler.transform(df.drop('Type', axis=1)) 
# df_feat =  pd.DataFrame(scaled_feature, columns=df.columns[:-1])
# print(df_feat)