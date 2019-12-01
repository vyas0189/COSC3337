#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
#from keras.utils import np_utils
#from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
#from sklearn.datasets import make_classification
from ann_visualizer.visualize import ann_viz
#import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Read in the data
dataset = pd.read_csv('page-blocks.data',sep="\s+", header= None)


# Seperate indep and dep vars
X = dataset.iloc[:, :10]
y = dataset.iloc[:, 10]


encoder = LabelEncoder()
y = encoder.fit_transform(y)

from keras.utils import to_categorical
y = to_categorical(y, num_classes=None)

seed = 7
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle = True, random_state=seed)

from sklearn.preprocessing import StandardScaler

cvscoresreluFn = []
for train, test in kfold.split(X, y):
	modelreluFn = Sequential()
	modelreluFn.add(Dense(6, input_dim=10, activation='relu'))
#	modelreluFn.add(Dense(6, activation='relu'))
	modelreluFn.add(Dense(5, activation='softmax'))
	modelreluFn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	X_train, X_test = X.iloc[train], X.iloc[test]
	y_train, y_test = y[train], y[test]
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.fit_transform(X_test)
	modelreluFn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
	scoresreluFn = modelreluFn.evaluate(X_test, y_test, verbose=0)
	print("relu %s: %.2f%%" % (modelreluFn.metrics_names[1], scoresreluFn[1]*100))
	cvscoresreluFn.append(scoresreluFn[1] * 100)
	


#def reluFn():
#	# create model
#	model = Sequential()
#	model.add(Dense(6, input_dim=10, activation='relu'))
#	model.add(Dense(6, activation='relu'))
#	model.add(Dense(5, activation='softmax'))
#	   # Compile model
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
#
#reluestimator = KerasClassifier(build_fn=reluFn, epochs=100, batch_size=10)
#relukfold = KFold(n_splits=10, shuffle=True)
#reluresults = cross_val_score(reluestimator, X_train, y_train, cv=relukfold)


#def eluFn():
#	# create model
#	model = Sequential()
#	model.add(Dense(6, input_dim=10, activation='elu'))
#	model.add(Dense(6, activation='elu'))
#	model.add(Dense(5, activation='softmax'))
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model
#
#eluFnestimator = KerasClassifier(build_fn=eluFn, epochs=100, batch_size=10)
#eluFnkfold = KFold(n_splits=10, shuffle=True)
#eluFnresults = cross_val_score(eluFnestimator, X_train, y_train, cv=eluFnkfold)
#
##y_pred = eluFn().predict(X_test)
#predictions = reluFn().predict_classes(X_test)
#print("reluFn: %.2f%% (%.2f%%)" % (reluresults.mean()*100, reluresults.std()*100))
#print("eluFn: %.2f%% (%.2f%%)" % (eluFnresults.mean()*100, eluFnresults.std()*100))

cvscoreseluFn = []
for train, test in kfold.split(X, y):
	modeleluFn = Sequential()
	modeleluFn.add(Dense(6, input_dim=10, activation='elu'))
#	modeleluFn.add(Dense(6, activation='elu'))
	modeleluFn.add(Dense(5, activation='softmax'))
	modeleluFn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	X_train, X_test = X.iloc[train], X.iloc[test]
	y_train, y_test = y[train], y[test]
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.fit_transform(X_test)
	
	modeleluFn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
	scoreseluFn = modeleluFn.evaluate(X_test, y_test, verbose=0)
	print("elu %s: %.2f%%" % (modeleluFn.metrics_names[1], scoreseluFn[1]*100))
	cvscoreseluFn.append(scoreseluFn[1] * 100)
	



# ------------------------------------------------------------------------------------------------------


cvscorestanhFn = []
for train, test in kfold.split(X, y):
	modeltanhFn = Sequential()
	modeltanhFn.add(Dense(6, input_dim=10, activation='tanh'))
#	modeltanhFn.add(Dense(6, activation='elu'))
	modeltanhFn.add(Dense(5, activation='softmax'))
	modeltanhFn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	X_train, X_test = X.iloc[train], X.iloc[test]
	y_train, y_test = y[train], y[test]
	
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.fit_transform(X_test)
	
	modeltanhFn.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
	scorestanhFn = modeltanhFn.evaluate(X_test, y_test, verbose=0)
	print("tanh %s: %.2f%%" % (modeleluFn.metrics_names[1], scorestanhFn[1]*100))
	cvscorestanhFn.append(scorestanhFn[1] * 100)




print("relu Acc/Std: %.2f%% (+/- %.2f%%)" % (np.mean(cvscoresreluFn), np.std(cvscoresreluFn)))
#ann_viz(modelreluFn, title="modelreluFn",  filename="modelreluFn")
y_predreluFn = modelreluFn.predict(X_test)
from sklearn.metrics import confusion_matrix
y_predreluFn = (y_predreluFn > 0.5)
cmreluFn = confusion_matrix(y_test.argmax(axis=1), y_predreluFn.argmax(axis=1))

print("elu Acc/Std: %.2f%% (+/- %.2f%%)" % (np.mean(cvscoreseluFn), np.std(cvscoreseluFn)))
#ann_viz(modeleluFn, title="modeleluFn", filename="modeleluFn")
y_predeluFn = modeleluFn.predict(X_test)
from sklearn.metrics import confusion_matrix
y_predeluFn = (y_predeluFn > 0.5)
cmeluFn = confusion_matrix(y_test.argmax(axis=1), y_predeluFn.argmax(axis=1))

	
print("tanh Acc/Std: %.2f%% (+/- %.2f%%)" % (np.mean(cvscorestanhFn), np.std(cvscorestanhFn)))
#ann_viz(modeltanhFn, title="modeltanhFn", filename="modeltanhFn")
y_predtanhFn = modeltanhFn.predict(X_test)
from sklearn.metrics import confusion_matrix
y_predtanhFn = (y_predtanhFn > 0.5)
cmtanhFn = confusion_matrix(y_test.argmax(axis=1), y_predtanhFn.argmax(axis=1))
