# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:36:12 2018

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
import numpy as np
from sklearn import svm
from sklearn.datasets import load_svmlight_file as load_svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.metrics import confusion_matrix
from anova_feature_selection import clf,xtrain,ytrain,xtest,ytest,X_important_train,X_important_test



sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)

clf_imp=Sequential()
clf_imp.add(Dense(output_dim=54,init='uniform',activation='relu',input_dim=330))
clf_imp.add(Dense(output_dim=12,init='uniform',activation='relu'))
clf_imp.add(Dense(output_dim=6,init='uniform',activation='relu'))
clf_imp.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
clf_imp.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
clf_imp.fit(xtrain,ytrain,batch_size=10,nb_epoch=100)

y_pred=clf.predict(xtest)
y_pred=y_pred>0.5
cm=confusion_matrix(y_pred,ytest)
print(cm)
###kfold
##from sklearn.model_selection import cross_val_score,KFold
##n_folds = []
##n_folds.append(('K2', 2))
##n_folds.append(('K4', 4))
##n_folds.append(('K5', 5))
##n_folds.append(('K10', 10))
##
##seed = 7
##
##for name, n_split in n_folds:
##        results = []
##        names = []
##        print(name)  
##        kfold = KFold(
##        n_splits=n_split, random_state=seed)
##        cv_results = cross_val_score(clf,xtrain,ytrain, cv=kfold, scoring='accuracy')
##        results.append(cv_results.mean())
##        #names.append(name)    
##        print(results)
##        
##print("training accuracy: {}".format(100*clf.score(xtrain,ytrain)))
##print("testing accuracy: {}".format(100*clf.score(xtest,ytest)))
##
y_true=ytest
##y_pred=clf.predict(xtest)
##
##from sklearn.metrics import precision_recall_fscore_support, accuracy_score
## 
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
##
##print("For full feature dataset...")
##print()
##print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 score: ", f1_score)
#print()

