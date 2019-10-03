import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
import numpy as np

df=pd.read_csv("arrhythmia.csv", ',', na_values=["?"])
a = df.fillna(df.mean())


print(df.mean())

ds=np.array(a)
ds.shape
x=ds[:,:-1]
y=ds[:,-1]
s = df.ix[1,:-1].index
labels=[]
labels=s
fvalue_selector = SelectKBest(f_classif, k=55)
X_kbest = fvalue_selector.fit_transform(x, y)
print('Original number of features:', x.shape[1])
print('Reduced number of features:', X_kbest.shape[1])
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clf.fit(xtrain,ytrain)
i=0
#for feature in zip(labels, clf.feature_importances_):
    #i+=1

for feature_list_index in fvalue_selector.get_support(indices=True):
    print(labels[feature_list_index])

X_important_train = fvalue_selector.transform(xtrain)
X_important_test = fvalue_selector.transform(xtest)