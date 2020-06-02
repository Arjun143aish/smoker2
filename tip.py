import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Tip = sns.load_dataset('tips')

Tip.isnull().sum()

#Visualization

plt.figure(figsize = [12,5])
sns.barplot(x = 'total_bill', y= 'sex',hue= 'smoker', data = Tip)

sns.boxenplot(x = 'total_bill', y= 'sex',hue= 'smoker', data = Tip)
sns.boxplot(x = 'sex', y= 'total_bill',hue= 'smoker', data = Tip)
sns.scatterplot(x = 'total_bill',y = 'tip',data= Tip)
sns.distplot(Tip['total_bill'])

dummyDf = pd.get_dummies(Tip[['sex','day','time']],drop_first = True)
FullRaw = Tip.drop(['sex','day','time'], axis =1)
FullRaw = pd.concat([FullRaw,dummyDf], axis =1)
FullRaw['smoker'] = np.where(FullRaw['smoker'] == 'No',1,0)

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['smoker'], axis =1)
Train_Y = Train['smoker'].copy()
Test_X = Test.drop(['smoker'], axis =1)
Test_Y = Test['smoker'].copy()

from statsmodels.api import Logit

M1_Model = Logit(Train_Y,Train_X).fit()
M1_Model.summary()

Test_pred = M1_Model.predict(Test_X)

from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

Test['Test_prob'] = Test_pred
Test['Test_Class'] = np.where(Test['Test_prob'] > 0.5,1,0)

Con_Mat = confusion_matrix(Test['Test_Class'],Test_Y)
sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

from sklearn.metrics import roc_auc_score,roc_curve

ROC = roc_auc_score(Test['Test_Class'],Test_Y)
AUC = roc_curve(Test['Test_Class'],Test_Y)

from sklearn.ensemble import RandomForestClassifier


RF_Model = RandomForestClassifier(random_state=123).fit(Train_X,Train_Y)
RF_Pred = RF_Model.predict(Test_X)

RF_Con = confusion_matrix(RF_Pred,Test_Y)
sum(np.diag(RF_Con))/Test_Y.shape[0]*100

from sklearn.model_selection import GridSearchCV

n_tree = [50,75,100]
n_split = [100,200]
my_param_grid = {'n_estimators': n_tree,'min_samples_split': n_split}

Grid = GridSearchCV(RandomForestClassifier(random_state =123),param_grid = my_param_grid,
                    scoring ='accuracy',cv = 5).fit(Train_X,Train_Y)


import pickle

pickle.dump(RF_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
    
