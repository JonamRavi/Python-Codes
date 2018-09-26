# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 07:49:50 2018

@author: mravi092816
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import gc
import os

###   Set the working directory

os.getcwd()
os.chdir("D:\Machine Learning\Hackathon\Hacker Earth")

### Importing Data

owner_use = pd.read_csv("Dataset/Building_Ownership_Use.csv")
structure = pd.read_csv("Dataset/Building_Structure.csv")
train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

### Defining some useful functions

def basic_info(df):
    print("No of rows and columns :", df.shape)
    print("Null values status :",df.isnull().values.any())
    print("Column Names :",df.columns)
    return df.head()

def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)
    return pd.concat([total,percent],axis=1,keys=["Total","Percent"])
    
def categorical_features(df):
    categorical_features = df.columns[df.dtypes == 'object']
    return list(categorical_features)
    
def OHE(df,categorical_features):
    df = pd.get_dummies(df, columns = categorical_features)
    return df
    

###  Handling Building_Ownership_Use

basic_info(owner_use)

check_missing_data(owner_use)

owner_use.count_families.fillna(0.0, inplace=True)

owner_use.has_secondary_use.fillna(0.0, inplace=True)

categorical_features(owner_use)

owner_use = pd.get_dummies(owner_use, columns = ["legal_ownership_status"])

###  Handling the Building Structure

basic_info(structure)

check_missing_data(structure)

categorical_features(structure)

structure["position"].describe()

structure["position"].fillna('Not attached',inplace=True)

structure["plan_configuration"].describe()

structure["plan_configuration"].fillna('Rectangular',inplace=True)

building_id = structure["building_id"]

structure.drop("building_id", axis=1,inplace=True)

structure = OHE(structure, categorical_features(structure))

structure["building_id"] = building_id

###  Handling Train and Test dataset

total = pd.concat([train,test])

basic_info(total)

check_missing_data(total)

total.has_repair_started.describe()

total.has_repair_started.value_counts()

total.has_repair_started.fillna(0,inplace=True)

categorical_features(total)

total.area_assesed.value_counts()

total = pd.get_dummies(total, columns = ["area_assesed"])

total.damage_grade.fillna(999,inplace=True) 

final_data = total.merge(structure,on="building_id",how="left").merge(owner_use,on="building_id",how="left")

train_data = final_data[final_data["damage_grade"] != 999]

test_data = final_data[final_data["damage_grade"] == 999]

test_data_1 = final_data[final_data["damage_grade"] == 999]

building_id_1 = test_data_1["building_id"]

building_list = building_id_1.tolist()

y = train_data["damage_grade"]

factor = pd.factorize(y)
y = factor[0]
definitions = factor[1]
print(y)
print(definitions)

train_data.drop("damage_grade", axis=1,inplace = True)

test_data.drop("damage_grade", axis=1,inplace = True)

test_data.drop("building_id",axis=1,inplace=True)

###  Splitting the Train data 

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_data,y,test_size=0.2)

X_train.drop("building_id",axis=1,inplace=True)
X_test.drop("building_id",axis=1,inplace=True)

###  Building a model 

# Random Forest

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier_rf.fit(X_train, y_train)

y_pred_rf = classifier_rf.predict(X_test)

f1_scre_rf = f1_score(y_test,y_pred_rf,average ='weighted')

y_pred_test = classifier_rf.predict(test_data)

##  Unfactorize the test results

reversefactor = dict(zip(range(5),definitions))
y_test = np.vectorize(reversefactor.get)(y_test)
# y_pred = np.vectorize(reversefactor.get)(y_pred)
y_pred_test = np.vectorize(reversefactor.get)(y_pred_test)

## Exporting the data to csv

y_pred_final = pd.Series(data=y_pred_test, name ="damage_grade")

y_list = y_pred_final.tolist()

# y_finaal = pd.concat([building_id_1,y_pred_final], axis =1)

y_final = pd.DataFrame({"building_id" : building_list,
              "damage_grade" : y_list
              })


y_final.to_csv("damage prediction.csv",index = False)

### Parameter Tuning - GridSearch
from sklearn.grid_search import GridSearchCV
"""
params = [{ 'n_estimators': [10, 50, 100, 250, 500, 1000],
            'max_features':['auto','sqrt','log2',None],           
            'max_depth':[1,2,3,4,5],
            'bootstrap':[True,False],
            'oob_score':[True,False],
            'min_samples_leaf': [5, 10, 15, 20, 25],
            'min_samples_split': [20, 40, 60, 80, 100],
           }]
"""
params = [{ 'n_estimators': [70, 100, 130, 160,200]}]
GridSearch = GridSearchCV( estimator = classifier_rf, param_grid = params, scoring = 'accuracy', cv=10,n_jobs=-1)           
Grid_result = GridSearch.fit(X_train,y_train)
Grid_result.best_params_
