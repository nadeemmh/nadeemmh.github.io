---
title: "Kaggle House Prices Predictions Using Advanced Regression Techniques"
categories:
  - data science
  - machine learning
  - predictive modelling
tags:
  - data
  - scientist
  - science
  - analyst
  - python
  - machine
  - learning
  - regression
---

In this post we will attempt to predict house prices Ames, Iowa, using 79 explanatory variables describing (almost) every aspect of residential homes.  


## Overview

### 1) Loading Data
### 2) Understand Data
### 3) Feature Engineering and Data Preprocessing
### 4) Model Testing and Building
### 5) Model Optimisation
### 6) Results
### 7) Conclusions

## Importing Packages

``` r
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns
```

## 1) Loading Data

``` r
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv') # importing training set
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv') # importing testing set
```

## 2) Understand Data

In order to use and accurately predict the data, we must first understand the data and be able to extract information from it properly. Understanding the data allows us to plan ahead and clean the data properly in the following stages.

``` r
df_train.head()
```
> ![layers](https://i.imgur.com/IwB2NLi.png)

<br></p>

``` r
df_test.head()
```
> ![layers](https://i.imgur.com/Byzz0qL.png)

<br></p>

The above outputs briefly show the contents of the data files (test and train) and it we can assume that there are many missing values to be dealt with just by looking at the first 5 observations. Now that we know this, we can see how many missing values there are in total for each feature and can attempt to solve this problem.

``` r
null_values_train = df_train.isnull().sum()
print(null_values_train)
```
> ![layers](https://i.imgur.com/OmxJkiJ.png)

``` r
null_values_test = df_test.isnull().sum()
print(null_values_test)
```
> ![layers](https://i.imgur.com/ZC2B1ie.png)

It is clear that our initial assumption was correct and there are many missing values. It would be unwise to use features with too many missing values so we will drop features with more than 50% missing values, and we will attempt to fill in the rest of the values. 

## 3) Feature Engineering

The training and testing data sets that we are using in this report have many null values and the data needs to be cleaned in order to give acceptable predictions. If the feature has too many null values and/or is not relevant to predicting the model, we will drop the feature entirely. If the feature is important and has missing/null values, we will be filling those values manually. For floats/integers, we will use the mean to fill in the null values, and for objects we will be using the mode, and other values to fill in the nulls.

We will be observing the training and testing data simultaneously and will be applying changes on both data sets where needed. This is often easier to do on two separate files, but in this case, we can handle both tasks together in one file.

The test set is missing a column called "SalePrice" which is what we will use the training set to predict for the test set, otherwise, both datasets have the same features.

We will go through the features systematically and will decide whether it is worth keeping or not. In this report, we will only mention the features that are relevant or worth looking at else the report would be too long!

We can create a few simple functions that will help remove null values with ease. One function is for the training set and one is for the testing set. Though these could have been put together, we can use these separately if/when needed.

``` r
# function for filling null values in the training set features using mean or mode.
def fill_null_train(feature, method):
    if method == 'mean':
        df_train[feature] = df_train[feature].fillna(df_train[feature].mean())
        return df_test[feature].isnull().sum()
    elif method == 'mode':
        df_train[feature] = df_train[feature].fillna(df_train[feature].mode()[0])
        return df_test[feature].isnull().sum()
    else:
        return 'Method Error: Choose mean or mode in second input.'
```

``` r
# function for filling null values in the testing set features using mean or mode.
def fill_null_test(feature, method):
    if method == 'mean':
        df_test[feature] = df_test[feature].fillna(df_test[feature].mean())
        return df_test[feature].isnull().sum()
    elif method == 'mode':
        df_test[feature] = df_test[feature].fillna(df_test[feature].mode()[0])
        return df_test[feature].isnull().sum()
    else:
        return 'Method Error: Choose mean or mode in second input.'
```

``` r
#function for dropping a feature in the training set.
def drop_train(feature):
    df_train.drop([feature], axis = 1, inplace = True)
```

``` r
#function for dropping a feature in the testing set.
def drop_test(feature):
    df_test.drop([feature], axis = 1, inplace = True)
```

We can now use these functions for our feature engineering. First we fill the features which only have missing values in the test set.

``` r
# fill missing values using the mode (since it is an object which can only take a few values)
# do this only for test set since train set already has zero null values in MSZoning
fill_null_test('MSZoning', 'mode')
```

Now we can fill in the values using the mean when necessary. We only do this if the feature is a float64 value.

``` r
# "LotFrontage" is a float64 value so we can fill the null values using the mean.
fill_null_test('LotFrontage', 'mean')
fill_null_train('LotFrontage', 'mean')
```

Next we can drop all the insignificant features that either provide no usefulness, or have too many missing values to be considered in the prediction.

``` r
# we remove these features entirely due to the large amount of null values 
drop_train('Alley')
drop_test('Alley')

drop_train('GarageYrBlt')
drop_test('GarageYrBlt')

drop_train('PoolQC')
drop_test('PoolQC')

drop_train('Fence')
drop_test('Fence')

drop_train('MiscFeature')
drop_test('MiscFeature')
```

Now we can fill in the missing values for all the features that have any, using the mode.

``` r
# fill missing values using the mode for both train and test
fill_null_train('BsmtCond', 'mode')
fill_null_test('BsmtCond', 'mode')

fill_null_train('BsmtQual', 'mode')
fill_null_test('BsmtQual', 'mode')

fill_null_train('FireplaceQu', 'mode')
fill_null_test('FireplaceQu', 'mode')

fill_null_train('GarageType', 'mode')
fill_null_test('GarageType', 'mode')

fill_null_train('GarageFinish', 'mode')
fill_null_test('GarageFinish', 'mode')

fill_null_train('GarageQual', 'mode')
fill_null_test('GarageQual', 'mode')

fill_null_train('GarageCond', 'mode')
fill_null_test('GarageCond', 'mode')

fill_null_train('MasVnrType', 'mode')
fill_null_test('MasVnrType', 'mode')

fill_null_train('MasVnrArea', 'mode')
fill_null_test('MasVnrArea', 'mode')

fill_null_train('BsmtExposure', 'mode')
fill_null_test('BsmtExposure', 'mode')

fill_null_train('BsmtFinType1', 'mode')
fill_null_test('BsmtFinType1', 'mode')

fill_null_train('BsmtFinType2', 'mode')
fill_null_test('BsmtFinType2', 'mode')
```

Here, we fill missing values for features that are only in train.

``` r
# fill missing values using the mode for only train
fill_null_train('Electrical', 'mode')
```

Here, we fill missing values for features that are only in test.

``` r
# fill missing values using the mode for only test
fill_null_test('Utilities', 'mode')

fill_null_test('Exterior1st', 'mode')

fill_null_test('Exterior2nd', 'mode')

fill_null_test('BsmtFullBath', 'mode')

fill_null_test('BsmtHalfBath', 'mode')

fill_null_test('KitchenQual', 'mode')

fill_null_test('Functional', 'mode')

fill_null_test('SaleType', 'mode')

# fill missing values using the mode for only test
fill_null_test('BsmtFinSF1', 'mean')

fill_null_test('BsmtFinSF2', 'mean')

fill_null_test('BsmtUnfSF', 'mean')

fill_null_test('TotalBsmtSF', 'mean')

fill_null_test('GarageCars', 'mean')

fill_null_test('GarageArea', 'mean')
```

Now that we can removed or filled all the null values, we can check if there are any remaining null values in any of the data.

``` r
df_train.isnull().sum()
```
> ![layers](https://i.imgur.com/IEMe5NH.png)


``` r
df_test.isnull().sum()
```
> ![layers](https://i.imgur.com/rSJbvSz.png)

We can now handle the catergorical features by creating a feature set that will be used later.

``` r
# Creating feature set to be used later

columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
           'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 
           'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 
           'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive'
]
len(columns)
```
> 39

We now create a function that converts all the features into categorical features. We use the above feature set here. 

```r
# function that converts all the features into categorical features

def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final
```

We make a copy of the original dataframe before we concatenate the train and test data into a single dataset.

```r
# copy of the original dataframe
main_df = df_train.copy()

# dataframe containing both datasets
final_df = pd.concat([df_train, df_test], axis = 0)

final_df.shape
```
> (2919, 76)


## 5) Data Preprocessing
Data preprocessing (or data mining) is used to transform the raw data and make it more usable and useful. Here, data is cleaned and missing values are restored or handled.

``` r
# create all the catergorical vairables that we did in previous steps for both train and test sets.
# most of this is taken from previous code
df_all['cabin_multiple'] = df_all.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
df_all['cabin_adv'] = df_all.Cabin.apply(lambda x: str(x)[0])
df_all['numeric_ticket'] = df_all.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
df_all['ticket_letters'] = df_all.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) > 0 else 0)
df_all['name_title'] = df_all.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# create nulls for continuous data
df_all.Age = df_all.Age.fillna(df_train.Age.median()) # take median of age to fill because it is not normally distributed
df_all.Fare = df_all.Fare.fillna(df_train.Fare.median()) # take median of fare to fill because it is not normally distributed
```

``` r
# drop null values in 'embarked' rows. This only happens twice in the training and never in test.
df_all.dropna(subset=['Embarked'], inplace = True)
``` 

``` r
# log norm of fare 
df_all['norm_fare'] = np.log(df_all.Fare+1)
df_all['norm_fare'].hist()
```
> ![layers](https://i.imgur.com/qBd6nrv.png)

``` r
# converted fare to category for use
df_all.Pclass = df_all.Pclass.astype(str)
```

``` r
# created dummy variables from categories 
df_dummies = pd.get_dummies(df_all[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare', 'Embarked', 'cabin_adv', 'cabin_multiple', 'numeric_ticket', 'name_title', 'train_test']])
```

``` r
# split to train test again
X_train = df_dummies[df_all.train_test == 1].drop(['train_test'], axis = 1)
X_test = df_dummies[df_all.train_test == 0].drop(['train_test'], axis = 1)

y_train = df_all[df_all.train_test == 1].Survived
```

## 6) Data Scaling
Data scaling is used to normalize the range of the values of the data. We transfrom the data so that it is normalised (between 0 and 1).

``` r
from sklearn.preprocessing import StandardScaler
df_scaler = StandardScaler()
df_dummies_scaled = df_dummies.copy()
df_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']] = df_scaler.fit_transform(df_dummies_scaled[['Age', 'SibSp', 'Parch', 'norm_fare']])
df_dummies_scaled
```
> ![layers](https://i.imgur.com/yZrEuw8.png)

``` r
X_train_scaled = df_dummies_scaled[df_dummies_scaled.train_test == 1].drop(['train_test'], axis = 1)
X_test_scaled = df_dummies_scaled[df_dummies_scaled.train_test == 0].drop(['train_test'], axis = 1)

y_train = df_all[df_all.train_test == 1].Survived
``` 

## 7) Model Testing and Building
Here we build the model and can experiment with different models with default parameters.

``` r
# import all necessary packages from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
```
We will try a few of the model using 5 fold cross validation to get a baseline model to see which model performs the best and delivers the most accurate results.

``` r
# We use Naive Bayes as a baseline for the classification tasks
gnb = GaussianNB()
cv = cross_val_score(gnb, X_train_scaled, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/NRjJMoJ.png)

``` r
# Logistic Regression test
lr = LogisticRegression(max_iter = 1000)
cv = cross_val_score(lr, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/654ZFWp.png)

``` r
# Decision Tree test
dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, X_train_scaled, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/jrcv9Cj.png)

``` r
# KNN test
knn = KNeighborsClassifier()
cv = cross_val_score(knn, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/xnMSN20.png)

``` r
# Random Forest test
rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/Nqju8IZ.png)

``` r
# Support Vector Machine
svc = SVC(probability = True)
cv = cross_val_score(svc, X_train_scaled, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/LRyzzXU.png)

``` r
# Extreme Gradient Boosting
xgb = XGBClassifier(random_state = 1)
cv = cross_val_score(xgb, X_train_scaled, y_train, cv = 5)
print(cv)
print(cv.mean())
``` 
> ![layers](https://i.imgur.com/6oyha7C.png)


> ![layers](https://i.imgur.com/CU5UHkl.png)

From the table above, it is clear that the Support Vector Machine model performed the best out of the group in the 5-fold cross validation test.

## 8) Model Optimisation
The "Voting Classifier" takes all of the inputs and averages the results. For a "hard" voting classifier each classifier gets 1 vote "yes" or "no" and the resutl is just a popular vote.

A "soft" classifier averages the confidence of each of the models. If the average confidence is > 50% that it is a 1 it will be counted as such.

``` r 
from sklearn.ensemble import VotingClassifier
```

``` r
voting_cl = VotingClassifier(estimators = [('lr', lr), ('knn', knn), ('rf', rf), ('dt', dt), ('gnb', gnb), ('svc', svc)], voting = 'soft')

cv = cross_val_score(voting_cl, X_train_scaled, y_train, cv = 5)
print(cv)
print(cv.mean())
```
> ![layers](https://i.imgur.com/NMsJ9la.png)

``` r
voting_cl.fit(X_train_scaled, y_train)

y_pred = voting_cl.predict(X_test_scaled).astype(int)

y_pred = pd.Series(voting_cl.predict(X_test_scaled).astype(int))
```

## 9) Results
These are the final results for the predictions.

``` r
print(y_pred)
```
> ![layers](https://i.imgur.com/SChKjYl.png)

``` r
print('The survival rate of the predicted Passengers is: ', "%.2f" % ((1 - sum(y_pred) / len(y_pred)) * 100), '%')
```
> ![layers](https://i.imgur.com/tyxM9s4.png)

``` r
submission = {'PassengerId': df_test.PassengerId, 'Survived': y_pred}

submission = pd.DataFrame(data = submission)

submission.to_csv('submission.csv', index=False)

print('Submitted!')
```
> ![layers](https://i.imgur.com/fAJyYe9.png)

## 10) Conclusion
On the first attempt, using the Logistic Regression predictor, we were able to predict data points at an accuracy of 0.75837 (75.84%). This was a good first prediction but it was clear that this could be improved. Tweaking parameters was not helpful and the accuracy was not increased at all. The only other option was to try more classifiers that might fit the model better. The Xtreme Gradient Boosting classifier (XGB) was then used and increased this accuracy to 0.77272 (77.27%) which was a slight improvement. From this point onwards it was almost impossible to increase the accuracy using the model we had built. Thus, the final accuracy of the model was 0.77272.

In future versions, we can look at other machine learning techniques such as deep learning neural networks to try to increase this accuracy and make a model more suitable for our data. We can also look into the best way to optimise parameters and observe how the model behaves as each parameter is changed.

