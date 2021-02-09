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

## 3) Feature Engineering and Data Preprocessing

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

Now we can use the function created earlier to convert all features in this new dataframe into catergorical features.

```r
final_df = category_onehot_multcols(columns)
```
> ![layers](https://i.imgur.com/yIQLbFn.png)

```r
final_df.shape
```
> (2919, 237)

We can now see the new dataframe

```r
final_df = final_df.loc[:,~final_df.columns.duplicated()]

final_df
```
> ![layers](https://i.imgur.com/AY9cWPN.png)

We can now split this dataframe into the train and test datasets.

```r
df_Train = final_df.iloc[:1460,:]
df_Test = final_df.iloc[1460:,:]
```

We now drop the "Sale Price" feature from the training set because we do not have any of these values (all the values are null).

```r
X_train = df_Train.drop(['SalePrice'], axis = 1).values
y_train = df_Train['SalePrice'].values
```

By doing this we have now created the train/test split which we can now use to build a model and generate predictions.

## 4) Model Testing and Building
Here we build the model and can experiment with different models with default parameters. We 

``` r
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
import xgboost as xgb
from sklearn import tree
from sklearn.linear_model import Lasso
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
```
We will test a few of the models using 5 fold cross validation to get a baseline model to see which one performs the best and delivers the most accurate results.

``` r
# Decision Tree Regressor
dt = tree.DecisionTreeRegressor()
cv = cross_val_score(dt, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
mean_dt = round(cv.mean(), 4)
``` 
> ![layers](https://i.imgur.com/jLLdV3n.png)

``` r
# Bayesian Ridge Regression
brr = linear_model.BayesianRidge()
cv = cross_val_score(brr, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
mean_brr = round(cv.mean(), 4)
``` 
> ![layers](https://i.imgur.com/DPuIagr.png)

``` r
# Ridge CV
rdg = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))
cv = cross_val_score(rdg, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
mean_rdg = round(cv.mean(), 4)
``` 
> ![layers](https://i.imgur.com/QnoitCh.png)

``` r
# Extreme Gradient Boosting
xgb = xgb.XGBRegressor(n_estimators=340, max_depth=2, learning_rate=0.2)
cv = cross_val_score(xgb, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
mean_xgb = round(cv.mean(), 4)
``` 
> ![layers](https://i.imgur.com/7dkJ4Pz.png)

``` r
# Lasso Regression
lso = Lasso(alpha=0.1,random_state=0)
cv = cross_val_score(lso, X_train, y_train, cv = 5)
print(cv)
print(cv.mean())
mean_lso = round(cv.mean(), 4)
``` 
> ![layers](https://i.imgur.com/d3sIfQ8.png)


> ![layers](https://i.imgur.com/UNuegrg.png)

The table above shows the average results (accuracy) for each model which gives us an idea of which model would be optimal for our case.

## 5) Model Optimisation
From the baseline models, it is clear that the Extreme Gradient Bossting model performed the best out of the group in the 5-fold cross validation test. We can now optimise this model to try and predict the data.

### Attempt 1:

``` r 
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Hyperparameters
params = {
    'objective': 'reg:squarederror', # MSE as loss function
    'eval_metric': 'rmse', # RMSE as metric
    'eta': 0.3, # Learning rate
}

# model
model = XGBRegressor(**params)

# Fit the model to the data
model.fit(X_train, y_train)
```
> ![layers](https://i.imgur.com/6M3Wc5k.png)

```r
# Predictions
y_test = model.predict(df_Test.drop(['SalePrice'], axis = 1).values)
```

### Attempt 2


```r
submission = pd.DataFrame({'Id':df_test.Id,'SalePrice':y_test})

submission.to_csv('submission.csv', index=False)

print('Submitted!')
```
> Submitted!

## 6) Results
These are the final results for the predictions.

### Attempt 1:

``` r
print(y_test)
```
> ![layers](https://i.imgur.com/MliD8Ke.png)

### Attempt 2:


## 7) Conclusion
Coming Soon
