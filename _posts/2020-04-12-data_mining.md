---
title: "University: CS3440 - Data Mining"
categories:
  - University
tags:
  - data mining
  - computer science
  - weka
  - data analysis
---

### Abstract 
This report investigates biological properties of protein sequences using data created by
parsing existing proteomic sequences, retrieved from the online databases IEDB. The training
data set has 68 unique attributes with 66 of them being numeric. The goal of this investigation
is to predit whether or not a given observation is a positive epitope.

## 1 Introduction
This report gives an outline solution of the 2020 Data Mining (CS3440) coursework. It is not
intended to be a complete report, but rather, shows the development process of a data mining
experiment.

The goal is to develop models to classify good and bad predictions. We will highlight some of
the results and key issues that occur when following a systematic data mining process.

## 2 Exploratory Data Analysis
Upon loading the training data (CWData_train.arff ), it is clear that from the 68 attributes, 1 is
a string, 1 is nominal, and 66 are numeric. From an initial observation, it is clear that there is
missing data which was confirmed by opening the file in the Visual Studio Code text editor and
searching for the ‘?’ character. Though degree of missing data differs, the majority of attributes
have 6 missing data entries which translates to a percentage close to zero (negligible). The notable
attributes with missing entries are 13, and 56.

Figure 1 shows that 75% of the data is missing from this attribute. Here, we have two options;
we can either remove this attribute entirely, or we can predict the missing values. Since 75% of
the data is missing (approximately 22500 entries), there is no guarantee that the predictions will
be accurate enough to be usable. Therefore, the best option is to remove this attribute entirely.
A similar thought process is applied to attribute 56 shown in Figure 2, which has 90% of the data
missing (approximately 27000 entries). By removing these attributes, we hope to improve the
accuracy of our results. In addition to this, since attribute 1 is an ID class, it provides no relevant
information and causes issues when using some models and filters. Thus, this attribute can also
be removed from the data.

![layers](https://i.imgur.com/OSr9CNT.png)

Figure 3 shows all the visualizations for the data set. It is important to consider the size of the
data set due to the fact that 68 attributes may impact the level of accuracy in our findings. Thus,
we much consider ways of reducing the attribute size to maximize the accuracy of our results.

![layers](https://i.imgur.com/0lK5Asx.png)

The two models that we will focus on in this investigation to model our data are Naive
Bayes model and the K-Nearest-Neighbours model (KNN). The Naive Bayes model has been chosen due to its simplicity and how easily it can be implemented. In addition to this, it
works well with data that has missing values and is also very time efficient. The KNN model has
been chosen for similar reasons but has certain benefits for our particular data set. These benefits
include adding new data without affecting accuracy, and having no training required which speeds
up the process significantly.

## 3 Data Preprocessing
In this section, we will run initial benchmarks for the Naive Bayes and the KNN models. This is
crucial in this investigation as it allows us the analyse how well the filters increase/decrease the
accuracy and becomes the criterion for the rest of the results.

### 3.1 Benchmark Models
We first apply the ‘filtered classifier’ to the whole data set under the ‘Preprocessing’ tab. We then
apply the ‘Remove’ filter to the relevant attributes which we have discussed above (attributes 1,
13, 56), and can then run each model and classifier using 10-fold cross-validation. This increases
the reliability of the error estimate and will help with the accuracy of our overall results. In
addition to this, for the KNN model, we must set ‘cross validate’ to true and the KNN value to
10. This gives us the best value for K which in our case is K = 1. The benchmark results are
shown in Table 1.
  
|           Model            | Benchmark Results |
|----------------------------|-------------------|
|         Naive Bayes        |       63.54%      |
| K-Nearest-Neighbour, K = 1 |       77.19%      |

Table 1: Benchmarks of both models with the Remove filter applied. The results are accuracies of each model given as a percentage.

### 3.2 Model Experimentation
We now build on the benchmark models and apply a variety of filters relevant to our data set
to observe the changes in accuracy. Our aim here is to try and maximize the accuracy and
analyse how different filters effect this. The filters we will apply are ‘Discretize’, ‘Normalize’, and
‘Standardize’, to each model. The results of each of these filters is shown in Table 2.

|           Model            | Discretize | Normalize | Standardize |
|----------------------------|------------|-----------|-------------|
|         Naive Bayes        |   64.26%   |  63.54%   |    63.54%   |  
| K-Nearest-Neighbour, K = 1 |   75.54%   |  77.19%   |    77.19%   |

Table 2: Accuracy of each model with the abovementioned filters applied.

The models classified above have used 65 attributes in total, though all of them may not be
useful and/or relevant since some attributes provide no clarity to our models and would only
increase computational complexity. Thus, Physical Component Analysis (PCA) was carried out
on both models with the results given in Table 3.

|           Model            | PCA Benchmark | Discretize | Normalize | Standardize |
|----------------------------|---------------|------------|-----------|-------------|
|         Naive Bayes        |     68.92%    |  69.3267%  |  68.92%   |    68.92%   |  
| K-Nearest-Neighbour, K = 1 |    76.6967%   |  73.3667%  |  76.6967% |   76.6967%  |

Table 3: Accuracy of each model and filter with PCA applied.

Using PCA reduced the number of attributes from 68 to 12 attributes for the Naive Bayes
model, and 13 attributes for the KNN model. From initial observations, it is clear that PCA has
been positively impactful on the accuracy for the Naive Bayes model. Meanwhile, for the KNN
model, PCA has reduced the accuracy from 77.19% to 76.6967%. However, this is expected due
to the fact that a lower number of attributes lead to a lower degree of neighbours for the KNN
model.

Upon further evaluation, we can see that with or without PCA applied, the discretize filter
returns a much lower accuracy for the KNN model and as a result, is always lower than the other
two filters and the benchmark. Furthermore, it appears that thus far, the accuracies for the KNN
benchmark, normalize and standardize filters are always identical.
