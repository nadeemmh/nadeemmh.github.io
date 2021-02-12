---
title: "Machine Learning Explained: Model Types Explained"
categories:
  - research
tags:
  - machine learning
  - theory
  - educational
---

This post explores different types of machine learning models used for different types of cases and the majority of common machine learning models used in practice. 

The [Machine Learning Explained: Introduction](https://hxmza.me/research/machine-learning-explained/) post explains the two main types of machine learning. All machine learning models are categorised as either supervised or unsupervised. 

# Supervised Learning Models
Supervised learning involves learning a function that maps an input to an output based on example input-output pairs. For example, if a dataset had two variables, age (input) and height (output), a supervised learning model could be implemented to predict the height of a person based on their age. An intelligent correlation analysis can lead to a greater understanding of the data. It is useful to use correlation alongside regression as it can more easily present the relationship between two varibles. 

Supervised learning models fall in two sub-categories: regression and classification.

## Correlation
Correlation is a measure of how strongly one variable depends on another. In terms of ML, this is how features correspond with the output. 

## Regression
Regression is typically the next step up after correlation and understanding the data better. In regression models, the output is continuous and it finds the causal relationship between variables X and Y allows for accurate prediction of the Y value for each X value. 

### Linear Regression
Linear regression finds a line that best fits the data as shown in the image below. Extensions of linear regression include multiple linear regression and polynomial regression.

<img src="https://miro.medium.com/max/2400/1*sOi6uKo3d-OxmA1caVmm0g.png" width="600">

#### Decision Tree
Decision trees are popular models used in research, planning and mahcine learning. each decision is called a **node**, and the more nodes there are, the more accurate the decision tree will be (generally). The last nodes of the decision tree, where a decision is made, are called the leaves of the tree. Decision trees are intuitive and easy to build but fall short when it comes to accuracy.

<img src="https://miro.medium.com/max/2400/1*L9AcBn8WmWN44s-NQiDbOQ.png" width="600">
In the image above, each box is a node and te final set of boxes are the leaves.

### Random Forest
Random forests are an ensemble learning technique that build off of decision trees. Ensemble learning is the process by which multiple models, such as classifiers or regression models, are generated and combined to solve a particular problem. Random forests involve creating multiple decision trees using bootstrapped datasets (resampling) of the original data and randomly selecting a subset of variables at each step of the decision tree. The model then selects the mode of all the predicionts of each decision tree and the most frequent model is chosem. This relies on a "majority wins" model and reduces the risk of error from an individual tree.

![layers](https://miro.medium.com/max/1050/1*RuglAQsrbJWG49kaXv6tdQ.png)

In the example abode, if a single decision tree was created (third one), it would predict 0, but by relying on the mode (most frequent occurence) of all four decision trees, the predicted value would be 1. This is the power of random forests.

### Neural Network

A neural network is essentially a network of mathematical equations. It takes one or more input variables and results in one or more output variables by going thorugh the entire network of equations. [This post](https://hxmza.me/research/deep_learning/) explains neural networks used in deep learning and how they are structured. 

## Classification
Classification is another type of supervised learning method in which the output is discrete (and finite). Classification is a process of categorising a given set of data into classes. The process starts with predicting the class of given data points. The classes are often referred to as target, label or categories. Below are some of the most common types of classification models.

### Logistic Regression
Logistic regression is similar to linear regression but is used to model the probability of a finite number of outcomes, typically two. There are many reasons why logistic regression is used over linear regression when modelling probabilities of outcomes, which include non-negative values, better results (unbiased), and lower variances. To summarise, a logistic equation is created in such  a way that the outpt values can only be between 0 and 1.

<img src="https://miro.medium.com/max/2400/1*USrdZ1puaFIIymBRcO51mg.png" width="600">

The above image shows typical logistic regression, which is clearly between 0 and 1.

### Support Vector Machine
A support vector machine (SVM) is a supervised classification technique that can get pretty complex. If there are two classes of data, a SVM will find a **hyperplane** or a boundary between the two classes of data that maximised the margin between the two classes (shown in the image below). There are many planes that can separate the two classes,  but only one plane can maximise the margin/distance between the classes.

![layers](https://i.imgur.com/wmvRVjN.png)

[This article](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72) written by Savan Patel goes into detail on the theory behind SVMs and how complicated they can be (and he definitely explains it better!) and it's a great read.

### Naive Bayes
Naive Bayes is another really popular classifier used in machine learning. The idea behind it is devien by Bayes Theorem, which any mathematician or statistician must be familiar with. The theorem is:

![layers](https://i.imgur.com/r3HhNZF.png)

This essentially translates to "what is the probability of event y occuring given event X?", where y is the output variable. For Naive Bayes, an assumption that variables are independent given the class is made, so it becomes (denominator removed):

![layers](https://i.imgur.com/eKqUaDN.png)

So, P(y|X) is proportional to the right-hand side:

!layers[https://i.imgur.com/H0Ef0sP.png]

Therefore, the goal is to find the class y with the maximum proportional probability.

# Unsupervised Learning
Unlike supervised learning, unsupervised learning is used to draw inferences and find patterns from input data wihtout references to labelled outcomes. Two main methods used in unsupervised learning include clustering and dimensionality reduction. 

## Clustering
Clustering involves the grouping or **clustering** of data points. The aim is to segregate groups with similar traits for tasks such as customer segmentation, fraud detection, etc.. Common clustering techniques include the following:

### K-Means 
K-means finds groups in data, with the number of groups represented by the variable "K". The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clusetered based on feature similarity. The results of K-means clustering algorithm are: 
  1. Centroids of the K-clusters, which can be used to label new data
  2. Labels for training data (each dta point is assigned to a single cluster).
Clustering allows for analysis of groups that have formed organically, rather than defining groups before looking at the data. Each centroid of a cluster is a collection of feature values which define the resulting groups. Examining the centroid feature weights can be used to qualitatively interpret what kind of group each cluster represents.

<img src="https://miro.medium.com/max/1050/1*tWaaZX75oumVwBMcKN-eHA.png" width="600">

The image above shows how K-means finds clusters in data.

### Heirarchical Clustering
Heirarchical clusering is an algorithm similar to the K-means, but outputs structure that is more informative than the unstructured set of flat clusters returned in the form of a heirarchy. Thus, it is easier to decide on the number of cluseters by looking at the dendrogram. The image shown below represents a dendrogram that shows the clustering of letters A-F.

![layers](https://46gyn61z4i0t1u1pnq2bbk2e-wpengine.netdna-ssl.com/wp-content/uploads/2018/03/What-is-a-Dendrogram.png)

### Density-Based Clustering
Density-based clustering is a clustering method that identifies distinctive groups/clusters in the data by detecting areas where points are concentrated (high density) and where they are separated by areas that are empty (low density).

![layers](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/GUID-A06A412D-2F4F-4D35-8FFF-1F4B3B3A8F16-web.png)
