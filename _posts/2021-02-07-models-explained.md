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

The [Machine Learning Explained: Introduction](https://hxmza.me/research/machine-learning-explained/){:target="_blank"} post explains the two main types of machine learning. All machine learning models are categorised as either supervised or unsupervised. 

## Supervised Learning Models
Supervised learning involves learning a function that maps an input to an output based on example input-output pairs. For example, if a dataset had two variables, age (input) and height (output), a supervised learning model could be implemented to predict the height of a person based on their age. An intelligent correlation analysis can lead to a greater understanding of the data. It is useful to use correlation alongside regression as it can more easily present the relationship between two varibles. 

Supervised learning models fall in two sub-categories: regression and classification.

### Correlation
Correlation is a measure of how strongly one variable depends on another. In terms of ML, this is how features correspond with the output. 

### Regression
Regression is typically the next step up after correlation and understanding the data better. In regression models, the output is continuous and it finds the causal relationship between variables X and Y allows for accurate prediction of the Y value for each X value. 

#### Linear Regression
Linear regression finds a line that best fits the data as shown in the image below. Extensions of linear regression include multiple linear regression and polynomial regression.

![layers](https://miro.medium.com/max/2400/1*sOi6uKo3d-OxmA1caVmm0g.png)

#### Decision Tree
Decision trees are popular models used in research, planning and mahcine learning. each decision is called a **node**, and the more nodes there are, the more accurate the decision tree will be (generally). The last nodes of the decision tree, where a decision is made, are called the leaves of the tree. Decision trees are intuitive and easy to build but fall short when it comes to accuracy.

![layers](https://miro.medium.com/max/2400/1*L9AcBn8WmWN44s-NQiDbOQ.png)
In the image above, each box is a node and te final set of boxes are the leaves.

#### Random Forest
Random forests are an ensemble learning technique that build off of decision trees. Ensemble learning is the process by which multiple models, such as classifiers or regression models, are generated and combined to solve a particular problem. Random forests involve creating multiple decision trees using bootstrapped datasets (resampling) of the original data and randomly selecting a subset of variables at each step of the decision tree. The model then selects the mode of all the predicionts of each decision tree and the most frequent model is chosem. This relies on a "majority wins" model and reduces the risk of error from an individual tree.

![layers](https://miro.medium.com/max/1050/1*RuglAQsrbJWG49kaXv6tdQ.png)

In the example abode, if a single decision tree was created (third one), it would predict 0, but by relying on the mode (most frequent occurence) of all four decision trees, the predicted value would be 1. This is the power of random forests.

#### Neural Network

A neural network is essentially a network of mathematical equations. It takes one or more input variables and results in one or more output variables by going thorugh the entire network of equations. [This post](https://hxmza.me/research/deep_learning/) explains neural networks used in deep learning and how they are structured. 

<a href="https://hxmza.me/research/deep_learning/" target="_blank">This post</a>


### Classification
Classification is another type of supervised learning method in which the output is discrete (and finite). Classification is a process of categorising a given set of data into classes. The process starts with predicting the class of given data points. The classes are often referred to as target, label or categories. Below are some of the most common types of classification models.

#### Logistic Regression
Logistic regression is similar to linear regression but is used to model the probability of a finite number of outcomes, typically two. 
