---
title: "Machine Learning Explained "
categories:
  - data science
  - machine learning
tags:
  - data
  - scientist
  - science
  - analyst
  - python
  - machine
  - learning
---

This post explores machine learning and the different methods that are used to develop predictive models. We will understand what each of these methods are, what they do, their pros and cons, and what is the best application for them. 

## There are two main types of learning:

1) Supervised Learning

2) Unsupervised Learning


## 1) Supervised Learning 

### What is Supervised Learning?

In supervised learning, a predictive model is developed based on both input data and output data. Classification refers to a supervised learning predictive modeling problem where a class label is predicted for a sample of input data. 

### Example:

Examples of such problem include classifying whether a picture is a cat or a dog, handwriting characters, etc. Supervised learning also includes regression analysis which is used to predict a continuous outcome variable (y) based on the value or predictor variables (x). The goal of a regression model is to build a mathematical equation that defines y as a function of the x variables. An example of regression modelling is predicting prices of a house given the features of house like size, price, area, etc. In a supervised learning model, the algorithm learns on a labeled dataset, providing an answer key that the algorithm can use to evaluate its accuracy on training data.

### Why Supervised Learning?

Supervised learning collects data and produces an output from previous experience. This allows for easy optimisation of performance and can result in better accuracy. Supervised learning algorithms can also help make predictions for new unseen data that we obtain later in the future.

### How does Supervised Learning work?

If you want to train a machine to help you predict how long it will take you to drive home from your workplace, you start by creating a set of **labeled data**. This data includes:

- Weather conditions
- Time of the day
- Holidays

All these details are your inputs. The output is the amount of time it took to drive back home on that specific day.

As humans, we know that if it's raining outside, then it will take you longer to drive home, but the machine needs data and statistics to analyse this.

To develop a supervised learning model to help the user to determine the commute time, we must first create a training data set. This training set will contain the total commute time and corresponding factors like weather, time, etc. Based on this training set, your machine might see there's a direct relationship between the amount of rain and time you will take to get home. The machine does this many times with the different features that we use to compute the predictions. 

The closer you're to 6 p.m. the longer time it takes for you to get home. Your machine may find some of the relationships with your labeled data.

The following diagram shows the learning phase of a machine learning model and the steps required in order to make predictions:

![label](https://i.imgur.com/dCXdMkl.png)

This is the start of the model. It begins to impact how rain impacts the way people drive. It also starts to see if more people travel during a particular time of day.

### Types of Supervised Learning techniques
![label](https://i.imgur.com/nwv4EHC.png)

- Regression: Regression technique predicts a single output value using training data. 
  
  Example: You can use regression to predict the house price from training data. The input variables will be locality, size of a house, etc.

- Classification: Classification means to group the output inside a class. If the algorithm tries to label input into two distinct classes, it is called binary classification. Selecting between more than two classes is referred to as multiclass classification.
  
  Example: Determining whether or not someone will be a defaulter of the loan.


## 2) Unsupervised Learning

### What is Unsupervised Learning?

In unsupervised learning, groups (clusters) of data are created and only input data is interpreted. Here, the model to works on its own to discover information and mainly deals with the unlabelled data. Cluster analysis or clustering is an example of unsupervised learning where data is grouped in such a way that objects in the same group are more similar to each other than to those in other groups. Other unsupervised learning algorithms include association, anomaly detection, neural networks, etc. 

### Example:

Examples of unsupervised learning include clustering of DNA patterns to analyse evolutionary biology, human detection without knowledge of who is in the photo. A real-world example goes as follows; A friend invites you to his party where you meet totally strangers. Now you will classify them using unsupervised learning (no prior knowledge) and this classification can be on the basis of gender, age group, dressing, educational qualification or whatever way you would like. This learning is different from Supervised Learning since you didn't use any past/prior knowledge about people and classified them "on-the-go".

### Why Unsupervised Learning?

Unsupervised learning finds many unknown patterns in data which can help in creating features which are useful for catergorisation. Unsupervised learning deals with unlabeled data which is very convenient since it is easier to get unlabeled data from a computer than labeled data, which requires manual intervention. 

### How does Unsupervised Learning work?

Unsupervised learning uses unlabeled data (no prior knowledge) to train a model and make predictions. Let's take the example of a child and the family cat. 

The child knows and identifies this dog. A few weeks later a family friend brings along a dog and tries to play with the child. 

The child has not seen this dog earlier. But it recognizes many features (2 ears, eyes, walking on 4 legs) are like her pet dog. She identifies a new animal like a dog. This learning is unsupervised, where the machine (or in this case the child), are not taught anything prior to being given data, but you learn from the data (in this case data about a dog). Had this been supervised learning, the family friend would have told the baby that it's a dog.

### Types of Unupervised Learning techniques
![label](https://i.imgur.com/99AtPGb.png)

- Clustering: Clustering deals with finding a structure or pattern in a collection of uncategorized data. Clustering algorithms will process your data and find natural clusters (groups) if they exist in the data. You can also modify how many clusters your algorithms should identify.

  Example: If you are a business trying to get the best return on your marketing investment, it is crucial that you target people in the right way. Here, clustering algorithms are able to group together people with similar traits and likelihood to purchase. Once you have the groups, you can run tests on each group with different marketing copy that will help you better target your messaging to them in the future.

- Association: Association rules allow you to establish associations amongst data objects inside large databases. This unsupervised technique is about discovering exciting relationships between variables in large databases. For example, people that buy a new home most likely to buy new furniture.

  Example: Groups of shopper based on their browsing and purchasing histories.

## Comparson

![labels](https://i.imgur.com/JAqrARe.png)