---
title: "Machine Learning Explained: Model Optimisation"
categories:
  - research
tags:
  - machine learning
  - theory
  - educational
---

This post explains how machine learning models are optimised and different ways to do it.

The objective of machine learning is to create a model that performs well and gives accurate predictions for a given data set. In order to make accurate predictions, we must optimise the model to suit our needs and train the data correctly. This process is called model optimisation. 

Machine learning optimisation is the process of adjusting the hyperparameters in order to minimise the cost function by using the optimisation techniques. It is important to minimise the cost function because it describes the discrepancy between the true value of the estimated parameter and what the model has predicted (minimising the error).

## K-Fold Cross-Validation
K-fold cross-validation is a technique for assessing how a model generalises to an independent data set.  It is primarily used in machine learning to estimate the proficiency of a model on unseen data. It uses a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

The procedure has a single parameter “k” that refers to the number of groups that a data sample will be split up into. When a specific value for k is chosen, the procedure becomes k-fold cross-validation, for example, k=10 become 10-fold cross-validation.

The general process goes as follows: 
  1.	Randomise the dataset.  
  2.	Split the data set into k groups.  
  3.	For each unique group:  
      1.	Take the group as a hold out or test data set.\
      2.	Take the remaining groups as a training data set.\
      3.	Fit a model on the training set and evaluate it on the test set.\
      4.	Retain the evaluation score and discard the model.\
  4.	Evaluate the skill of each model using the sample of model evaluation scores.
  
After this is done, the best performing model is chosen for model optimisation for the final predictions.

## Parameters and Hyperparameters of the Model
Parameters and hyperparameters of a model are different from each other. 

Hyperparameter: Hyperparameters must be set **before** starting to train the model. They include number of clusters, learning rate, etc. They describe the structure of the model.

Parameters: Parameters of the model are obtained **during** the training of the model. There is not way to acquire them in advance as they are internal to the model and change based on the inputs. Examples are weights and biases for neural networks.  

A model is tuned using **hyperparameter** optimisation. By finding the optimal combination of the hyperparameter values, the error can be significantly decreased thus building the most accurate model.

After each iteration, the output is compared with the expected results, assess the accuracy, and adjust the hyperparameters if necessary. This is a repeated process which can be done manually or using one of the many optimisation techniques that exist for working with large amounts of data.

## Best Model Optimisation Techniques in Machine Learning 
### Gradient descent 
Gradient descent is one of the most common model algorithms for minimising error. The idea of this method is to iterate over the training dataset while updating the model. With every update, this method guides the model to find the target and gradually converge to the optimal value of the objective function.

<div style="text-align:center"><img src="https://lh3.googleusercontent.com/YWGY5PRhm3cEO7gjlt4EYQn3rrgB1ii8mnnO7G5GJ9V8nVZkOXWEafXMYTc3NNNYeZJTEuu4Zcg1cCck8gHS6W-TlcrlPI0vFrQ_XLYGB5oLddUCAgYYvCh4HNN74ixK-WTqdyK6" width="750" /></div>

The graph above represents how the gradient descent algorithm travels in the variable space. To get started, take a random point on the graph and arbitrarily choose a direction. If the error is getting larger, then the direction is wrong and must it must be in the opposite direction. When the error stops decreasing, the optimisation is complete and a local minimum (for the error) has been found.

In gradient descent, steps must be the same size. If the chosen learning rate is too large, the algorithm will jump around without getting closer to the right answer. If it’s too small, the computation will start mimicking exhaustive search take (brute-force search where the most optimal hyperparameters are determined by checking whether each candidate is a good match), which is, of course, inefficient.

### Adaptive Learning Rate Method
Learning rate is one of the key hyperparameters that undergo optimisation. Learning rate decides whether the model will skip certain segments of the data. If the learning rate is too high, then the model might miss on subtler aspects of the data. Alternatively, if the learning rate is too low, then the model will take significantly longer to train as it makes very tiny updates to the weights in the model.

This method is widely used in Deep Neural Networks (DNN) where methods like RMSProp, Adam, use the exponential averaging to provide effective updates and simplify the calculation. Since training requires so much computing power to train deep learning models, it is important to use efficient algorithms. Stochastic gradient descent with momentum, RMSProp, and Adam Optimiser are algorithms (amongst others) that are created specifically for deep learning optimisation .

### Stochastic Gradient Descent with Momentum
Stochastic gradient descent refers to a few samples are selected randomly instead of the whole data set for each iteration and calculating the update immediately (unlike the regular gradient decent). Suppose there are a million samples in the dataset. A typical Gradient Descent optimization technique, will use all of the samples for completing one iteration while performing the Gradient Descent, and it has to be done for every iteration until the minima is reached. Hence, it becomes computationally very expensive to perform.

Stochastic gradient descent solves this problem by using only a single sample, i.e., a batch size of one, to perform each iteration. The sample is randomly shuffled and selected for performing the iteration. The cost function of a single example is calculated at each iteration instead of the sum of the gradient of the cost function of all the examples. Since only one sample from the dataset is chosen at random for each iteration, the path taken by the algorithm to reach the minima is usually noisier than your typical Gradient Descent algorithm.

Even though it requires a higher number of iterations to reach the minima than typical Gradient Descent (due to noise), it is still computationally much less expensive than typical Gradient Descent. Hence, in most scenarios, SGD is preferred over Batch Gradient Descent for optimizing a learning algorithm.

### RMSProp 
RMSProp is a deep learning opimisation algorithm that adjusts the weights where a high gradient will have low learning rate and vice versa, such that it reduces its monotonically decreasing learning rate. It is useful to normalize the gradient itself because it balances out the step size. It can even work with the smallest batches.

### Adam Optimiser
The Adam Optimser is also a deep learning algorithm that is almost the same as the RMSProp algorithm but with momentum. Adam Optimizer can handle the noise problem and even works with large datasets and parameters.
