---
title: "University: Machine Learning Project"
categories:
  - university
tags:
  - machine learning
  - mathematics
  - matlab
---

### Background
In medical decision-making, a doctor observes ‘features’ of a patient and makes a decision
on the basis of these features whether the patient is healthy or has a particular disease. In this
question, you are asked to train and test a machine learning classifer which can discriminate
people who are healthy or Parkinson’s based on some training data "parkinsons.csv". There
are two features (dimension of y is d = 2), with feature y1 in column 1, y<sub>2</sub> in column 2, and
a associated training class x∈{0, 1} (i.e. 2 classes) in column 3 in the CSV file, where x=0 denotes healthy and x = 1 for Parkinson’s.

## 1 Visualisation 
![layers](https://i.imgur.com/wh6mlRg.png)

Figure 1: MATLAB Code for visualising data.

The MATLAB code shown in Figure 1 makes use of logical statements to separate entries
where x=0 and x=1 (line 3-4). This simplifies the code significantly because it gets rid of the need
for an "if-else" statement which would make the code a lot more complex. This data was then
used to plot a scatter graph which used values for each y<sub>1</sub> and y<sub>2</sub> where x=0 and x=1 (lines
8-9). This plot was then given a legend, labels, and a title (lines 10-13).

In Figure 2 we can see the resulting graph which shows healthy participants as black dots, and
participants with Parkinson’s as blue crosses.

![layers](https://i.imgur.com/JN2DP30.png)

## 2 Linear Discriminant
![layers](https://i.imgur.com/ufNRUCa.png)

Figure 3: MATLAB Code for Linear Discriminant.

b<sub>0</sub>



This question focuses on Linear Discriminant (LD) and its parameters. The questions asks us
to estimate the LD parameters L(y | b<sub>0</sub>, b<sub>1</sub>, ∑, P<sub>0</sub>, b<sub>1</sub>). Figure 3 shows the code used to determine
these parameters. The equations that were used in this code are given in the equations below.
Equation 1 was used to work out b<sub>x=0</sub> and b<sub>x=1</sub> which are the class means for x = 0 and x = 1.
These were calculated on Lines 19-20 in Figure 3. Equation 2 was used to work out P<sub>x=1</sub> and P<sub>x=1</sub>
which are the proportions for each class which is shown in Lines 22-23. The covariance matrix was
calculated using the "cov" command in MATLAB (Line 25). Since we are told to assume that the
covariance matrix for each individual class (∑<sub>0</sub>, ∑<sub>1</sub>) are the same, this means that the covariance
of the entire data set must be the same. We know this because the combined covariance matrix
is determined using the formula in Equation 4.

Since P<sub>0</sub>+P<sub>1</sub>=1, if ∑<sub>0</sub>=∑<sub>1</sub>=∑*, then:

∑ = (∑* x P<sub>0</sub>) + (∑* x P<sub>1</sub>) = ∑* x (P<sub>0</sub>+P<sub>1</sub>) = ∑*

Therefore, the covariance of the entire data set is the same as covariance for each individual class
IF covariance for each class is the same.

Finally, these parameters were used to determined the Linear Discriminant Function (L) which
was done using Equation 3. Line 27 of the code shows the equation in MATLAB. The question
informs us that if an L value is positive, then its corresponding value for y must be y∈C0. Line
28 of the code tells us exactly which values these are and their location in the data.
The values for all parameters L(y | b<sub>0</sub>, b<sub>1</sub>, ∑, P<sub>0</sub>, P<sub>1</sub>) are recorded neatly in Table 1.

![layers](https://i.imgur.com/UqmRKff.png)

|          Variable            | Denotation | Value |
|------------------------------|------------|-------|
| Total number of data entries |     N      |  195 |
|     Class mean for x = 0     | b<sub>0</sub> | (0.1230  2.1545)<sup>T</sup>  |
|     Class mean for x = 1     | b<sub>1</sub> | (0.2338  2.4561)<sup>T</sup>  |
|      Covariance Matrix       |     ∑      | (0.0081  0.0166) <br/> (0.0166  0.1465)  |
|  Class proportion for x = 0  | P<sub>0</sub> | 0.2462  |
|  Class proportion for x = 1  | P<sub>1</sub> | 0.2462  |
|     Linear Discriminant      |     L      |  (195x1) column vector |

Table 1: Estimates of Linear Discriminant Parameters

## 3 Posterior Probability 

![layers]<https://i.imgur.com/ixyjlyS.png>

Figure 4: MATLAB Code for Posterior Probabilities.

This question asks us to calculate the Posterior Probability P(C0|y) for each training data pair
y<sub>1</sub>, y<sub>2</sub> and class x=0. Line 30 of the code shown in Figure 4 finds all the Linear Discriminant
values for the healthy participants and stores in into the vector L1. The equations used and the
derivation is given in below and explained in detail. These were used from Slide 6, Page 13-17 of
the Lecture notes.

Consider the case when ∑<sub>k</sub> = ∑ , and the covariance model is identical for both classes.
Then the general case for the LDF is:

![layers](https://i.imgur.com/NWqNjvg.png)

In our case, x is y, m<sub>1</sub> is b<sub>0</sub> and m<sub>2</sub> is b<sub>1</sub>. In addition to this, P(C<sub>x</sub>) is the same as defined
in Equation 2.

This is then rearranges to give:

![layers](https://i.imgur.com/VNYnvPZ.png)

When both classes have identical covariance models, ∑, but different prior probabilities
P(C1)≠P(C2), the class separation boundary is still a linear. We must now give a Bayes
interpretation to this linear class separation.
If we wish to predict (in our case) P(C<sub>0</sub>|y), then from Bayes’ theorem we have,

![layers](https://i.imgur.com/8TZbFlI.png)

where we have simply divided through by P(y|C<sub>0)P(C<sub>0), and defined

![layers](https://i.imgur.com/ITQ9AnT.png)

Now, we model the quantity a with some linear function g(x;w):

![layers](https://i.imgur.com/VTGAFNw.png)

We then substitute for P(y|C<sub>0) to obtain:

![layers](https://i.imgur.com/i3KWMGK.png)

So, applying the logistic sigmoid activation function to the discriminant gives

![layers](https://i.imgur.com/TsedVqs.png)

we can interpret y(x;w) as the posterior probability P(C<sub>0|y). Since Equation 12 is equal to
Equation 7, this result also applies to the LDF as a "special case", and we can use L instead of a,
(L=a) as shown in Equation 13.

![layers](https://i.imgur.com/it1wcmg.png)

We must note that this done with the assumption that two classes are generated with equal
covariance matrix, but no assumptions have been made on prior probabilities. The posterior probability
is calculated in Line 35-36 in the code. The posterior probability for healthy participants
(x=0) has been calculated as well as the posterior probability for the entire data set. This
was done to visualize the data and to be able to compare the two plots. The graph for healthy
participants is shown in Figure 5, and the graph for all participants is shown in Figure 6.

![layers](https://i.imgur.com/O0RyiPn.png)
