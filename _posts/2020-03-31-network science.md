---
title: "University: Network Science Project"
categories:
  - university
tags:
  - network science
  - computer science
  - matlab
  - python
---

### Background
This report presents methodology and results for the AM30NE Coursework. A combination
of Python and MATLAB has been used to solve the problems for the three questions
asked. The student number "160106794" has been used as the seed.


## 1 Erdös-Rényi Random Graphs
We must generate a sequence of G(N, p) graphs which have p = 0.15 (constant) and different N values N = 10, 50, 100, 200, 500, 1000. Here, N is the number of nodes and p is the probability. These graphs are shown in Figures 1 - 6.

![layers]<https://i.imgur.com/BSg76tm.png>

Table 1 displays all the information in a presentable manner with values for the average clustering
coefficient (C), average degree (k) and the values for (C)/(k). All values are given to an accuracy
of 4 decimal places.

|No. of Nodes (N)| Avg Clustering Coefficient (C)| Avg Degree (k)| (C)/(k)|
|:----------------:|:-------------------------------:|:---------------:|:--------:|
|10| 0 |1.8| 0|
|50 |0.1236| 7.04| 0.0176|
|100| 0.1440| 14.42| 0.0100|
|200 |0.1496| 30.08| 0.0050|
|500| 0.1498| 74.596| 0.0020|
|1000 |0.1504 |150.058| 0.0010|

Table 1: This table shows values for the average degree and clustering coefficient for the graphs
in Figures 1 - 6.

Table 2 adds a new column for the values of f(N) which is given by f(N) = 1/N−1. Values
for f(N) were plot against the values for (C)/(k) to analyse what happens as the size is increased.
From Figure 7, it is clear that as the size of N increases, the values for f(N) and (C)/(k) become very similar. Figure 8 shows the same plot on logarithmic scales which verifies this observation as it
the two lines are almost identical for higher values of N.

|No. of Nodes (N)| Avg Clustering Coefficient (C)| Avg Degree (k)| (C)/(k)|f(N)
|:----------------:|:-------------------------------:|:---------------:|:--------:|:---:|
|10| 0 |1.8| 0| 0.1111|
|50 |0.1236| 7.04| 0.0176| 0.0204|
|100| 0.1440| 14.42| 0.0100|0.1010 |
|200 |0.1496| 30.08| 0.0050| 0.005|
|500| 0.1498| 74.596| 0.0020| 0.002|
|1000 |0.1504 |150.058| 0.0010|0.001 |

Table 2: This table adds a new column for the values of f(N) = 1/N−1.

![layers]<https://i.imgur.com/oUXUFI7.png>

Now, we calculate the average distance in each one of the graphs, we are asked to plot this
against the number of nodes of the graph. The results for this are shown in Table 3. Figure 9
shows N plotted against the average distance.

![layers]<https://i.imgur.com/PHDBFed.png>

|Nodes (N)|Average Distance|
|10| 0|
|50 |2.1608|
|100| 1.9497|
|200| 1.8577|
|500 |1.8505|
|1000| 1.8498|

Table 3: This table shows number
of nodes N and their average
distance.

Now, we must generate a sequence of G(N, L) graphs with L = 2N and the values for
N remaining the same. Then we can determine their degree distribution and plot them. These
graphs and their plots are shown below with left being the graphs and the right being its degree
distribution.

![layers]<https://i.imgur.com/gPsX5EI.png>
![layers]<https://i.imgur.com/BVaMzut.png>

