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

![layers](https://i.imgur.com/BSg76tm.png

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

![layers](https://i.imgur.com/oUXUFI7.png)

Now, we calculate the average distance in each one of the graphs, we are asked to plot this
against the number of nodes of the graph. The results for this are shown in Table 3. Figure 9
shows N plotted against the average distance.

![layers](https://i.imgur.com/PHDBFed.png)

|Nodes (N)|Average Distance|
|:---:|:---:|
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

![layers](https://i.imgur.com/gPsX5EI.png)
![layers](https://i.imgur.com/BVaMzut.png)

## 2 The Watts-Strogatz (WS) Model
Here, we use the Watts-Strogatz (WS) Model. First, we must create an undirected
graph, G, with 100 nodes arranged in a circle formation where p = 0. The visualization of this
graph’s adjacency matrix is shown in Figure 22, where black represents a link and white represents
no links.

![layers](https://i.imgur.com/kaVCGAG.png)

Next, we generate a second graph, G'', with p = 0.3. The visualization of this graph’s adjacency
matrix is shown in Figure 23.

![layers](https://i.imgur.com/sZvSY9B.png)

The degree distribution for G and G'' can be seen in the Figure 24 and Figure 25 respectively.
The degree distribution tells us that the average degree for G = 6 and G'' = 6.

![layers](https://i.imgur.com/qLPbnTR.png)

Next, we work with varying probability pd = 0.005, 0.01, 0.05, 0.1, 0.5, 1.0. The graph in Figure
shows the average clustering coefficient for each probability. The values for each probability and
its corresponding clustering coefficient is shown in Table 4.

|Probability P<sub>d</sub>|Average Clustering Coefficient|
|:---:|:---:|
|0.005| 0.5|
|0.01 |0.4887|
|0.05| 0.4203|
|0.1 |0.3637|
|0.5 |0.0688|
|1.0 |0.0089|

Table 4

![layers](https://i.imgur.com/KNVKkRF.png)

Now, we use the same model to calculate and plot the average distance relative to its probability.
The plot is given in Figure 27 and its corresponding table is Table 5. From the table, it is clear
that as the probability increases, the average distance decreases exponentially and is a smooth
transition. This verifies our intuition, since we know that as the probability of nodes being
connected increases, the average distance between nodes in the network should be lower.

|Probability P<sub>d</sub>|Average Clustering Coefficient|
|:---:|:---:|
|0.005| 25.3769|
|0.01| 17.9267|
|0.05| 7.0483|
|0.1| 5.8737|
|0.5| 4.1987|
|1.0| 3.9594|

![layers](https://i.imgur.com/lqdAeYS.png)

## 3 The Barabasi-Albert (BA) Model
We now use the Barabasi-Albert (BA) Model. We generate a graph with N=100
nodes using the BA model with m = 2 and starting with 2 connected nodes. A visualization of
the adjacency matrix is shown in Figure 28. This visualization shows black dot as linked nodes
and the blank area as unlinked nodes.

![layers](https://i.imgur.com/oJMwpXE.png)

Here, we use the KL algorithm to partition this graph into two sets of 50 nodes each. A graph
showing the cut size value at each accepted node exchange is shown in Figure 29. The number of
node swaps stop at the 17th value because the cut size repeats the same value after this for the
rest of the output. As a result of this repetition, we can conclude that the minimum cut found
was of size 53.
