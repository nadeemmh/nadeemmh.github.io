## Deep Learning: Neural Networks

### Definitions: 
- Neuron/Unit: A mathematical function that models the functioning of a biological neuron (human brain). A neuron computes the weighted average of its input, and this sum is passed through a function. The output of the neuron can then be sent as input to the neurons of another layer, which could repeat the same computation.
- Layers: A general term that applies to a collection of 'nodes' operating together at a specific depth (position) within a neural network. NNs are constructed from 3 type of layers: Input layer — initial data for the neural network. Hidden layers — intermediate layer between input and output layer and place where all the computation is done. Output layer — produce the result for given inputs.
- Weights: A parameter within a neural network that transforms input data within the network's hidden layers.

<br></p>

![layers](https://user-images.githubusercontent.com/71942932/103166839-e9481980-481d-11eb-925b-e5cfc5132da6.png)

In the image above, each circle represents a "unit/neuron", each line represents a "connection", and each section of units represents a "layer". Each connection from one unit to the next in a network will have an assigned weight. Weights are between 0 and 1, and represent the strengths of the connection between 2 units.

So, when input is received, it is passed to the next unit through the connection and the input is multiplied by the weight assigned to that particular connection. Then, a weighted sum is computed with each of the connections going to a particular neuron. This sum is then passed through an "activation" function which transforms the result to a number between 0 and 1. 

As a function, we can write the following:
> output = activation(weighted sum of inputs).

This result of the transformation from the activation function is then passed onto the next neuron in the next layer. This happens over and over again until it reaches the output layer. 

The output layer has output units according to the possible categories of classifications. Example, if the above model has the task of classifying whether images are "cats" or "dogs", then the two units in the output layer represent two possible outputs, one being "cats", and one being "dogs". 

Note: Weights are continuously changing to each the optimized weights for each connection as the model continues to learn from the data.

PYTHON:

`model = Sequential([
	Dense("number of units", "input_shape", "activation")
])`

Note: Only the first layer within the sequential model requires an "input shape", because the model needs to understand the shape of the data its initially going to be dealing with.
