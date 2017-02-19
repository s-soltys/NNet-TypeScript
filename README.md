# NNet-TypeScript
![travis build results](https://travis-ci.org/s-soltys/NNet-TypeScript.svg?branch=master)

A simple Neural Network library written in TypeScript.
This project was initially written in Action Script 3 as part of this project: https://github.com/s-soltys/LipSync

### Neural Network implementaiton
This implementation is a neural network with a single hidden layer.
Neurons have a sigmoid activation function (https://en.wikipedia.org/wiki/Sigmoid_function)
The backpropagation algorithm is used as the training function.

### How to install
```
npm install --save nnet-typescript
```

### How to use
Example implementation of a XOR function:
```
// Create the Neural Network
let nnet: NeuralNetwork = new NeuralNetwork({
    inputCount: 2,
    outputCount: 1,
    numberOfHiddenLayers: 0,
    neuronsPerLayer: 30,
    initialWeightRange: 1,
    neuronalBias: 0.5
});

// XOR truth table
let patterns: TrainingPattern[] = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];

// training the network using the generated patterns
// Training parameters:
// Pattern generation function, shuffle patterns in each epoch, number of epochs, learning rate, target MSE
nnet.train(() => patterns, true, 2000, 0.8, 0.001);

// Expected results
const delta = 0.2;
assertNetworkResult(nnet, [1, 1], 0, delta);
assertNetworkResult(nnet, [1, 0], 1, delta);
assertNetworkResult(nnet, [0, 1], 1, delta);
assertNetworkResult(nnet, [0, 0], 0, delta);
```