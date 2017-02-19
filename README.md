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
npm install --save-dev nnet-typescript
```

### How to use
Example implementation of a XOR function:
```
// Create the Neural Network
let network: NeuralNetwork = new NeuralNetwork({
    inputCount: 2,
    outputCount: 1,
    numberOfHiddenLayers: 1,
    neuronsPerLayer: 20,
    initialWeightRange: 1,
    neuronalBias: 1
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
network.train(() => patterns, true, 3000, 0.6, 0.05);

// Expected results
const delta = 0.05;
expect(network.run([1, 1])).toBeCloseTo(0, delta);
expect(network.run([1, 0])).toBeCloseTo(1, delta);
expect(network.run([0, 1])).toBeCloseTo(1, delta);
expect(network.run([0, 0])).toBeCloseTo(0, delta);
```