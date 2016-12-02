# NNet-TypeScript
A simple Neural Network library written in TypeScript.

This project was initially written in Action Script 3 as part of this project: https://github.com/s-soltys/LipSync

### Neural Network implementaiton
This implementation is a neural network with a single hidden layer.

Neurons have a sigmoid activation function (https://en.wikipedia.org/wiki/Sigmoid_function)

The backpropagation algorithm is used as the training function.

### How to use?

1. Clone the repo
2. Install npm dependencies with: `npm install`
3. Run unit test: `npm run test`

#### Known issues

Currently the training function is very slow.
Example training time:
* 7 seconds for 3 inputs, 2 outputs, 50 neurons in the hidden layer, 2000 epochs

#### Roadmap
* Implement an option to export and import the network settings.
* Open to suggestions if anybody is interested.
