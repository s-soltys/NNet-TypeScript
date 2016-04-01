import * as assert from 'intern/chai!assert';
import * as registerSuite from 'intern!object';
import {Neuron, NeuralNetwork, TrainingPattern} from '../../src/nnet';

registerSuite({
    name: 'hello',

    greet: function() {
        var neuron: Neuron = new Neuron(0, 0, 0);
        assert.isNotNull(neuron);
    },
    
    training: function () {
        var network: NeuralNetwork = new NeuralNetwork(3, 2, 1000);
        
        var patterns: TrainingPattern[] = [
            { input: [1, 1, 1], output: [0, 0] },
            { input: [1, 1, 0], output: [1, 0] },
            { input: [0, 1, 1], output: [0, 1] },
            { input: [0, 0, 0], output: [1, 1] },
        ];
        
        network.train(patterns, 100, 100, 0.9);
        
        assert.equal(network.run([1, 1, 0]), [1, 0]);
    }
});