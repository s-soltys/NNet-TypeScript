import * as assert from 'intern/chai!assert';
import * as registerSuite from 'intern!object';
import {Neuron, NeuralNetwork, TrainingPattern} from '../../src/nnet';

function assertEqual(actual: number[], expected: number[], delta: number) {
    assert.strictEqual(actual.length, expected.length);

    for (let i = 0; i < actual.length; i++) {
        var diff: number = Math.abs(expected[i] - actual[i]);
        assert.isBelow(diff, delta, `Wrong result, expected ${JSON.stringify(expected)}, actual ${JSON.stringify(actual)}. `);
    }
}

registerSuite({
    name: 'hello',

    greet: function() {
        var neuron: Neuron = new Neuron(0, 0, 0);
        assert.isNotNull(neuron);
    },

    training: function() {
        var network: NeuralNetwork = new NeuralNetwork(3, 2, 50);

        var patterns: TrainingPattern[] = [
            { input: [1, 1, 1], output: [1, 1] },
            { input: [1, 1, 0], output: [1, 0] },
            { input: [0, 1, 1], output: [0, 1] },
            { input: [1, 0, 0], output: [0, 0] },
            { input: [0, 1, 0], output: [0, 0] },
            { input: [0, 0, 1], output: [0, 0] },
            { input: [0, 0, 0], output: [0, 0] },
            { input: [0.9, 0.9, 0.9], output: [1, 1] },
            { input: [0.9, 0.9, 0], output: [1, 0] },
            { input: [0, 0.9, 0.9], output: [0, 1] },
            { input: [0, 0, 0], output: [0, 0] },
        ];

        network.train(patterns, 2000, 0.75, 0.025);
        
        var delta: number = 0.3;
        
        assertEqual(network.run([1, 1, 1]), [1, 1], delta);
        assertEqual(network.run([1, 1, 0]), [1, 0], delta);
        assertEqual(network.run([0, 1, 1]), [0, 1], delta);
        assertEqual(network.run([1, 0, 0]), [0, 0], delta);
        assertEqual(network.run([0, 1, 0]), [0, 0], delta);
        assertEqual(network.run([0, 0, 1]), [0, 0], delta);
        assertEqual(network.run([0, 0, 0]), [0, 0], delta);
    }

});