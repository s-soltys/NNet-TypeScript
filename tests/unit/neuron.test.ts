import * as assert from 'intern/chai!assert';
import * as registerSuite from 'intern!object';
import {Neuron, NeuralNetwork, NeuralNetworkSettings, TrainingPattern} from '../../src/nnet';

function assertEqual(actual: number[], expected: number[], delta: number) {
    assert.strictEqual(actual.length, expected.length);

    for (let i = 0; i < actual.length; i++) {
        var diff: number = Math.abs(expected[i] - actual[i]);
        assert.isBelow(diff, delta, `Wrong result, expected ${JSON.stringify(expected)}, actual ${JSON.stringify(actual)}. `);
    }
}

registerSuite({
    name: 'NeuralNetwork',

    canInstantiateANeuron: function() {
        var neuron: Neuron = new Neuron(0, 0, 0);
        assert.isNotNull(neuron);
    },

    canBeTrainedUsingSimplePatterns: function() {
        var network: NeuralNetwork = new NeuralNetwork({
            inputCount: 3,
            outputCount: 2,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        var patterns: TrainingPattern[] = [
            { input: [1, 1, 1], output: [1, 1] },
            { input: [1, 1, 0], output: [1, 0] },
            { input: [0, 1, 1], output: [0, 1] },
            { input: [1, 0, 0], output: [0, 0] },
            { input: [0, 1, 0], output: [0, 0] },
            { input: [0, 0, 1], output: [0, 0] },
            { input: [0, 0, 0], output: [0, 0] },
            { input: [0.9, 0.9, 0.9], output: [1, 1] },
            { input: [0.9, 0.9, 0.0], output: [1, 0] },
            { input: [0.0, 0.9, 0.9], output: [0, 1] },
            { input: [0.0, 0.0, 0.0], output: [0, 0] }
        ];

        network.train(patterns, 1000, 0.75, 0.02);

        var delta: number = 0.25;
        assertEqual(network.run([1, 1, 1]), [1, 1], delta);
        assertEqual(network.run([1, 1, 0]), [1, 0], delta);
        assertEqual(network.run([0, 1, 1]), [0, 1], delta);
        assertEqual(network.run([1, 0, 0]), [0, 0], delta);
        assertEqual(network.run([0, 1, 0]), [0, 0], delta);
        assertEqual(network.run([0, 0, 1]), [0, 0], delta);
        assertEqual(network.run([0, 0, 0]), [0, 0], delta);
    },

    canBeTrainedUsingRandomlyGeneratedPatterns: function() {
        var network: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 2,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        var generatePatterns = (function() {
            let g1 = () => 1 - 0.2 * Math.random();
            let g0 = () => 0.2 * Math.random();
            return () => NeuralNetwork.shufflePatterns([
                { input: [g1(), g1()], output: [1, 1] },
                { input: [g1(), g0()], output: [0, 1] },
                { input: [g0(), g1()], output: [1, 0] },
                { input: [g0(), g0()], output: [1, 1] }
            ]);
        } ());

        network.trainWithGeneratedPatterns(generatePatterns, 1000, 0.8, 0.015);

        var delta: number = 0.25;
        assertEqual(network.run([1, 1]), [1, 1], delta);
        assertEqual(network.run([1, 0]), [0, 1], delta);
        assertEqual(network.run([0, 1]), [1, 0], delta);
        assertEqual(network.run([0, 0]), [1, 1], delta);
    }

});