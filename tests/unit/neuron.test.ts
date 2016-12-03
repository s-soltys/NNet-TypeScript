import 'intern';
import * as assert from 'intern/chai!assert';
import * as registerSuite from 'intern!object';
import { Neuron, NeuralNetwork, NeuralNetworkSettings, TrainingPattern } from '../../src/nnet';
import { assertArraysSimilar } from '../util/assertions';

registerSuite({
    name: 'NeuralNetwork',

    'canInstantiateANeuron': function () {
        let neuron = new Neuron(0, 0, 0);
        assert.isNotNull(neuron);
    },

    'canBeTrainedUsingSimplePatterns': function () {
        let network: NeuralNetwork = new NeuralNetwork({
            inputCount: 3,
            outputCount: 2,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        let patterns: TrainingPattern[] = [
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

        let delta: number = 0.3;
        assertArraysSimilar(network.run([1, 1, 1]), [1, 1], delta);
        assertArraysSimilar(network.run([1, 1, 0]), [1, 0], delta);
        assertArraysSimilar(network.run([0, 1, 1]), [0, 1], delta);
        assertArraysSimilar(network.run([1, 0, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 1, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 1]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 0]), [0, 0], delta);
    },

    'canBeTrainedUsingRandomlyGeneratedPatterns': function () {
        let network: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 2,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        let generatePatterns = (function () {
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

        let delta: number = 0.3;
        assertArraysSimilar(network.run([1, 1]), [1, 1], delta);
        assertArraysSimilar(network.run([1, 0]), [0, 1], delta);
        assertArraysSimilar(network.run([0, 1]), [1, 0], delta);
        assertArraysSimilar(network.run([0, 0]), [1, 1], delta);
    }

});