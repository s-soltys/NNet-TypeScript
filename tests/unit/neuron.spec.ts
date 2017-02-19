import { Neuron, NeuralNetwork, NeuralNetworkSettings, TrainingPattern } from '../../src/nnet';
import { assertArraysSimilar } from '../util/assertions';

describe("NeuralNetwork should", () => {
    it('can instantiate a neuron', () => {
        let neuron = new Neuron(0, 0, 0);
        expect(neuron).not.toBeNull();
    }),

    it('can create a XOR function', () => {
        let network: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 1,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 20,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        let patterns: TrainingPattern[] = [
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }
        ];

        network.train(() => patterns, true, 3000, 0.6, 0.05);

        const delta = 0.05;
        expect(network.run([1, 1])).toBeCloseTo(0, delta);
        expect(network.run([1, 0])).toBeCloseTo(1, delta);
        expect(network.run([0, 1])).toBeCloseTo(1, delta);
        expect(network.run([0, 0])).toBeCloseTo(0, delta);
    });

    it('can be trained using simple patterns', () => {
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

        network.train(() => patterns, true, 5000, 0.6, 0.05);

        const delta = 0.1;
        assertArraysSimilar(network.run([1, 1, 1]), [1, 1], delta);
        assertArraysSimilar(network.run([1, 1, 0]), [1, 0], delta);
        assertArraysSimilar(network.run([0, 1, 1]), [0, 1], delta);
        assertArraysSimilar(network.run([1, 0, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 1, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 1]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 0]), [0, 0], delta);
    });
});