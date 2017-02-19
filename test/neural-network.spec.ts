import { Neuron, NeuralNetwork, NeuralNetworkSettings, TrainingPattern } from '../src/nnet';

describe("NeuralNetwork should", () => {
    it('can instantiate a neuron', () => {
        let neuron = new Neuron(0, 0, 0);
        expect(neuron).not.toBeNull();
    }),

    it('can create a XOR function', () => {
        let nnet: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 1,
            numberOfHiddenLayers: 0,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 0.5
        });

        let patterns: TrainingPattern[] = [
            { input: [0, 0], output: [0] },
            { input: [0, 1], output: [1] },
            { input: [1, 0], output: [1] },
            { input: [1, 1], output: [0] }
        ];

        nnet.train(() => patterns, true, 2000, 0.8, 0.001);

        const delta = 0.2;
        assertNetworkResult(nnet, [1, 1], 0, delta);
        assertNetworkResult(nnet, [1, 0], 1, delta);
        assertNetworkResult(nnet, [0, 1], 1, delta);
        assertNetworkResult(nnet, [0, 0], 0, delta);
    });

    it('can be trained using patterns with discrete output', () => {
        let nnet: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 1,
            numberOfHiddenLayers: 0,
            neuronsPerLayer: 60,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        let patterns: TrainingPattern[] = [
            { input: [1, 1], output: [1.00] },
            { input: [1, 0], output: [0.75] },
            { input: [0, 1], output: [0.25] },
            { input: [0, 0], output: [0.00] }
        ];

        nnet.train(() => patterns, true, 2000, 0.8, 0.0001);

        const delta = 0.15;
        assertNetworkResult(nnet, [1, 1], 1.00, delta);
        assertNetworkResult(nnet, [1, 0], 0.75, delta);
        assertNetworkResult(nnet, [0, 1], 0.25, delta);
        assertNetworkResult(nnet, [0, 0], 0.00, delta);
    });

    function assertNetworkResult(network: NeuralNetwork, input: number[], expected: number, maxError: number) {
        let output = network.run(input);
        let result = output[0];
        let error = Math.abs(expected - result);
        expect(error < maxError).toBeTruthy(`Expected output ${result} to be close to ${expected} for input [${input}]`);
    }
});