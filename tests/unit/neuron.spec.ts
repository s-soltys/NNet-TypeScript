import { Neuron, NeuralNetwork, NeuralNetworkSettings, TrainingPattern } from '../../src/nnet';
import { assertArraysSimilar } from '../util/assertions';

describe("NeuralNetwork should", () => {

    it('canInstantiateANeuron', () => {
        let neuron = new Neuron(0, 0, 0);
        expect(neuron).not.toBeNull();
    }),

    // xit('neuronalActivationFunctionIsClamped', () => {
    //     let ret0 = Neuron.activate(0, 1);
    //     let ret1 = Neuron.activate(1, 1);

    //     expect(ret0).toBeCloseTo(0, 0.01);
    //     expect(ret1).toBeCloseTo(1, 0.01);
    // }),

    it('canBeTrainedUsingSimplePatterns', () => {
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

        network.train(() => patterns, true, 500, 0.8, 0.01);

        let delta: number = 0.1;
        assertArraysSimilar(network.run([1, 1, 1]), [1, 1], delta);
        assertArraysSimilar(network.run([1, 1, 0]), [1, 0], delta);
        assertArraysSimilar(network.run([0, 1, 1]), [0, 1], delta);
        assertArraysSimilar(network.run([1, 0, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 1, 0]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 1]), [0, 0], delta);
        assertArraysSimilar(network.run([0, 0, 0]), [0, 0], delta);
    })

    it('canBeTrainedUsingRandomlyGeneratedPatterns', () => {
        let network: NeuralNetwork = new NeuralNetwork({
            inputCount: 2,
            outputCount: 2,
            numberOfHiddenLayers: 1,
            neuronsPerLayer: 30,
            initialWeightRange: 1,
            neuronalBias: 1
        });

        const variability = 0.1;
        let generatePatterns = () => {
            let g1 = () => 1 - variability * Math.random();
            let g0 = () => variability * Math.random();
            return [
                { input: [g1(), g1()], output: [1, 1] },
                { input: [g1(), g0()], output: [0, 1] },
                { input: [g0(), g1()], output: [1, 0] },
                { input: [g0(), g0()], output: [1, 1] }
            ];
        };

        network.train(generatePatterns, true, 3000, 0.8, 0.01);

        let delta: number = 0.1;
        assertArraysSimilar(network.run([1, 1]), [1, 1], delta);
        assertArraysSimilar(network.run([1, 0]), [0, 1], delta);
        assertArraysSimilar(network.run([0, 1]), [1, 0], delta);
        assertArraysSimilar(network.run([0, 0]), [1, 1], delta);
    });
});