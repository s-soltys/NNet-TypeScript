import {Neuron} from './neuron';
import {TrainingPattern} from './training-pattern';

export interface NeuralNetworkSettings {
    inputCount: number;
    outputCount: number;
    numberOfHiddenLayers: number; // default: 1
    neuronsPerLayer: number; // default: 50
    initialWeightRange: number; // default: 1
    neuronalBias: number; // default: 1
}

export class NeuralNetwork {
    private layers: Neuron[][];
    private momentum: number = 0.5;
    private realLearningRate: number = NaN;

    constructor(s: NeuralNetworkSettings) {
        this.layers = [];

        var numberOfLayers = s.numberOfHiddenLayers + 2;

        for (let i: number = 0; i < numberOfLayers - 1; i++) {
            this.layers[i] = this.createLayer(s.neuronsPerLayer, s.inputCount, s.neuronalBias, s.initialWeightRange);
        }

        this.layers[numberOfLayers - 1] = this.createLayer(s.outputCount, s.neuronsPerLayer, s.neuronalBias, s.initialWeightRange);
    }

    private createLayer(neuronsCount: number, inputCount: number, bias: number, weightRange: number): Neuron[] {
        var layer: Neuron[] = new Array<Neuron>();

        for (let i: number = 0; i < neuronsCount; i++) {
            var neuron: Neuron = new Neuron(inputCount, bias, weightRange);
            layer.push(neuron);
        }

        return layer;
    }

    run(input: number[]): number[] {
        var layerOutputs: number[][] = [];

        for (let i: number = 0; i <= this.layers.length; i++) {
            layerOutputs[i] = [];
        }

        var inputForCurrentLayer: number[] = input;
        for (let layerIndex = 0; layerIndex < this.layers.length; layerIndex++) {
            let currentLayer = this.layers[layerIndex];
            let outputForCurrentLayer = layerOutputs[layerIndex + 1];

            currentLayer.forEach((neuron: Neuron) => {
                neuron.calculateOutputValue(inputForCurrentLayer);
                outputForCurrentLayer.push(neuron.outputValue);
            });

            inputForCurrentLayer = outputForCurrentLayer;
        }

        return layerOutputs[layerOutputs.length - 1];
    }

    train(patterns: TrainingPattern[], epochs: number = 50, learningRate: number = 0.5, targetMSE: number = 0.025): number {
        return this.trainWithGeneratedPatterns(() => NeuralNetwork.shufflePatterns(patterns), epochs, learningRate, targetMSE);
    }

    trainWithGeneratedPatterns(generatePatterns: () => TrainingPattern[], epochs: number = 50, learningRate: number = 0.5, targetMSE: number = 0.025): number {
        if (isNaN(this.realLearningRate)) {
            this.realLearningRate = learningRate;
        }

        let measuredMSE: number;
        for (let epoch = 0; epoch < epochs; epoch++) {
            measuredMSE = 0;
            
            let patterns = generatePatterns();
            patterns.forEach(pattern => {
                this.run(pattern.input);
                measuredMSE += this.adjust(pattern.output, this.realLearningRate);
            });

            measuredMSE = measuredMSE / patterns.length;
            
            this.realLearningRate = learningRate * measuredMSE;

            if (measuredMSE <= targetMSE) {
                break;
            }
        }

        return measuredMSE;
    }

    private adjust(outputArray: number[], learningRate: number): number {
        var MSEsum: number = 0;
        var layerCount: number = this.layers.length - 1;
        var error: number[][] = [];

        for (let l: number = layerCount; l >= 0; --l) {
            var layer: Neuron[] = this.layers[l];

            error[l] = [];
            for (let i: number = 0; i < layer[0].layerSize; i++) {
                error[l].push(0);
            }

            var nError: number = 0;
            for (let n: number = 0; n < layer.length; n++) {
                var neuron: Neuron = layer[n];

                if (l == layerCount) {
                    nError = outputArray[n] - neuron.outputValue;
                    MSEsum += nError * nError;
                } else {
                    nError = error[l + 1][n];
                }

                neuron.adjustWeights(nError, learningRate, this.momentum, error[l]);
            }
        }

        return MSEsum / (this.layers[layerCount].length);
    }

    public static shufflePatterns(array: TrainingPattern[]): TrainingPattern[] {
        for (let i: number = 0; i < array.length; i++) {
            var randomIndex: number = Math.floor(Math.random() * array.length)
            var randomElement: TrainingPattern = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = randomElement;
        }
        
        return array;
    }
    
}