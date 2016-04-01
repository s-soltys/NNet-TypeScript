import {Neuron} from './neuron';
import {TrainingPattern} from './training-pattern';

export class NeuralNetwork {
    private layers: Neuron[][];
    private momentum: number = 0.5;
    private neuronalBias: number = 1;
    private initialWeightRange: number = 1;
    private realLearningRate: number = NaN;
    private hiddenLayers: number = 2;

    constructor(inputs: number, outputs: number, neuronsPerLayer: number): void {
        this.layers = [];

        this.layers[0] = this.createLayer(neuronsPerLayer, inputs, this.neuronalBias, this.initialWeightRange);
        for (let i: number = 1; i < this.hiddenLayers; i++) {
            this.layers[i] = this.createLayer(neuronsPerLayer, inputs, this.neuronalBias, this.initialWeightRange);
        }
        this.layers[this.hiddenLayers] = this.createLayer(outputs, neuronsPerLayer, this.neuronalBias, this.initialWeightRange);
    }

    private createLayer(neurons: number, inputs: number, bias: number, weightRange: number): Neuron[] {
        var layer: Neuron[] = new Array<Neuron>();
        
        for (let i: number = 0; i < neurons; i++) {
            var neuron: Neuron = new Neuron(inputs, bias, weightRange);
            layer.push(neuron);
        }
        
        return layer;
    }


    run(inputArray: number[]): number[] {
        var layerOutputs: number[][] = [];
        
        for (let i: number = 0; i <= this.layers.length; i++) {
            layerOutputs[i] = [];
        }

        var inputs: number[] = inputArray;
        for (let i = 0; i < this.layers.length; i++) {
            var output: number[] = layerOutputs[i + 1];

            this.layers[i].forEach((neuron: Neuron) => {
                var calculationResult = neuron.calculateValue(inputs);
                output.push(calculationResult);
            });
            
            inputs = output;
        }

        return layerOutputs[layerOutputs.length - 1];
    }

    train(patterns: TrainingPattern[], epochs: number, learningRate: number, targetMSE: number = 0.02): number {
        if (isNaN(this.realLearningRate)) {
            this.realLearningRate = learningRate;
        }

        var MSE: number = 0;
        for (var r: number = 0; r < epochs; r++) {
            patterns = this.shufflePatterns(patterns);

            MSE = 0;
            for (var i: number = 0; i < patterns.length; i++) {
                var input: Array<number> = (patterns[i] as TrainingPattern).input;
                var output: Array<number> = (patterns[i] as TrainingPattern).output;

                this.run(input);
                MSE += this.adjust(output, this.realLearningRate);
            }

            MSE = MSE / patterns.length;
            this.realLearningRate = learningRate * MSE;

            if (MSE <= targetMSE) {
                break;
            }
        }

        return MSE;
    }

    private adjust(outputArray: number[], learningRate: number): number {
        var MSEsum: number = 0;
        var layerCount: number = this.layers.length - 1;
        var error: number[][] = [];

        for (var l: number = layerCount; l >= 0; --l) {
            var layer: Neuron[] = this.layers[l];

            error[l] = [];
            for (var i: number = 0; i < layer[0].size; i++) {
                error[l].push(0);
            }

            var nError: number = 0;
            for (var n: number = 0; n < layer.length; n++) {
                var neuron: Neuron = layer[n];

                if (l == layerCount) {
                    nError = outputArray[n] - neuron.value;
                    MSEsum += nError * nError;
                } else {
                    nError = error[l + 1][n];
                }
                
                neuron.adjustWeights(nError, learningRate, this.momentum, error[l]);
            }
        }
        
        return MSEsum / (this.layers[layerCount].length);
    }


    shufflePatterns(array: TrainingPattern[]): TrainingPattern[] {
        for (let i: number = 0; i < array.length; i++) {
            var randomIndex: number = Math.floor(Math.random() * array.length)
            var randomElement: TrainingPattern = array[i];
            array[i] = array[randomIndex];
            array[randomIndex] = randomElement;
        }
        return array;
    }

    // save(): ByteArray {
    //     var output: ByteArray = new ByteArray();

    //     output.writenumber(LP.order);
    //     output.writenumber(LipsyncSettings.outputCount);
    //     output.writenumber(LipsyncSettings.samplingDecimate);
    //     output.writenumber(LipsyncSettings.windowLength);

    //     output.writeDouble(momentum);
    //     output.writeDouble(realLearningRate);
    //     output.writenumber(layers.length);

    //     for (var l: number = 0; l < layers.length; l++) {
    //         var layer: Array<Neuron> = layers[l];

    //         output.writenumber(layer.length);
    //         for (var n: number = 0; n < layer.length; n++) {
    //             var neuron: Neuron = layer[n];

    //             output.writeDouble(neuron.value);
    //             output.writeDouble(neuron.bias);
    //             output.writeDouble(neuron.momentum);

    //             output.writenumber(neuron.size);
    //             for (var s: number = 0; s < neuron.size; s++) {
    //                 output.writeDouble(neuron.inputs[s]);
    //                 output.writeDouble(neuron.weights[s]);
    //                 output.writeDouble(neuron.momentums[s]);
    //             }
    //         }
    //     }

    //     output.compress();
    //     output.position = 0;

    //     return output;
    // }

    // load(input: ByteArray): void {
    //     input.uncompress();
    //     input.position = 0;

    //     LP.order = input.readnumber();
    //     LipsyncSettings.outputCount = input.readnumber();
    //     LipsyncSettings.samplingDecimate = input.readnumber();
    //     LipsyncSettings.windowLength = input.readnumber();

    //     this.momentum = input.readDouble();
    //     this.realLearningRate = input.readDouble();

    //     var layersLength: number = input.readnumber();
    //     layers = new Array(layersLength);

    //     for (var l: number = 0; l < layersLength; l++) {
    //         var layerLength: number = input.readnumber();
    //         var layer: Array<Neuron> = new Array<Neuron>();

    //         for (var n: number = 0; n < layerLength; n++) {
    //             var neuron: Neuron = new Neuron();

    //             neuron.value = input.readDouble();
    //             neuron.bias = input.readDouble();
    //             neuron.momentum = input.readDouble();

    //             neuron.size = input.readnumber();
    //             neuron.inputs = new Array<number>(neuron.size);
    //             neuron.weights = new Array<number>(neuron.size);
    //             neuron.momentums = new Array<number>(neuron.size);

    //             for (var s: number = 0; s < neuron.size; s++) {
    //                 neuron.inputs[s] = input.readDouble();
    //                 neuron.weights[s] = input.readDouble();
    //                 neuron.momentums[s] = input.readDouble();
    //             }

    //             layer[n] = neuron;


    //         }

    //         layers[l] = layer;
    //     }

    // }

}