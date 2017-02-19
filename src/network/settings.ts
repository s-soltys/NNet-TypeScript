export class NeuralNetworkSettings {
    inputCount: number;
    outputCount: number;
    numberOfHiddenLayers: number = 1;
    neuronsPerLayer: number = 50;
    initialWeightRange: number = 1;
    neuronalBias: number = 1;
}