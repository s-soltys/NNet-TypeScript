export class Neuron {
    private bias: number;
    private momentum: number;
    private inputs: number[];
    private weights: number[];
    private momentums: number[];
    public layerSize: number;
    public outputValue: number;

    constructor(inputsCount: number, bias: number, weightRange: number = 1) {
        this.layerSize = inputsCount;
        this.bias = bias;
        this.momentum = 0;
        
        this.inputs = new Array<number>(this.layerSize);
        this.weights = new Array<number>(this.layerSize);
        this.momentums = new Array<number>(this.layerSize);

        for (let i: number = 0; i < this.layerSize; i++) {
            this.inputs[i] = NaN;
            this.weights[i] = (Math.random() * (weightRange + weightRange)) - weightRange;
            this.momentums[i] = 0;
        }
    }

    adjustWeights(nError: number, learningRate: number, globalMomentum: number, error: number[]): void {
        var delta: number = nError * this.outputValue * (1 - this.outputValue);

        for (let i: number = 0; i < this.layerSize; i++) {
            var weightChange: number = delta * this.inputs[i] * learningRate + this.momentums[i] * globalMomentum;
            this.momentums[i] = weightChange;
            this.weights[i] += weightChange;
            error[i] += delta * this.weights[i];
        }

        var biasChange: number = delta * learningRate + this.momentum * globalMomentum;
        this.momentum = biasChange;
        this.bias += biasChange;
    }
    
    calculateOutputValue(currentInputs: number[]): void {
        var sum: number = 0;
        this.inputs = currentInputs;
         
        for (let i: number = 0; i < this.layerSize; i++) {
            sum += this.weights[i] * this.inputs[i];
        }

        this.outputValue = 1 / (1 + Math.exp(-1 * (sum + this.bias)));
    }

}