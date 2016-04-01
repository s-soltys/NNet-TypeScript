export class Neuron {
    private bias: number;
    private momentum: number;
    private inputs: number[];
    private weights: number[];
    private momentums: number[];
    
    size: number;
    value: number;

    constructor(inputsCount: number, bias: number, weightRange: number = 1) {
        this.size = inputsCount;
        this.bias = bias;
        this.momentum = 0;
        
        this.inputs = new Array<number>(this.size);
        this.weights = new Array<number>(this.size);
        this.momentums = new Array<number>(this.size);

        for (let i: number = 0; i < this.size; i++) {
            this.inputs[i] = NaN;
            this.weights[i] = (Math.random() * (weightRange + weightRange)) - weightRange;
            this.momentums[i] = 0;
        }
    }

    adjustWeights(nError: number, learningRate: number, globalMomentum: number, error: number[]): void {
        var delta: number = nError * this.value * (1 - this.value);

        for (let i: number = 0; i < this.size; i++) {
            var weightChange: number = delta * this.inputs[i] * learningRate + this.momentums[i] * globalMomentum;
            this.momentums[i] = weightChange;
            this.weights[i] += weightChange;
            error[i] += delta * this.weights[i];
        }

        var biasChange: number = delta * learningRate + this.momentum * globalMomentum;
        this.momentum = biasChange;
        this.bias += biasChange;
    }

    /*
        adjustN(nError:number, learningRate:number, globalMomentum:number, error:Array):void {
        var delta:number = nError * this.value * (1 - this.value);
    	
        for (var i:number = 0; i < size; i++) {
            var weightChange:number = delta * inputs[i] * learningRate + momentums[i] * globalMomentum;
            error[i] += delta * weights[i];
        }
    }
    */

    calculateValue(inputsArray: number[]): number {
        var sum: number = 0;

        for (let i: number = 0; i < this.size; i++) {
            this.inputs[i] = inputsArray[i];
            sum += this.weights[i] * this.inputs[i];
        }

        this.value = 1 / (1 + Math.exp(-1 * (sum + this.bias)));
        return this.value;
    }

}