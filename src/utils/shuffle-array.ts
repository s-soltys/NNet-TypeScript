export function shuffle<T>(array: T[]): T[] {
    for (let i: number = 0; i < array.length; i++) {
        const randomIndex = Math.floor(Math.random() * array.length);
        const randomElement = array[i];
        array[i] = array[randomIndex];
        array[randomIndex] = randomElement;
    }
    
    return array;
}