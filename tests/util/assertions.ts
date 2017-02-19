export function assertArraysSimilar(actual: number[], expected: number[], delta: number) {
    expect(actual.length).toBe(expected.length, 'Arrays are not of equal length.');

    for (let i = 0; i < actual.length; i++) {
        expect(actual[i]).toBeCloseTo(expected[i], delta, `Wrong result, expected ${ JSON.stringify(expected) }, actual ${ JSON.stringify(actual) }`);
    }
}