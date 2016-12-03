import * as assert from 'intern/chai!assert';

export function assertArraysSimilar(actual: number[], expected: number[], delta: number) {
    assert.strictEqual(actual.length, expected.length, 'Arrays are not of equal length.');

    for (let i = 0; i < actual.length; i++) {
        assert.closeTo(actual[i], expected[i], delta, `Wrong result, expected ${ JSON.stringify(expected) }, actual ${ JSON.stringify(actual) }.`);
    }
}