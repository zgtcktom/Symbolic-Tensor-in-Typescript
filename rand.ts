import { empty } from './empty.ts';
import { Tensor } from './tensor.ts';
import { unwrap } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function rand(shape: number[]): Tensor<number>;
export function rand(...shape: number[]): Tensor<number>;
export function rand(...shape: number[] | [number[]]): Tensor<number> {
    shape = unwrap(shape);

    let x = empty(shape);
    let { data, size } = x;
    for (let i = 0; i < size; i++) {
        data[i] = Math.random();
    }

    return x;
}

export function randn(shape: number[]): Tensor<number>;
export function randn(...shape: number[]): Tensor<number>;
export function randn(...shape: number[] | [number[]]): Tensor<number> {
    shape = unwrap(shape);

    let x = empty(shape);
    let { data, size } = x;
    for (let i = 0; i < size; i++) {
        let u;
        while ((u = Math.random()) <= 0);
        data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
    }

    return x;
}

export function randint(high: number, shape: number[]): Tensor<number>;
export function randint(low: number, high: number, shape: number[]): Tensor<number>;
export function randint(low: number, high: number | number[], shape?: number[]): Tensor<number> {
    if (shape == undefined) {
        [low, high, shape] = [0, low, high as number[]];
    } else high = high as number;

    let x = empty(shape);
    let { data, size } = x;
    for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * (high - low)) + low;
    }

    return x;
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => rand(1, 2, 3).shape,
            () => [1, 2, 3]
        )
        .equal(
            () => randn().shape,
            () => []
        );
