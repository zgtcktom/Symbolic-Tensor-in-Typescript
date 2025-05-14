import { Tensor } from './tensor.ts';
import { unwrap } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function empty<T = any>(shape: number[]): Tensor<T>;
export function empty<T = any>(...shape: number[]): Tensor<T>;
export function empty<T = any>(...shape: number[] | [number[]]): Tensor<T> {
    shape = unwrap(shape);

    let ndim = shape.length;
    let strides: number[] = Array(ndim);
    let size = 1;

    for (let axis = ndim - 1; axis >= 0; axis--) {
        strides[axis] = size;
        size *= shape[axis];
    }

    return new Tensor(Array(size), shape, strides, 0);
}

export function full<T>(shape: number[], value: T): Tensor<T> {
    let x = empty(shape);
    x.data.fill(value);
    return x;
}

export function zeros(shape: number[]): Tensor<number>;
export function zeros(...shape: number[]): Tensor<number>;
export function zeros(...shape: number[] | [number[]]): Tensor<number> {
    shape = unwrap(shape);
    return full(shape, 0);
}

export function ones(shape: number[]): Tensor<number>;
export function ones(...shape: number[]): Tensor<number>;
export function ones(...shape: number[] | [number[]]): Tensor<number> {
    shape = unwrap(shape);
    return full(shape, 1);
}

export const test = (suite: TestSuite) =>
    suite.equal(
        () => empty([1, 2, 3]),
        () => empty(1, 2, 3)
    );
