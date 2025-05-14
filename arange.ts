import { tensor, Tensor } from './tensor.ts';
import { TestSuite } from './tester.ts';

export function arange(start: number, end?: number, step: number = 1): Tensor<number> {
    if (end == undefined) {
        [start, end] = [0, start];
    }

    let size = Math.max(0, ((end - start) / step) | 0);
    let data = Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = start + i * step;
    }

    return new Tensor(data, [size], [1], 0);
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => arange(5),
            () => tensor([0, 1, 2, 3, 4])
        )
        .equal(
            () => arange(1, 4),
            () => tensor([1, 2, 3])
        )
        .equal(
            () => arange(1, 2.5, 0.5),
            () => tensor([1.0, 1.5, 2.0])
        )
        .equal(
            () => arange(3, 7, 2),
            () => tensor([3, 5])
        )
        .equal(
            () => arange(10, 0, 1),
            () => tensor([])
        )
        .equal(
            () => arange(10, 0, -1),
            () => tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        );
