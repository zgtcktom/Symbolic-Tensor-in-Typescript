import { Tensor, tensor } from './tensor.ts';
import { each } from './tensorfn.ts';
import { TestSuite } from './tester.ts';

export function nonzero(x: Tensor): Tensor<number> {
    let { ndim } = x;
    let data: number[] = [];
    _nonzero.for(ndim)(x, data);
    return new Tensor(data, [(data.length / ndim) | 0, ndim]);
}

const _data = Symbol('data');
const _nonzero = each(
    ({ x, i, [_data]: data }) => {
        return `if (${x[0].value}) ${data}.push(${i});`;
    },
    1,
    [_data]
);

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => nonzero(tensor([1, 1, 1, 0, 1])),
            () => tensor([[0], [1], [2], [4]])
        )
        .equal(
            () =>
                nonzero(
                    tensor([
                        [0.6, 0.0, 0.0, 0.0],
                        [0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 1.2, 0.0],
                        [0.0, 0.0, 0.0, -0.4],
                    ])
                ),
            () =>
                tensor([
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 3],
                ])
        );
