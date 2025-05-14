import { broadcast } from './broadcast.ts';
import { empty } from './empty.ts';
import { gt } from './operator.ts';
import { Tensor, tensor } from './tensor.ts';
import { map } from './tensorfn.ts';
import { TestSuite } from './tester.ts';

export function where(cond: Tensor<boolean>): Tensor<number>;
export function where<T>(cond: Tensor<boolean>, x: Tensor<T>, y: Tensor<T>, out?: Tensor<T> | null): Tensor<T>;
export function where(cond: Tensor<boolean>, x: Tensor, y: Tensor, out?: Tensor | null): Tensor;
export function where(cond: Tensor<boolean>, x?: Tensor, y?: Tensor, out?: Tensor | null): Tensor {
    if (x != undefined && y != undefined) {
        [cond, x, y] = broadcast(cond, x, y);
        out ??= empty(x.shape);
        _where.for(x.ndim)(cond, x, y, out);
        return out;
    }

    return cond.nonzero();
}

const _where = map(({ x }) => {
    return `${x[0].value} ? ${x[1].value} : ${x[2].value}`;
}, 3);

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => {
                let x = tensor([
                    [-0.462, 0.3139],
                    [0.3898, -0.7197],
                    [0.0478, -0.1657],
                ]);
                return where(gt(x, tensor(0)), tensor(1), tensor(0));
            },
            () =>
                tensor([
                    [0, 1],
                    [1, 0],
                    [1, 0],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [-0.462, 0.3139],
                    [0.3898, -0.7197],
                    [0.0478, -0.1657],
                ]);
                let y = empty(3, 2).set(1);
                return where(gt(x, tensor(0)), x, y);
            },
            () =>
                tensor([
                    [1.0, 0.3139],
                    [0.3898, 1.0],
                    [0.0478, 1.0],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [1.0779, 0.0383],
                    [-0.8785, -1.1089],
                ]);
                return where(gt(x, tensor(0)), x, tensor(0));
            },
            () =>
                tensor([
                    [1.0779, 0.0383],
                    [0.0, 0.0],
                ])
        );
