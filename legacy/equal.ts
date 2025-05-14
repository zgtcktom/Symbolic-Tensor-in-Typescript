import { tensor, Tensor } from '../tensor';
import { each } from '../tensorfn';
import { shapeEqual } from '../tensorlike';
import { TestSuite } from '../tester';

export function equal(x: Tensor, y: Tensor): boolean {
    if (x === y) return true;

    if (!shapeEqual(x.shape, y.shape)) return false;

    return _equal.for(x.ndim)(x, y);
}

const _equal = each(({ x }) => {
    return `if (${x[0].value} !== ${x[1].value}) return false;`;
}, 2).return(() => 'true');

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => equal(tensor(5), tensor(5)),
            () => true
        )
        .equal(
            () => equal(tensor(5), tensor(6)),
            () => false
        )
        .equal(
            () => equal(tensor([1, 2, 3]), tensor([1, 2, 3])),
            () => true
        )
        .equal(
            () => equal(tensor([1, 2, 3]), tensor([1, 2, 4])),
            () => false
        )
        .equal(
            () => equal(tensor([1, 2]), tensor([1, 2, 3])),
            () => false
        )
        .equal(
            () =>
                equal(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ]),
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                ),
            () => true
        )
        .equal(
            () =>
                equal(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ]),
                    tensor([
                        [1, 2],
                        [3, 5],
                    ])
                ),
            () => false
        )
        .equal(
            () =>
                equal(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ]),
                    tensor([
                        [1, 2, 3],
                        [4, 5, 6],
                    ])
                ),
            () => false
        )
        .equal(
            () => equal(tensor([]), tensor([])),
            () => true
        )
        .equal(
            () => equal(tensor([]), tensor([1])),
            () => false
        )
        .equal(
            () => equal(tensor(5), tensor([5])),
            () => false
        )
        .equal(
            () =>
                equal(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ]),
                    tensor([1, 2, 3, 4])
                ),
            () => false
        )
        .equal(
            () => {
                let t = tensor([1, 2, 3]);
                return equal(t, t);
            },
            () => true
        )
        .equal(
            () => {
                let t = tensor([
                    [1, 2],
                    [2, 1],
                ]);
                return equal(t, t.transpose());
            },
            () => true
        )
        .equal(
            () => {
                let t = tensor([
                    [1, 2],
                    [3, 4],
                ]);
                return equal(t, t.transpose());
            },
            () => false
        );
