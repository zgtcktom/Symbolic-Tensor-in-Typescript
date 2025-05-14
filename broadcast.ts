import { empty } from './empty.ts';
import { tensor } from './tensor.ts';
import { TensorLike } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function broadcast<T extends TensorLike>(...xs: T[]): T[] {
    let n = xs.length;

    let shapes: number[][] = Array(n);
    for (let i = 0; i < n; i++) {
        shapes[i] = xs[i].shape;
    }
    let shape = broadcastShape(shapes);

    let out = Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = xs[i].expand(shape);
    }

    return out;
}

export function broadcastShape(shapes: number[][]): number[] {
    let n = shapes.length;

    let ndim = 0;
    for (let i = 0; i < n; i++) {
        let _ndim = shapes[i].length;
        if (_ndim > ndim) ndim = _ndim;
    }

    let shape: number[] = Array(ndim);
    for (let axis = 0; axis < ndim; axis++) {
        let dim = 1;
        for (let i = 0; i < n; i++) {
            let shape = shapes[i];
            let _ndim = shape.length;
            if (axis >= _ndim) continue;

            let _dim = shape[_ndim - 1 - axis];
            if (_dim == 1) continue;

            if (dim == 1) dim = _dim;
            else if (dim != _dim) {
                throw new Error(`incompatible shapes: ${shapes}`);
            }
        }
        shape[ndim - 1 - axis] = dim;
    }

    return shape;
}

export const test = (suite: TestSuite) =>
    suite
        .assert(() => {
            let x = tensor(0);
            let y = tensor([0, 0, 0]);
            [x, y] = broadcast(x, y);
            return x.equal(y);
        })
        .equal(
            () => broadcast(tensor([1, 2, 3]), empty(3, 3))[0],
            () =>
                tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ])
        )
        .equal(
            () => broadcastShape([[1, 2, 3], []]),
            () => [1, 2, 3]
        )
        .equal(
            () =>
                broadcastShape([
                    [4, 2, 3],
                    [2, 1],
                ]),
            () => [4, 2, 3]
        )
        .equal(
            () =>
                broadcastShape([
                    [1, 2, 3],
                    [1, 1, 1],
                ]),
            () => [1, 2, 3]
        );
