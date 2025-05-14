import { tensor, Tensor } from './tensor.ts';
import { empty } from './empty.ts';
import { _axis, shapeEqual } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function stack<T>(tensors: Tensor<T>[], axis?: number, out?: Tensor<T> | null): Tensor<T>;
export function stack(tensors: Tensor[], axis?: number, out?: Tensor | null): Tensor;
export function stack(tensors: Tensor[], axis: number = 0, out?: Tensor | null): Tensor {
    let n = tensors.length;
    if (n == 0) {
        throw new Error('cannot stack an empty array of tensors');
    }
    let { shape, ndim } = tensors[0];
    axis = _axis(axis, ndim + 1);

    for (let i = 1; i < n; i++) {
        let { shape: _shape, ndim: _ndim } = tensors[i];
        if (_ndim != ndim) {
            throw new Error('all tensors must have the same number of dimensions');
        }
        if (!shapeEqual(_shape, shape)) {
            throw new Error('all tensors must have the same shape');
        }
    }

    out ??= empty(shape.toSpliced(axis, 0, n));
    let { data, strides, offset } = out;

    let view = new Tensor(data, shape, strides.toSpliced(axis, 1), offset);
    let stride = strides[axis];
    for (let i = 0; i < n; i++) {
        view.set(tensors[i]);
        view.offset += stride;
    }

    return out;
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => {
                let x = tensor([
                    [0.3367, 0.1288, 0.2345],
                    [0.2303, -1.1229, -0.1863],
                ]);
                return stack([x, x]);
            },
            () =>
                tensor([
                    [
                        [0.3367, 0.1288, 0.2345],
                        [0.2303, -1.1229, -0.1863],
                    ],
                    [
                        [0.3367, 0.1288, 0.2345],
                        [0.2303, -1.1229, -0.1863],
                    ],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [0.3367, 0.1288, 0.2345],
                    [0.2303, -1.1229, -0.1863],
                ]);
                return stack([x, x], 1);
            },
            () =>
                tensor([
                    [
                        [0.3367, 0.1288, 0.2345],
                        [0.3367, 0.1288, 0.2345],
                    ],
                    [
                        [0.2303, -1.1229, -0.1863],
                        [0.2303, -1.1229, -0.1863],
                    ],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [0.3367, 0.1288, 0.2345],
                    [0.2303, -1.1229, -0.1863],
                ]);
                return stack([x, x], 2);
            },
            () =>
                tensor([
                    [
                        [0.3367, 0.3367],
                        [0.1288, 0.1288],
                        [0.2345, 0.2345],
                    ],
                    [
                        [0.2303, 0.2303],
                        [-1.1229, -1.1229],
                        [-0.1863, -0.1863],
                    ],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [0.3367, 0.1288, 0.2345],
                    [0.2303, -1.1229, -0.1863],
                ]);
                return stack([x, x], -1);
            },
            () =>
                tensor([
                    [
                        [0.3367, 0.3367],
                        [0.1288, 0.1288],
                        [0.2345, 0.2345],
                    ],
                    [
                        [0.2303, 0.2303],
                        [-1.1229, -1.1229],
                        [-0.1863, -0.1863],
                    ],
                ])
        );
