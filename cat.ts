import { Tensor, tensor } from './tensor.ts';
import { empty } from './empty.ts';
import { _axis } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function cat<T>(tensors: Tensor<T>[], axis?: number, out?: Tensor | null): Tensor<T>;
export function cat(tensors: Tensor[], axis?: number, out?: Tensor | null): Tensor;
export function cat(tensors: Tensor[], axis: number = 0, out?: Tensor | null): Tensor {
    let n = tensors.length;
    if (n == 0) {
        throw new Error('cannot concatenate an empty array of tensors');
    }
    let { shape, ndim } = tensors[0];
    axis = _axis(axis, ndim);

    let dim = shape[axis];
    for (let i = 1; i < n; i++) {
        let { shape: _shape, ndim: _ndim } = tensors[i];
        dim += _shape[axis];
        if (_ndim != ndim) {
            throw new Error('all tensors must have the same number of dimensions');
        }
        for (let _axis = 0; _axis < ndim; _axis++) {
            if (_axis != axis && _shape[_axis] != shape[_axis]) {
                throw new Error('all tensors must have the same shape except along the concatenation axis');
            }
        }
    }

    out ??= empty(shape.with(axis, dim));
    let { data, strides, offset } = out;

    let view = new Tensor(data, shape.slice(), strides, offset);
    let stride = strides[axis];
    for (let i = 0; i < n; i++) {
        let tensor = tensors[i];
        let dim = tensor.shape[axis];
        view.shape[axis] = dim;
        view.set(tensor);
        view.offset += stride * dim;
    }

    return out;
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => {
                let x = tensor([
                    [0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497],
                ]);
                return cat([x, x, x], 0);
            },
            () =>
                tensor([
                    [0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497],
                    [0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497],
                    [0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497],
                ])
        )
        .equal(
            () => {
                let x = tensor([
                    [0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497],
                ]);
                return cat([x, x, x], 1);
            },
            () =>
                tensor([
                    [0.658, -1.0969, -0.4614, 0.658, -1.0969, -0.4614, 0.658, -1.0969, -0.4614],
                    [-0.1034, -0.579, 0.1497, -0.1034, -0.579, 0.1497, -0.1034, -0.579, 0.1497],
                ])
        );
