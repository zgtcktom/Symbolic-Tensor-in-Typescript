import { empty } from './empty.ts';
import { tensor, Tensor } from './tensor.ts';
import { _axis } from './tensorlike.ts';
import { TestSuite } from './tester.ts';

export function squeeze<T>(x: Tensor<T>, axis?: number | number[]): Tensor<T> {
    let { data, shape, strides, offset, ndim } = x;
    let _shape: number[], _strides: number[];

    if (axis == undefined) {
        _shape = [];
        _strides = [];

        for (let axis = 0; axis < ndim; axis++) {
            let dim = shape[axis];
            if (dim != 1) {
                _shape.push(dim);
                _strides.push(strides[axis]);
            }
        }
        if (_shape.length == ndim) return x;
    } else if (Array.isArray(axis)) {
        let mask: boolean[] = Array(ndim);
        let _ndim = ndim;

        for (let i = 0, n = axis.length; i < n; i++) {
            let ax = _axis(axis[i], ndim);
            if (!mask[ax] && shape[ax] == 1) {
                mask[ax] = true;
                _ndim--;
            }
        }

        if (_ndim == ndim) return x;

        _shape = Array(_ndim);
        _strides = Array(_ndim);
        for (let axis = 0, _axis = 0; axis < ndim; axis++) {
            if (!mask[axis]) {
                _shape[_axis] = shape[axis];
                _strides[_axis] = strides[axis];
                _axis++;
            }
        }
    } else {
        axis = _axis(axis, ndim);
        if (shape[axis] != 1) return x;
        _shape = shape.toSpliced(axis, 1);
        _strides = strides.toSpliced(axis, 1);
    }

    return new Tensor(data, _shape, _strides, offset);
}

export function unsqueeze<T>(x: Tensor<T>, axis: number): Tensor<T> {
    let { data, shape, strides, offset, ndim } = x;
    axis = _axis(axis, ndim + 1);

    shape = shape.toSpliced(axis, 0, 1);
    strides = strides.toSpliced(axis, 0, 0);

    return new Tensor(data, shape, strides, offset);
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => unsqueeze(tensor([1, 2, 3, 4]), 0),
            () => tensor([[1, 2, 3, 4]])
        )
        .equal(
            () => unsqueeze(tensor([1, 2, 3, 4]), 1),
            () => tensor([[1], [2], [3], [4]])
        )
        .equal(
            () => squeeze(empty(2, 1, 2, 1, 2)).shape,
            () => [2, 2, 2]
        )
        .equal(
            () => squeeze(empty(2, 1, 2, 1, 2), 0).shape,
            () => [2, 1, 2, 1, 2]
        )
        .equal(
            () => squeeze(empty(2, 1, 2, 1, 2), 1).shape,
            () => [2, 2, 1, 2]
        )
        .equal(
            () => squeeze(empty(2, 1, 2, 1, 2), [1, 2, 3]).shape,
            () => [2, 2, 2]
        )
        .equal(
            () => squeeze(tensor(1)).shape,
            () => []
        );
