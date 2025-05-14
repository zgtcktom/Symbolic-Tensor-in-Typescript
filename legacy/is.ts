import { Tensor, tensor } from '../tensor';

export function contiguous(x: Tensor): boolean {
    let { shape, strides, ndim } = x;

    for (let axis = ndim - 1, size = 1; axis >= 0; axis--) {
        let stride = strides[axis];
        if (stride < 0 || stride != size) {
            return false;
        }
        size *= shape[axis];
    }

    return true;
}

export function scalar(x: any): boolean {
    if (x && typeof x == 'object') {
        return tensor(x).ndim == 0;
    }
    return true;
}