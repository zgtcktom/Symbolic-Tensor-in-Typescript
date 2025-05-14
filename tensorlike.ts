import { slice, Slice } from './slice.ts';
import { tensor, Tensor } from './tensor.ts';

export type BasicIndex = number | Slice | null;
export type AdvancedIndex = Tensor<number | boolean> | undefined;
export type Index = BasicIndex | AdvancedIndex;

export type ND<T> = T | ND<T>[];

export abstract class TensorLike<T = any> {
    shape: number[];
    ndim: number;
    size: number;

    constructor(shape: number[], ndim?: number, size?: number) {
        ndim ??= shape.length;
        if (size == undefined) {
            size = 1;
            for (let i = 0; i < ndim; i++) size *= shape[i];
        }
        this.shape = shape;
        this.ndim = ndim;
        this.size = size;
    }

    get length() {
        return this.shape[0];
    }

    abstract forward(): Tensor<T>;

    abstract copy(): TensorLike<T>;

    abstract contiguous(): TensorLike<T>;

    abstract flatten(): TensorLike<T>;
    abstract flatten(start: number, end: number): TensorLike<T>;

    abstract ravel(): TensorLike<T>;

    abstract expand(shape: number[]): TensorLike<T>;
    abstract expand(...shape: number[]): TensorLike<T>;

    abstract reshape(shape: number[]): TensorLike<T>;
    abstract reshape(...shape: number[]): TensorLike<T>;

    abstract transpose(): TensorLike<T>;
    abstract transpose(axes: number[]): TensorLike<T>;
    abstract transpose(...axes: number[]): TensorLike<T>;

    abstract unsqueeze(axis: number): TensorLike<T>;

    abstract squeeze(axis?: number | number[]): TensorLike<T>;

    abstract at(...index: (Index | any)[]): TensorLike<T>;

    abstract sum(axis?: number | number[], keepdim?: boolean): TensorLike<T>;

    abstract add(x: any): TensorLike<T>;

    abstract sub(x: any): TensorLike<T>;

    abstract mul(x: any): TensorLike<T>;

    abstract div(x: any): TensorLike<T>;

    abstract reciprocal(): TensorLike<T>;

    abstract neg(): TensorLike<T>;

    abstract pow(x: any): TensorLike<T>;

    abstract log(): TensorLike<T>;

    abstract abs(): TensorLike<T>;

    abstract sign(): TensorLike<T>;

    abstract dot(x: any): TensorLike<T>;

    abstract matmul(x: any): TensorLike<T>;

    abstract [Symbol.iterator](): Iterator<TensorLike<T>>;
}

export function unwrap(shape: number[] | [number[]]): number[] {
    return shape.length && Array.isArray(shape[0]) ? shape[0] : (shape as number[]);
}

export function _axis(axis: number, ndim: number): number {
    if (axis >= ndim || axis < -ndim) {
        throw new Error(`axis ${axis} out of bounds for ndim ${ndim}`);
    }
    return axis < 0 ? axis + ndim : axis;
}

export function _axes(axes: number | number[] | null | undefined, ndim: number): number[] {
    if (typeof axes == 'number') return [_axis(axes, ndim)];

    if (axes?.length) {
        let mask: boolean[] = Array(ndim);
        for (let i = 0, n = axes.length; i < n; i++) {
            mask[_axis(axes[i], ndim)] = true;
        }

        axes = [];
        for (let i = 0; i < ndim; i++) {
            if (mask[i]) axes.push(i);
        }
        return axes;
    }

    axes = Array(ndim);
    for (let i = 0; i < ndim; i++) axes[i] = i;
    return axes;
}

export function shapeEqual(x: number[], y: number[]): boolean {
    if (x === y) return true;
    let n = x.length;
    if (n !== y.length) return false;
    for (let i = 0; i < n; i++) {
        if (x[i] !== y[i]) return false;
    }
    return true;
}

export function normalizeIndex(index: any[], inplace: boolean = true): Index[] {
    let n = index.length;
    let _index: Index[] = inplace ? index : Array(n);
    for (let i = 0; i < n; i++) {
        let ind = index[i];
        if (ind instanceof Tensor) {
            if (ind.ndim > 0) {
                _index[i] = ind;
                continue;
            }
            ind = ind.item();
        }

        _index[i] = ind == null ? null : typeof ind == 'number' || ind instanceof Slice ? ind : typeof ind == 'string' ? slice(ind) : tensor(ind);
    }
    return _index;
}
