import { Slice, slice } from './slice.ts';
import { each, map, mapfn } from './tensorfn.ts';
import { nonzero } from './nonzero.ts';
import { squeeze, unsqueeze } from './squeeze.ts';
import { and, eq, ge, gt, le, lt, ne, or } from './operator.ts';
import { empty } from './empty.ts';
import { arange } from './arange.ts';
import { add, sub, mul, div, pow, neg, log, dot, reciprocal, abs, sum, sign } from './math.ts';
import { matmul } from './matmul.ts';
import { normalizeIndex, shapeEqual, TensorLike, unwrap, _axis, type AdvancedIndex, type ND } from './tensorlike.ts';
import { broadcastShape } from './broadcast.ts';
import { stack } from './stack.ts';
import { TestSuite } from './tester.ts';

export const is = {
    contiguous(x: Tensor): boolean {
        let { shape, strides, ndim } = x;

        for (let axis = ndim - 1, size = 1; axis >= 0; axis--) {
            let stride = strides[axis];
            if (stride < 0 || stride != size) {
                return false;
            }
            size *= shape[axis];
        }

        return true;
    },
    scalar(x: any): boolean {
        if (x && typeof x == 'object') {
            return tensor(x).ndim == 0;
        }
        return true;
    },
};

export class Tensor<T = any> extends TensorLike<T> {
    data: T[];
    offset: number;
    strides: number[];

    constructor(data: T[], shape?: number[], strides?: number[], offset: number = 0) {
        shape ??= [data.length];

        if (strides == undefined) {
            let ndim = shape.length;
            let size = 1;
            strides = Array(ndim);
            for (let axis = ndim - 1; axis >= 0; axis--) {
                strides[axis] = size;
                size *= shape[axis];
            }
            super(shape, ndim, size);
        } else {
            super(shape);
        }

        this.data = data;
        this.offset = offset;
        this.strides = strides;
    }

    forward(): Tensor<T> {
        return this;
    }

    array() {
        let { data, shape, strides, offset, ndim } = this;
        if (ndim == 0) return this.item();
        return _array(0, offset, data, ndim, shape, strides);
    }

    item(): T {
        return this.data[this.offset];
    }

    copy(): Tensor<T> {
        if (is.contiguous(this)) {
            return new Tensor(this.data.slice(this.offset, this.offset + this.size), this.shape, this.strides);
        }

        let data = Array(this.size);
        _flatten.for(this.ndim)(this, data);
        return new Tensor(data, this.shape);
    }

    contiguous(): Tensor<T> {
        if (is.contiguous(this)) {
            return this;
        }

        let data = Array(this.size);
        _flatten.for(this.ndim)(this, data);
        return new Tensor(data, this.shape);
    }

    flatten(start: number = 0, end: number = -1): Tensor<T> {
        let { ndim } = this;
        if (ndim == 1) return this;
        if (ndim == 0) {
            return new Tensor(this.data, [1], [1], this.offset);
        }

        start = _axis(start, ndim);
        end = _axis(end, ndim);

        let { size } = this;
        let shape: number[];
        if (start == 0 && end == ndim - 1) {
            shape = [size];
        } else {
            let { shape: _shape } = this;
            shape = Array(ndim - (end - start));
            let i = 0,
                j = 0;
            while (i < start) shape[j++] = _shape[i++];
            let size = 1;
            while (i <= end) size *= _shape[i++];
            shape[j++] = size;
            while (i < ndim) shape[j++] = _shape[i++];
        }

        if (is.contiguous(this)) {
            return new Tensor(this.data, shape, undefined, this.offset);
        }

        let data = Array(size);
        _flatten.for(ndim)(this, data);
        return new Tensor(data, shape);
    }

    ravel(): Tensor<T> {
        let { size } = this;

        if (is.contiguous(this)) {
            return new Tensor(this.data, [size], [1], this.offset);
        }

        let data = Array(size);
        _flatten.for(this.ndim)(this, data);
        return new Tensor(data, [size], [1]);
    }

    expand(...shape: number[] | [number[]]): Tensor<T> {
        shape = unwrap(shape);

        if (shapeEqual(shape, this.shape)) return this;

        let { data, shape: _shape, ndim: _ndim, strides: _strides, offset } = this;
        let ndim = shape.length;
        let strides = Array(ndim).fill(0);

        if (ndim < _ndim) {
            throw new Error(`target ndim ${ndim} < tensor ndim ${_ndim}`);
        }

        for (let axis = ndim - _ndim, _axis = 0; axis < ndim; axis++, _axis++) {
            let _dim = _shape[_axis];
            if (_dim == 1) continue;

            let dim = shape[axis];
            if (dim == -1) {
                shape[axis] = _dim;
            } else if (dim != _dim) {
                throw new Error(`tensor shape [${_shape}] doesn't match [${shape}] at axis ${axis}`);
            }
            strides[axis] = _strides[_axis];
        }

        return new Tensor(data, shape, strides, offset);
    }

    reshape(...shape: number[] | [number[]]): Tensor<T> {
        shape = unwrap(shape);
        let { shape: shape_, ndim: ndim_, size: size_ } = this;
        let ndim = shape.length;
        let size = 1;
        let inferred = -1;
        for (let axis = 0; axis < ndim; axis++) {
            let dim = shape[axis];
            if (dim < 0) {
                if (inferred != -1) {
                    throw new Error(`more than one inferred dimension`);
                }
                inferred = axis;
            } else {
                size *= dim;
            }
        }

        if (inferred != -1) {
            size *= shape[inferred] = size ? (size_ / size) | 0 : 0;
        }

        if (size != size_) {
            throw new Error(`invalid shape [${shape}] for size ${size_}`);
        }

        let compatible = false;
        for (let axis = 0, axis_ = 0; ; axis++, axis_++) {
            while (axis < ndim && shape[axis] == 1) axis++;
            while (axis_ < ndim_ && shape_[axis_] == 1) axis_++;

            if (axis == ndim) {
                if (axis_ == ndim_) compatible = true;
            } else if (axis_ == ndim_) {
                if (axis == ndim) compatible = true;
            } else if (shape[axis] == shape_[axis_]) continue;
            break;
        }

        if (compatible) {
            let { strides: strides_ } = this;
            let strides = Array(ndim);
            for (let axis = 0, axis_ = 0; axis < ndim; axis++) {
                if (shape[axis] == 1) {
                    strides[axis] = 1;
                    continue;
                }
                while (axis_ < ndim_ && shape_[axis_] == 1) axis_++;
                strides[axis] = axis_ < ndim_ ? strides_[axis_++] : 1;
            }
            return new Tensor(this.data, shape, strides, this.offset);
        }

        if (is.contiguous(this)) {
            return new Tensor(this.data, shape, undefined, this.offset);
        }

        let data = Array(size_);
        _flatten.for(ndim_)(this, data);
        return new Tensor(data, shape);
    }

    transpose(...axes: number[] | [number[]]): Tensor<T> {
        let { data, shape, strides, offset } = this;

        axes = unwrap(axes);

        if (axes.length == 0 || (axes.length == 1 && axes[0] == null)) {
            return new Tensor(data, shape.toReversed(), strides.toReversed(), offset);
        }

        let { ndim } = this;

        if (axes.length != ndim) {
            throw new Error(`axes.length != ndim`);
        }
        let shape_ = Array(ndim);
        let strides_ = Array(ndim);

        for (let i = 0; i < ndim; i++) {
            let axis = _axis(axes[i], ndim);
            shape_[i] = shape[axis];
            strides_[i] = strides[axis];
        }

        return new Tensor(data, shape_, strides_, offset);
    }

    unsqueeze(axis: number): Tensor<T> {
        return unsqueeze(this, axis);
    }

    squeeze(axis?: number | number[]): Tensor<T> {
        return squeeze(this, axis);
    }

    at(...index: any[]): Tensor<T> {
        let { data, shape: _shape, strides: _strides, ndim, offset } = this;
        let shape: number[] = [];
        let strides: number[] = [];

        let advanced;
        let axis = 0;
        for (let ind of normalizeIndex(index)) {
            if (ind == null) {
                shape.push(1);
                strides.push(0);
                continue;
            }

            if (typeof ind == 'number') {
                let dim = _shape[axis];
                if (ind < -dim || ind >= dim) {
                    throw new Error(`index ${ind} out of bounds at dimension ${axis}`);
                }
                if (ind < 0) ind += dim;
                offset += _strides[axis] * ind;
                axis++;
                continue;
            }

            if (ind instanceof Slice) {
                let stride = _strides[axis];
                let { start, step, length } = ind.indices(_shape[axis]);
                offset += stride * start;
                shape.push(length);
                strides.push(stride * step);
                axis++;
                continue;
            }

            advanced ??= Array(ndim) as AdvancedIndex[];
            advanced[axis] = ind;
            shape.push(_shape[axis]);
            strides.push(_strides[axis]);
            axis++;
        }

        for (; axis < ndim; axis++) {
            shape.push(_shape[axis]);
            strides.push(_strides[axis]);
        }

        let base = new Tensor(data, shape, strides, offset);
        if (advanced) return new IndexedTensor(base, advanced).forward();
        return base;
    }

    sum(axis?: number | number[], keepdim: boolean = false, out?: Tensor | null): Tensor<T> {
        return sum(this, axis, keepdim, out);
    }

    add(x: any, out?: Tensor | null): Tensor<T> {
        return add(this, tensor(x), out);
    }

    sub(x: any, out?: Tensor | null): Tensor<T> {
        return sub(this, tensor(x), out);
    }

    mul(x: any, out?: Tensor | null): Tensor<T> {
        return mul(this, tensor(x), out);
    }

    div(x: any, out?: Tensor | null): Tensor<T> {
        return div(this, tensor(x), out);
    }

    reciprocal(out?: Tensor | null): Tensor<T> {
        return reciprocal(this, out);
    }

    neg(out?: Tensor | null): Tensor<T> {
        return neg(this, out);
    }

    pow(x: any, out?: Tensor | null): Tensor<T> {
        return pow(this, tensor(x), out);
    }

    log(out?: Tensor | null): Tensor<T> {
        return log(this, out);
    }

    abs(out?: Tensor | null): Tensor<T> {
        return abs(this, out);
    }

    sign(out?: Tensor | null): Tensor<T> {
        return sign(this, out);
    }

    dot(x: any, out?: Tensor | null): Tensor<T> {
        return dot(this, tensor(x), out);
    }

    matmul(x: any, out?: Tensor | null): Tensor<T> {
        return matmul(this, tensor(x), out);
    }

    *[Symbol.iterator](): Iterator<Tensor<T>> {
        if (this.ndim == 0) {
            throw new Error(`iteration over a 0-d tensor`);
        }
        let { data, shape, strides, offset } = this;
        let dim = shape[0];
        let stride = strides[0];
        shape = shape.slice(1);
        strides = strides.slice(1);
        for (let i = 0; i < dim; i++) {
            yield new Tensor(data, shape, strides, offset + i * stride);
        }
    }

    [Symbol.for('nodejs.util.inspect.custom')]() {
        return `tensor(${this.toString()})`;
    }

    nonzero(): Tensor<number> {
        return nonzero(this);
    }

    set(x: any): Tensor<T> {
        if (is.contiguous(this)) {
            if (is.scalar(x)) {
                if (x instanceof Tensor) x = x.item();
                this.data.fill(x, this.offset, this.offset + this.size);
                return this;
            }

            if (x instanceof Tensor && shapeEqual(this.shape, x.shape) && is.contiguous(x)) {
                let { data, offset, size } = this;
                let { data: _data, offset: _offset } = x;
                for (let i = 0; i < size; i++) {
                    data[offset + i] = _data[_offset + i];
                }
                return this;
            }
        }

        x = tensor(x).expand(this.shape);
        _set.for(this.ndim)(x, this);

        return this;
    }

    equal(other: any): boolean {
        if (this === other) return true;
        let x = tensor(other);

        if (!shapeEqual(this.shape, x.shape)) return false;

        return _equal.for(this.ndim)(this, x);
    }

    toString(digits = 4, leadingSpace = 0): string {
        if (this.ndim == 0) {
            let x = this.item();
            return `${typeof x == 'number' && !Number.isInteger(x) ? x.toFixed(digits) : x}`;
        }
        let results = Array.from(this).map(element => element.toString(digits, leadingSpace + 1));
        const MAX_LINE_WIDTH = 75;
        if (this.ndim > 1 || results.reduce((a, b) => a + b.length, 0) > MAX_LINE_WIDTH) {
            return `[${results.join(', \n' + ' '.repeat(leadingSpace + 1))}]`;
        }
        return `[${results.join(', ')}]`;
    }

    [Symbol.toPrimitive]() {
        return this.item();
    }

    map(fn: (...args: any[]) => any, out?: Tensor | null): Tensor {
        out ??= empty(this.shape);
        mapfn(fn, 1).for(this.ndim)(this, out);
        return out;
    }

    // logical

    gt(x: any, out?: Tensor | null): Tensor {
        return gt(this, tensor(x), out);
    }

    ge(x: any, out?: Tensor | null): Tensor {
        return ge(this, tensor(x), out);
    }

    lt(x: any, out?: Tensor | null): Tensor {
        return lt(this, tensor(x), out);
    }

    le(x: any, out?: Tensor | null): Tensor {
        return le(this, tensor(x), out);
    }

    eq(x: any, out?: Tensor | null): Tensor {
        return eq(this, tensor(x), out);
    }

    ne(x: any, out?: Tensor | null): Tensor {
        return ne(this, tensor(x), out);
    }

    and(x: any, out?: Tensor | null): Tensor {
        return and(this, tensor(x), out);
    }

    or(x: any, out?: Tensor | null): Tensor {
        return or(this, tensor(x), out);
    }
}

function _array<T>(axis: number, offset: number, data: T[], ndim: number, shape: number[], strides: number[]): ND<T> {
    let dim = shape[axis];
    let stride = strides[axis];
    let array: ND<T> = Array(dim);

    if (axis == ndim - 1) {
        for (let i = 0; i < dim; i++) {
            array[i] = data[offset];
            offset += stride;
        }
    } else {
        for (let i = 0; i < dim; i++) {
            array[i] = _array(axis + 1, offset, data, ndim, shape, strides);
            offset += stride;
        }
    }

    return array;
}

const _equal = each(({ x }) => {
    return `if (${x[0].value} !== ${x[1].value}) return false;`;
}, 2).return(() => 'true');

const _set = map(({ x }) => `${x[0].value}`, 1);

const _i = Symbol('i');
const _data = Symbol('data');
const _flatten = each(
    ({ x, [_data]: data, [_i]: i }) => {
        return `${data}[${i}++] = ${x[0].value};`;
    },
    1,
    [_data, _i],
    { [_i]: 0 }
);

export function tensor<T>(x: Tensor<T>): Tensor<T>;
export function tensor<T>(x: ND<T>): Tensor<T>;
export function tensor<T>(x: any): Tensor<T>;
export function tensor<T = any>(x: any): Tensor<T> {
    if (x instanceof TensorLike) return x.forward();

    let shape: number[] = [];
    let _x = x;
    while (_x) {
        if (Array.isArray(_x.shape)) {
            shape.push(..._x.shape);
            break;
        }
        if (typeof _x != 'object' || _x.length == undefined) break;
        shape.push(_x.length);
        _x = _x[0];
    }

    return new Tensor(flatten(x, shape), shape);
}

function flatten(x: any, shape: number[]): any[] {
    let data = [x];
    for (let axis = 0, ndim = shape.length; axis < ndim; axis++) {
        let dim = shape[axis];
        let size = data.length;
        let _data = Array(size * dim);
        for (let i = 0, index = 0; i < size; i++) {
            let x = data[i];
            if (x instanceof TensorLike) x = x.forward().array();
            for (let j = 0; j < dim; j++) {
                _data[index++] = x[j];
            }
        }
        data = _data;
    }
    return data;
}

export class IndexedTensor<T> {
    base: TensorLike<T>;
    shape: number[];
    axes: number[] | null;
    indexDims: number;

    index: (Tensor<number> | undefined)[];
    indexShape: number[];
    _indices?: (number | Slice)[][];

    constructor(base: TensorLike<T>, index: AdvancedIndex[]) {
        let consecutive = true;
        let first = -1;
        let last = -1;
        let n = index.length;
        for (let i = 0; i < n; i++) {
            if (index[i]) {
                if (first == -1) first = i;
                if (last >= 0 && i != last) {
                    consecutive = false;
                    break;
                }
                last = i + 1;
            }
        }

        if (!consecutive) {
            let prepend: number[] = [];
            let rest: number[] = [];
            for (let i = 0, ndim = base.ndim; i < ndim; i++) {
                if (i < n && index[i]) {
                    prepend.push(i);
                } else rest.push(i);
            }
            first = 0;
            last = prepend.length;
            base = base.transpose(...prepend, ...rest);
            let _index: AdvancedIndex[] = Array(last);
            for (let i = 0; i < last; i++) {
                _index[i] = index[prepend[i]];
            }
            index = _index;
        }

        if (first == 0 && last - first == 1) {
            base = base.flatten();
            index = [(index[0] as Tensor).flatten()];
            n = 1;
        }
        let _index: (Tensor<number> | undefined)[] = Array(n);
        let shapes: number[][] = [];
        for (let i = 0; i < n; i++) {
            let ind = index[i];
            if (ind) {
                if (typeof ind.data[ind.offset] == 'boolean') {
                    ind = ind.nonzero().squeeze(-1);
                }
                _index[i] = ind as Tensor<number>;
                shapes.push(ind.shape);
            }
        }

        let shape = broadcastShape(shapes);

        this.index = _index;
        this.indexShape = shape;

        let indexDims = shape.length;
        this.indexDims = indexDims;

        shape = base.shape.toSpliced(first, last - first, ...shape);
        if (first > 0) {
            let ndim = shape.length;
            let axes: number[] = Array(ndim);
            for (let i = 0; i < ndim; i++) {
                axes[i] = i < indexDims ? first + i : i < indexDims + first ? i - indexDims : i;
            }
            this.axes = axes;
        } else {
            this.axes = null;
        }

        this.shape = shape;
        this.base = base;
    }

    get indices(): (number | Slice)[][] {
        if (this._indices) return this._indices;

        let shape = this.indexShape;
        let index = this.index;
        let n = index.length;

        let _index: AdvancedIndex[] = Array(n);
        for (let i = 0; i < n; i++) {
            _index[i] = index[i]?.expand(shape).ravel();
        }

        let size = 1;
        let ndim = this.indexDims;
        for (let axis = 0; axis < ndim; axis++) size *= shape[axis];

        let indices: (number | Slice)[][] = Array(size);
        let colon = slice();
        for (let i = 0; i < size; i++) {
            let index = Array(n);
            for (let j = 0; j < n; j++) {
                let ind = _index[j];
                index[j] = ind ? ind.data[ind.offset + i] : colon;
            }
            indices[i] = index;
        }
        this._indices = indices;
        return indices;
    }

    forward(): Tensor<T> {
        let base = this.base.forward();
        let slices: Tensor<T>[] = [];
        let indices = this.indices;
        for (let i = 0, n = indices.length; i < n; i++) {
            slices.push(base.at(...indices[i]));
        }

        let out = stack(slices).reshape(this.shape);
        out.set = (x: any): Tensor<T> => {
            this.set(tensor(x));
            return out;
        };
        return out;
    }

    set(x: Tensor): this {
        x = x.expand(this.shape);
        if (this.axes) x = x.transpose(this.axes);
        x = x.flatten(0, this.indexDims - 1);

        let base = this.base.forward();
        let indices = this.indices;
        for (let i = 0, n = indices.length; i < n; i++) {
            base.at(...indices[i]).set(x.at(i));
        }
        return this;
    }
}

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () =>
                tensor(
                    arange(2 * 3 * 4)
                        .reshape(2, 3, 4)
                        .at(1)
                        .transpose()
                        .array()
                ),
            () =>
                arange(2 * 3 * 4)
                    .reshape(2, 3, 4)
                    .at(1)
                    .transpose()
        )
        .equal(
            () =>
                arange(2 * 3 * 4)
                    .reshape(2, 3, 4)
                    .transpose()
                    .reshape(-1),
            () => tensor([0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23])
        )
        .equal(
            () =>
                arange(2 * 3 * 4)
                    .reshape(2, 3, 4)
                    .transpose()
                    .flatten(),
            () => tensor([0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23])
        )
        .equal(
            () => [...tensor([[1], [2], [3], [4]])],
            () => [tensor([1]), tensor([2]), tensor([3]), tensor([4])]
        )
        .equal(
            () => tensor([new Float32Array([1])]),
            () => tensor([[1]])
        )
        .equal(
            () => tensor([tensor([1])]),
            () => tensor([[1]])
        )
        .equal(
            () => {
                let x = tensor([1, 2, 3]);
                x.set(-1);
                return x;
            },
            () => tensor([-1, -1, -1])
        )
        .equal(
            () => tensor([1, 2, 3]).map(x => x ** 2),
            () => tensor([1, 4, 9])
        )
        .equal(
            () => tensor([1, 2, 3]).map(x => x ** 2),
            () => tensor([1, 4, 9])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).map(x => x * 2),
            () =>
                tensor([
                    [2, 4],
                    [6, 8],
                ])
        )
        .equal(
            () => tensor(5).map(x => x + 1),
            () => tensor(6)
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).flatten(),
            () => tensor([1, 2, 3, 4])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).transpose(),
            () =>
                tensor([
                    [1, 3],
                    [2, 4],
                ])
        )
        .equal(
            () => tensor(1).flatten(),
            () => tensor([1])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ])
                    .transpose()
                    .flatten(),
            () => tensor([1, 3, 2, 4])
        )
        .equal(
            () => tensor([1, 2, 3, 4]).reshape(2, 2),
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ])
        )
        .equal(
            () => tensor([1, 2, 3, 4, 5, 6]).reshape(2, -1),
            () =>
                tensor([
                    [1, 2, 3],
                    [4, 5, 6],
                ])
        )
        .equal(
            () => tensor([1, 2, 3]).at(1),
            () => tensor(2)
        )
        .equal(
            () => tensor([1, 2, 3]).at(-1),
            () => tensor(3)
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).at(0, 1),
            () => tensor(2)
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).at(1),
            () => tensor([3, 4])
        )
        .equal(
            () =>
                tensor([
                    [1, 2, 3],
                    [4, 5, 6],
                ]).at(':', 1),
            () => tensor([2, 5])
        )
        .equal(
            () =>
                tensor([
                    [1, 2, 3],
                    [4, 5, 6],
                ]).at('::2', 1),
            () => tensor([2])
        )
        .equal(
            () => {
                let t = tensor([1, 2, 3]);
                t.set(0);
                return t;
            },
            () => tensor([0, 0, 0])
        )
        .equal(
            () => {
                let t = tensor([1, 2, 3]);
                let other = tensor([4, 5, 6]);
                t.set(other);
                return t;
            },
            () => tensor([4, 5, 6])
        )
        .equal(
            () => {
                let t = tensor([
                    [1, 2],
                    [3, 4],
                ]);
                let x = tensor([10, 20]);
                t.set(x);
                return t;
            },
            () =>
                tensor([
                    [10, 20],
                    [10, 20],
                ])
        )
        .equal(
            () => tensor([1, 2, 3]).expand(2, 3),
            () =>
                tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                ])
        )
        .equal(
            () => {
                let t = tensor([1, 2, 3]);
                let c = t.copy();
                t.set(0);
                return c;
            },
            () => tensor([1, 2, 3])
        )
        .equal(
            () => tensor([]),
            () => tensor([])
        )
        .equal(
            () => tensor([[[1]]]).flatten(),
            () => tensor([1])
        )
        .equal(
            () => tensor(42).item(),
            () => 42
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).ravel(),
            () => tensor([1, 2, 3, 4])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ])
                    .transpose()
                    .ravel(),
            () => tensor([1, 3, 2, 4])
        )
        .equal(
            () => tensor([1, 2, 3]).equal(tensor([1, 2, 3])),
            () => true
        )
        .equal(
            () => tensor([1, 2, 3]).equal(tensor([1, 2, 4])),
            () => false
        )
        .equal(
            () => tensor([1, 2, 3]).equal(tensor([1, 2])),
            () => false
        )
        .equal(
            () => tensor([1, 2, 3]).unsqueeze(0),
            () => tensor([[1, 2, 3]])
        )
        .equal(
            () => tensor([1, 2, 3]).unsqueeze(1),
            () => tensor([[1], [2], [3]])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).unsqueeze(0),
            () =>
                tensor([
                    [
                        [1, 2],
                        [3, 4],
                    ],
                ])
        )
        .equal(
            () =>
                tensor([
                    [
                        [1, 2],
                        [3, 4],
                    ],
                ]).squeeze(),
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ])
        )
        .equal(
            () =>
                tensor([
                    [[1], [2]],
                    [[3], [4]],
                ]).squeeze(2),
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ])
        )
        .equal(
            () => tensor([0, 1, 2, 0, 3]).nonzero(),
            () => tensor([[1], [2], [4]])
        )
        .equal(
            () =>
                tensor([
                    [0, 1],
                    [2, 0],
                ]).nonzero(),
            () =>
                tensor([
                    [0, 1],
                    [1, 0],
                ])
        )
        .equal(
            () => tensor([1, 2, 3]).sum(),
            () => tensor(6)
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).sum(0),
            () => tensor([4, 6])
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).sum(1, true),
            () => tensor([[3], [7]])
        )
        .equal(
            () => [...tensor([1, 2, 3])],
            () => [tensor(1), tensor(2), tensor(3)]
        )
        .equal(
            () => [
                ...tensor([
                    [1, 2],
                    [3, 4],
                ]),
            ],
            () => [tensor([1, 2]), tensor([3, 4])]
        )
        .equal(
            () => {
                let x = arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5);
                new IndexedTensor(
                    x,
                    [
                        ,
                        [[[0], [1]]],
                        [
                            [[1], [3]],
                            [[1], [2]],
                            [[-1], [-2]],
                        ],
                    ].map(x => (x != null ? tensor(x) : x))
                ).set(tensor(1));
                return x;
            },
            () =>
                tensor([
                    [
                        [
                            [0, 1, 2, 3, 4],
                            [1, 1, 1, 1, 1],
                            [10, 11, 12, 13, 14],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [40, 41, 42, 43, 44],
                            [45, 46, 47, 48, 49],
                            [50, 51, 52, 53, 54],
                            [55, 56, 57, 58, 59],
                        ],
                    ],
                    [
                        [
                            [60, 61, 62, 63, 64],
                            [1, 1, 1, 1, 1],
                            [70, 71, 72, 73, 74],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [80, 81, 82, 83, 84],
                            [85, 86, 87, 88, 89],
                            [1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1],
                        ],
                        [
                            [100, 101, 102, 103, 104],
                            [105, 106, 107, 108, 109],
                            [110, 111, 112, 113, 114],
                            [115, 116, 117, 118, 119],
                        ],
                    ],
                ])
        )
        .equal(
            () => {
                let x = arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5);
                new IndexedTensor(
                    x,
                    [
                        ,
                        [[[0], [1]]],
                        ,
                        [
                            [[1], [3]],
                            [[1], [2]],
                            [[-1], [-2]],
                        ],
                    ].map(x => (x != null ? tensor(x) : x))
                ).set(tensor(1));
                return x;
            },
            () =>
                tensor([
                    [
                        [
                            [0, 1, 2, 3, 1],
                            [5, 1, 7, 8, 1],
                            [10, 1, 12, 13, 1],
                            [15, 1, 17, 18, 1],
                        ],
                        [
                            [20, 21, 1, 1, 24],
                            [25, 26, 1, 1, 29],
                            [30, 31, 1, 1, 34],
                            [35, 36, 1, 1, 39],
                        ],
                        [
                            [40, 41, 42, 43, 44],
                            [45, 46, 47, 48, 49],
                            [50, 51, 52, 53, 54],
                            [55, 56, 57, 58, 59],
                        ],
                    ],
                    [
                        [
                            [60, 1, 62, 63, 1],
                            [65, 1, 67, 68, 1],
                            [70, 1, 72, 73, 1],
                            [75, 1, 77, 78, 1],
                        ],
                        [
                            [80, 81, 1, 1, 84],
                            [85, 86, 1, 1, 89],
                            [90, 91, 1, 1, 94],
                            [95, 96, 1, 1, 99],
                        ],
                        [
                            [100, 101, 102, 103, 104],
                            [105, 106, 107, 108, 109],
                            [110, 111, 112, 113, 114],
                            [115, 116, 117, 118, 119],
                        ],
                    ],
                ])
        )
        .equal(
            () => {
                let x = arange(3 * 3 * 3).reshape(3, 3, 3);
                new IndexedTensor(
                    x,
                    [, [0, -1]].map(x => (x != null ? tensor(x) : x))
                ).set(
                    tensor([
                        [1, 2, 3],
                        [-1, -2, -3],
                    ])
                );
                return x;
            },
            () =>
                tensor([
                    [
                        [1, 2, 3],
                        [3, 4, 5],
                        [-1, -2, -3],
                    ],
                    [
                        [1, 2, 3],
                        [12, 13, 14],
                        [-1, -2, -3],
                    ],
                    [
                        [1, 2, 3],
                        [21, 22, 23],
                        [-1, -2, -3],
                    ],
                ])
        )
        .equal(
            () => tensor(5).equal(5),
            () => true
        )
        .equal(
            () => tensor(5).equal(tensor(6)),
            () => false
        )
        .equal(
            () => tensor([1, 2, 3]).equal(tensor([1, 2, 3])),
            () => true
        )
        .equal(
            () => tensor([1, 2, 3]).equal(tensor([1, 2, 4])),
            () => false
        )
        .equal(
            () => tensor([1, 2]).equal(tensor([1, 2, 3])),
            () => false
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).equal(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                ),
            () => true
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).equal(
                    tensor([
                        [1, 2],
                        [3, 5],
                    ])
                ),
            () => false
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).equal(
                    tensor([
                        [1, 2, 3],
                        [4, 5, 6],
                    ])
                ),
            () => false
        )
        .equal(
            () => tensor([]).equal(tensor([])),
            () => true
        )
        .equal(
            () => tensor([]).equal(tensor([1])),
            () => false
        )
        .equal(
            () => tensor(5).equal(tensor([5])),
            () => false
        )
        .equal(
            () =>
                tensor([
                    [1, 2],
                    [3, 4],
                ]).equal(tensor([1, 2, 3, 4])),
            () => false
        )
        .equal(
            () => {
                let t = tensor([1, 2, 3]);
                return t.equal(t);
            },
            () => true
        )
        .equal(
            () => {
                let t = tensor([
                    [1, 2],
                    [2, 1],
                ]);
                return t.equal(t.transpose());
            },
            () => true
        )
        .equal(
            () => {
                let t = tensor([
                    [1, 2],
                    [3, 4],
                ]);
                return t.equal(t.transpose());
            },
            () => false
        )
        .equal(
            () => {
                let t = arange(10);
                return t.at(t.gt(5));
            },
            () => tensor([6, 7, 8, 9])
        )
        .equal(
            () => {
                let t = arange(10).reshape(5, 2);
                return t.at(t.sum(-1).lt(10), [true, false]);
            },
            () => tensor([0, 2, 4])
        )
        .equal(
            () => {
                let t = arange(10).reshape(5, 2);
                return t.at(t.gt(5));
            },
            () => tensor([6, 7, 8, 9])
        );
