import { IndexedTensor, Tensor, tensor } from './tensor.ts';
import { broadcast, broadcastShape } from './broadcast.ts';
import { Slice } from './slice.ts';
import { _axis, _axes, type Index, normalizeIndex, shapeEqual, TensorLike, unwrap } from './tensorlike.ts';
import { zeros } from './empty.ts';
import { TestSuite } from './tester.ts';
import { randn } from './rand.ts';
import { arange } from './arange.ts';

function _shape(shape: number[], axes: number[], keepdim: boolean): number[] {
    let n = axes.length;
    if (keepdim) {
        shape = shape.slice();
        for (let i = 0; i < n; i++) {
            shape[axes[i]] = 1;
        }
        return shape;
    }
    let _shape: number[] = [];
    for (let i = 0, j = 0, ndim = shape.length; i < ndim; i++) {
        if (j < n && i == axes[j]) {
            j++;
        } else {
            _shape.push(shape[i]);
        }
    }
    return _shape;
}

export function symbolic(x: any): Symbolic {
    return x instanceof Symbolic ? x : new Constant(tensor(x));
}

export class Symbolic extends TensorLike<number> {
    forward(): Tensor<number> {
        throw new Error('unimplemented');
    }

    backward(grad?: TensorLike) {}

    copy(): Symbolic {
        return new Copy(this);
    }

    contiguous(): Symbolic {
        return new Contiguous(this);
    }

    flatten(start: number = 0, end: number = -1): Symbolic {
        return new Flatten(this, start, end);
    }

    ravel(): Symbolic {
        return new Ravel(this);
    }

    expand(...shape: number[] | [number[]]): Symbolic {
        shape = unwrap(shape);

        if (shapeEqual(shape, this.shape)) return this;

        return new Expand(this, shape);
    }

    reshape(...shape: number[] | [number[]]): Symbolic {
        shape = unwrap(shape);

        for (let i = 0; i < shape.length; i++) {
            if (shape[i] == -1) {
                let size = -1;
                for (let i = 0, ndim = shape.length; i < ndim; i++) {
                    size *= shape[i];
                }
                if (size < 0) throw new Error('only one -1 is allowed');
                shape[i] = this.size / size;
            }
        }

        if (shapeEqual(shape, this.shape)) return this;

        return new Reshape(this, shape);
    }

    transpose(...axes: number[] | [number[]]): Symbolic {
        let _axes: number[] | undefined;
        if (Array.isArray(axes[0])) _axes = axes[0];
        else if (axes.length > 0 && axes[0] != undefined) _axes = axes as number[];

        return new Transpose(this, _axes);
    }

    unsqueeze(axis: number): Symbolic {
        return new Unsqueeze(this, axis);
    }

    squeeze(axis?: number | number[]): Symbolic {
        return new Squeeze(this, axis);
    }

    at(...index: (Index | number[] | string)[]): Symbolic {
        return new At(this, normalizeIndex(index));
    }

    sum(axis?: number | number[], keepdim: boolean = false) {
        return new Sum(this, axis, keepdim);
    }

    add(x: any): Symbolic {
        return new Add(this, symbolic(x));
    }

    sub(x: any): Symbolic {
        return new Sub(this, symbolic(x));
    }

    mul(x: any): Symbolic {
        return new Mul(this, symbolic(x));
    }

    div(x: any): Symbolic {
        return new Div(this, symbolic(x));
    }

    reciprocal(): Symbolic {
        return new Reciprocal(this);
    }

    neg(): Symbolic {
        return new Neg(this);
    }

    pow(x: any): Symbolic {
        return new Pow(this, symbolic(x));
    }

    log(): Symbolic {
        return new Log(this);
    }

    abs(): Symbolic {
        return new Abs(this);
    }

    sign(): Symbolic {
        return new Sign(this);
    }

    dot(x: any): Symbolic {
        return new Dot(this, symbolic(x));
    }

    matmul(x: any): Symbolic {
        return new Matmul(this, symbolic(x));
    }

    *[Symbol.iterator](): Iterator<Symbolic> {
        for (let i = 0; i < this.shape[0]; i++) {
            yield this.at(i);
        }
    }
}

export class Constant extends Symbolic {
    value: Tensor;
    constructor(value: Tensor) {
        super(value.shape);
        this.value = value;
    }

    forward(): Tensor {
        return this.value;
    }

    backward(grad?: TensorLike) {}
}

export class Parameter extends Symbolic {
    value: Tensor;
    grad?: TensorLike;
    constructor(value: Tensor) {
        super(value.shape);
        this.value = value;
    }

    forward(): Tensor {
        return this.value;
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.grad = this.grad ? this.grad.add(grad) : grad;
    }
}

export class At extends Symbolic {
    x: Symbolic;
    index: Index[];
    base: Tensor;
    scatter: Tensor;
    constructor(x: Symbolic, index: Index[]) {
        let shape: number[] = [];

        let advanced = Array(x.ndim);
        let basicOnly = true;

        let axis = 0;
        for (let ind of index) {
            if (ind instanceof Tensor && ind.ndim == 0) ind = ind.item() as number;

            if (ind == null) {
                shape.push(1);
                continue;
            }

            if (typeof ind == 'number') {
                let dim = x.shape[axis];
                if (ind < -dim || ind >= dim) {
                    throw new Error(`index ${ind} out of bounds at dimension ${axis}`);
                }
                if (ind < 0) ind += dim;
                axis += 1;
                continue;
            }

            if (ind instanceof Slice) {
                let { length } = ind.indices(x.shape[axis]);
                shape.push(length);
                axis += 1;
                continue;
            }

            advanced[axis] = ind;
            shape.push(x.shape[axis]);
            axis++;
            basicOnly = false;
        }

        for (; axis < x.ndim; axis++) {
            shape.push(x.shape[axis]);
        }

        if (!basicOnly) {
            shape = new IndexedTensor(new Symbolic(shape), advanced).shape;
        }

        super(shape);
        this.x = x;
        this.index = index;
        this.base = zeros(this.x.shape);
        this.scatter = this.base.at(...this.index);
    }

    forward(): Tensor {
        return this.x.forward().at(...this.index);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.scatter.set(grad);
        this.x.backward(this.base);
    }
}

export class Copy extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().copy();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad);
    }
}

export class Contiguous extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().contiguous();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad);
    }
}

export class Flatten extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic, start: number, end: number) {
        let { size, ndim } = x;
        start = _axis(start, ndim);
        end = _axis(end, ndim);
        let shape: number[];
        if (start == 0 && end == ndim - 1) {
            shape = [size];
        } else {
            shape = Array(ndim - (end - start));
            let j = 0;
            for (let i = 0; i < start; i++) {
                shape[j++] = x.shape[i];
            }
            let size = 1;
            for (let i = start; i <= end; i++) {
                size *= x.shape[i];
            }
            shape[j++] = size;
            for (let i = end + 1; i < ndim; i++) {
                shape[j++] = x.shape[i];
            }
        }
        super(shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().flatten();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        grad = grad.reshape(this.x.shape);
        this.x.backward(grad);
    }
}

export class Ravel extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        let size = 1;
        for (let axis = 0; axis < x.shape.length; axis++) {
            size *= x.shape[axis];
        }
        super([size]);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().ravel();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        grad = grad.reshape(this.x.shape);
        this.x.backward(grad);
    }
}

export class Expand extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic, shape: number[]) {
        super(shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().expand(this.shape);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);

        let pad = this.shape.length - this.x.shape.length;
        let axes: number[] = [];
        for (let axis = 0; axis < this.shape.length; axis++) {
            if ((axis < pad || this.x.shape[axis - pad] == 1) && this.shape[axis] > 1) {
                axes.push(axis);
            }
        }

        grad = grad.sum(axes, true);

        if (pad > 0) {
            let axes: number[] = Array(pad);
            for (let axis = 0; axis < pad; axis++) {
                axes[axis] = axis;
            }
            grad = grad.squeeze(axes);
        }

        this.x.backward(grad);
    }
}

export class Reshape extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic, shape: number[]) {
        super(shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().reshape(this.shape);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        grad = grad.reshape(this.x.shape);
        this.x.backward(grad);
    }
}

export class Transpose extends Symbolic {
    x: Symbolic;
    axes: number[];

    constructor(x: Symbolic, axes?: number[] | null) {
        let shape: number[];
        if (axes == null) {
            shape = x.shape.toReversed();
        } else {
            axes = _axes(axes, x.ndim);
            if (axes.length != x.ndim) {
                throw new Error('axes must be a permutation of [0, 1, ..., ndim-1]');
            }
            shape = Array(x.ndim);
            for (let axis = 0; axis < x.ndim; axis++) {
                shape[axis] = x.shape[axes[axis]];
            }
        }
        super(shape);
        this.x = x;
        this.axes = axes ?? [];
    }

    forward(): Tensor {
        return this.x.forward().transpose(this.axes);
    }

    backward(grad: TensorLike = tensor(1)) {
        if (this.axes == null || this.axes.length == 0) {
            grad = grad.transpose();
        } else {
            let axes = new Array(this.ndim);
            for (let axis = 0; axis < this.ndim; axis++) {
                axes[this.axes[axis]] = axis;
            }
            grad = grad.transpose(axes);
        }
        this.x.backward(grad);
    }
}

export class Squeeze extends Symbolic {
    x: Symbolic;
    axis: number[];
    constructor(x: Symbolic, axis?: number | number[]) {
        axis = _axes(axis, x.shape.length);
        let shape: number[] = [];
        for (let i = 0, j = 0; i < x.shape.length; i++) {
            if (j < axis.length && i == axis[j] && x.shape[i] == 1) {
                j++;
            } else {
                shape.push(x.shape[i]);
            }
        }
        super(shape);
        this.x = x;
        this.axis = axis;
    }

    forward(): Tensor {
        return this.x.forward().squeeze(this.axis);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        grad = grad.reshape(this.x.shape);
        this.x.backward(grad);
    }
}

export class Unsqueeze extends Symbolic {
    x: Symbolic;
    axis: number;
    constructor(x: Symbolic, axis: number) {
        axis = _axis(axis, x.shape.length + 1);
        let shape = x.shape.toSpliced(axis, 0, 1);
        super(shape);
        this.x = x;
        this.axis = axis;
    }

    forward(): Tensor {
        return this.x.forward().unsqueeze(this.axis);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        grad = grad.squeeze(this.axis);
        this.x.backward(grad);
    }
}

export class Sum extends Symbolic {
    x: Symbolic;
    axis: number[];
    keepdim: boolean;
    constructor(x: Symbolic, axis?: number | number[], keepdim: boolean = false) {
        axis = _axes(axis, x.ndim);
        let shape = _shape(x.shape, axis, keepdim);
        super(shape);
        this.x = x;
        this.axis = axis;
        this.keepdim = keepdim;
    }

    forward(): Tensor {
        return this.x.forward().sum(this.axis, this.keepdim);
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad);
    }
}

export class Add extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        [x, y] = broadcast(x, y);
        super(x.shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().add(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad);
        this.y.backward(grad);
    }
}

export class Sub extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        [x, y] = broadcast(x, y);
        super(x.shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().sub(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad);
        this.y.backward(grad.neg());
    }
}

export class Mul extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        [x, y] = broadcast(x, y);
        super(x.shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().mul(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.y));
        this.y.backward(grad.mul(this.x));
    }
}

export class Div extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        [x, y] = broadcast(x, y);
        super(x.shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().div(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.y.reciprocal()));
        this.y.backward(grad.mul(this.x.div(this.y.mul(this.y)).neg()));
    }
}

export class Reciprocal extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().reciprocal();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.x.mul(this.x).reciprocal().neg()));
    }
}

export class Neg extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().neg();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.neg());
    }
}

export class Pow extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        [x, y] = broadcast(x, y);
        super(x.shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().pow(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.y).mul(this.x.pow(this.y.sub(tensor(1)))));
        this.y.backward(grad.mul(this).mul(this.x.log()));
    }
}

export class Log extends Symbolic {
    x: Symbolic;
    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor {
        return this.x.forward().log();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.x.reciprocal()));
    }
}

export class Abs extends Symbolic {
    x: Symbolic;

    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor<number> {
        return this.x.forward().abs();
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.x.sign()));
    }
}

export class Sign extends Symbolic {
    x: Symbolic;

    constructor(x: Symbolic) {
        super(x.shape);
        this.x = x;
    }

    forward(): Tensor<number> {
        return this.x.forward().sign();
    }

    backward(grad?: TensorLike) {}
}

export class Dot extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        super([]);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().dot(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);
        this.x.backward(grad.mul(this.y));
        this.y.backward(grad.mul(this.x));
    }
}

export class Matmul extends Symbolic {
    x: Symbolic;
    y: Symbolic;
    constructor(x: Symbolic, y: Symbolic) {
        let shape: number[];

        if (x.ndim === 1 && y.ndim === 1) {
            if (x.shape[0] !== y.shape[0]) {
                throw new Error('Incompatible shapes for dot product');
            }
            shape = [];
        } else if (x.ndim === 1) {
            if (y.shape[y.ndim - 2] !== x.shape[0]) {
                throw new Error('Incompatible shapes for matrix multiplication');
            }
            shape = y.shape.slice(0, -2).concat([y.shape[y.ndim - 1]]);
        } else if (y.ndim === 1) {
            if (x.shape[x.ndim - 1] !== y.shape[0]) {
                throw new Error('Incompatible shapes for matrix multiplication');
            }
            shape = x.shape.slice(0, -1);
        } else {
            const xbatch = x.shape.slice(0, -2);
            const ybatch = y.shape.slice(0, -2);
            const batch = broadcastShape([xbatch, ybatch]);
            const m = x.shape[x.ndim - 2];
            const n = x.shape[x.ndim - 1];
            const p = y.shape[y.ndim - 2];
            const q = y.shape[y.ndim - 1];
            if (n !== p) {
                throw new Error('Incompatible shapes for matrix multiplication');
            }
            shape = batch.concat([m, q]);
        }
        super(shape);
        this.x = x;
        this.y = y;
    }

    forward(): Tensor {
        return this.x.forward().matmul(this.y.forward());
    }

    backward(grad: TensorLike = tensor(1)) {
        grad = grad.expand(this.shape);

        const x = this.x;
        const y = this.y;

        const x_ndim = x.ndim;
        const y_ndim = y.ndim;

        if (x_ndim === 1 && y_ndim === 1) {
            this.x.backward(grad.mul(y));
            this.y.backward(grad.mul(x));
            return;
        }

        if (x_ndim === 2 && y_ndim === 2) {
            this.x.backward(grad.matmul(y.transpose()));
            this.y.backward(grad.transpose().matmul(x).transpose());
            return;
        }

        if (x_ndim === 1) {
            let grad2 = grad.unsqueeze(0);
            let x2 = x.unsqueeze(0);

            this.x.backward(grad2.matmul(y.transpose()).squeeze(0));
            this.y.backward(grad2.transpose().matmul(x2).transpose().squeeze(0));
            return;
        }

        if (y_ndim === 1) {
            let grad2 = grad.unsqueeze(-1);
            let y2 = y.unsqueeze(-1);
            let x2 = x;
            this.x.backward(grad2.matmul(y2.transpose()).squeeze(-1));
            this.y.backward(grad2.transpose().matmul(x2).transpose().squeeze(-1));
            return;
        }

        let x_t_axes = [...Array(x.ndim).keys()];
        let y_t_axes = [...Array(y.ndim).keys()];

        [x_t_axes[x.ndim - 2], x_t_axes[x.ndim - 1]] = [x_t_axes[x.ndim - 1], x_t_axes[x.ndim - 2]];
        [y_t_axes[y.ndim - 2], y_t_axes[y.ndim - 1]] = [y_t_axes[y.ndim - 1], y_t_axes[y.ndim - 2]];

        this.x.backward(grad.matmul(y.transpose(y_t_axes)));
        this.y.backward(grad.transpose().matmul(x).transpose(x_t_axes));
    }
}

export const test = (suite: TestSuite) =>
    suite
        // Test Case 1: Addition
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let y = new Parameter(tensor([4, 5, 6]));
                let z = x.add(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [tensor([5, 7, 9]), tensor([1, 1, 1]), tensor([1, 1, 1])]
        )
        // Test Case 2: Subtraction
        .equal(
            () => {
                let x = new Parameter(tensor([4, 5, 6]));
                let y = new Parameter(tensor([1, 2, 3]));
                let z = x.sub(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [tensor([3, 3, 3]), tensor([1, 1, 1]), tensor([-1, -1, -1])]
        )
        // Test Case 3: Multiplication
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let y = new Parameter(tensor([4, 5, 6]));
                let z = x.mul(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [tensor([4, 10, 18]), tensor([4, 5, 6]), tensor([1, 2, 3])]
        )
        // Test Case 4: Division
        .equal(
            () => {
                let x = new Parameter(tensor([4, 5, 6]));
                let y = new Parameter(tensor([2, 2, 2]));
                let z = x.div(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [tensor([2, 2.5, 3]), tensor([0.5, 0.5, 0.5]), tensor([-1, -1.25, -1.5])]
        )
        // Test Case 5: Negation
        .equal(
            () => {
                let x = new Parameter(tensor([1, -2, 3]));
                let z = x.neg();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor([-1, 2, -3]), tensor([-1, -1, -1])]
        )
        // Test Case 6: Reciprocal
        .equal(
            () => {
                let x = new Parameter(tensor([2, 4, 5]));
                let z = x.reciprocal();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor([0.5, 0.25, 0.2]), tensor([-0.25, -0.0625, -0.04])]
        )
        // Test Case 7: Power
        .equal(
            () => {
                let x = new Parameter(tensor([2, 3, 4]));
                let y = new Parameter(tensor([3, 2, 1]));
                let z = x.pow(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor([8, 9, 4]), tensor([12, 6, 1])]
        )
        // Test Case 8: Logarithm
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let z = x.log();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor([0, Math.log(2), Math.log(3)]), tensor([1, 0.5, 0.3333333333333333])]
        )
        // Test Case 9: Dot Product
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let y = new Parameter(tensor([4, 5, 6]));
                let z = x.dot(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [tensor(32), tensor([4, 5, 6]), tensor([1, 2, 3])]
        )
        // Test Case 10: Matrix Multiplication
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let y = new Parameter(
                    tensor([
                        [5, 6],
                        [7, 8],
                    ])
                );
                let z = x.matmul(y);
                let result = z.forward();
                z.backward();
                return [result, x.grad, y.grad];
            },
            () => [
                tensor([
                    [19, 22],
                    [43, 50],
                ]),
                tensor([
                    [11, 15],
                    [11, 15],
                ]),
                tensor([
                    [4, 4],
                    [6, 6],
                ]),
            ]
        )
        // Test Case 11: Expand
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let z = x.expand([2, 3]);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([
                    [1, 2, 3],
                    [1, 2, 3],
                ]),
                tensor([2, 2, 2]),
            ]
        )
        // Test Case 12: Reshape
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3, 4]));
                let z = x.reshape([2, 2]);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([
                    [1, 2],
                    [3, 4],
                ]),
                tensor([1, 1, 1, 1]),
            ]
        )
        // Test Case 13: Transpose
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.transpose();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([
                    [1, 3],
                    [2, 4],
                ]),
                tensor([
                    [1, 1],
                    [1, 1],
                ]),
            ]
        )
        // Test Case 14: Squeeze
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [[1], [2]],
                        [[3], [4]],
                    ])
                );
                let z = x.squeeze(2);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([
                    [1, 2],
                    [3, 4],
                ]),
                tensor([
                    [[1], [1]],
                    [[1], [1]],
                ]),
            ]
        )
        // Test Case 15: Unsqueeze
        .equal(
            () => {
                let x = new Parameter(tensor([1, 2, 3]));
                let z = x.unsqueeze(0);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor([[1, 2, 3]]), tensor([1, 1, 1])]
        )
        // Test Case 16: Flatten
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.flatten();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([1, 2, 3, 4]),
                tensor([
                    [1, 1],
                    [1, 1],
                ]),
            ]
        )
        // Test Case 17: Ravel
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.ravel();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([1, 2, 3, 4]),
                tensor([
                    [1, 1],
                    [1, 1],
                ]),
            ]
        )
        // Test Case 18: Indexing (at)
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2, 3],
                        [4, 5, 6],
                    ])
                );
                let z = x.at(0, 1);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor(2),
                tensor([
                    [0, 1, 0],
                    [0, 0, 0],
                ]),
            ]
        )
        // Test Case 19: Sum with Axis
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.sum(0);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([4, 6]),
                tensor([
                    [1, 1],
                    [1, 1],
                ]),
            ]
        )
        // Test Case 20: Sum with Keepdim
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.sum(1, true);
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [
                tensor([[3], [7]]),
                tensor([
                    [1, 1],
                    [1, 1],
                ]),
            ]
        )
        // Test Case 21: Second-Order Derivative
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ])
                );
                let z = x.pow(tensor(3));
                z.backward(new Constant(tensor(1)));
                let grad = x.grad as Symbolic;
                x.grad = undefined;
                grad.backward();
                if (!x.grad) throw new Error('Gradient not set');
                return [z.forward(), grad.forward(), x.grad as Tensor];
            },
            () => [
                tensor([
                    [1, 8],
                    [27, 64],
                ]),
                tensor([
                    [3, 12],
                    [27, 48],
                ]),
                tensor([
                    [6, 12],
                    [18, 24],
                ]),
            ]
        )
        .equal(
            () => {
                let x = new Parameter(tensor([]));
                let z = x.sum();
                let result = z.forward();
                z.backward();
                return [result, x.grad];
            },
            () => [tensor(0), tensor([])]
        )
        .equal(
            () => {
                let x = new Parameter(tensor([1, -2, 0]));
                let z = x.abs();
                let result = z.forward();
                z.backward(tensor([1, 1, 1]));
                return [result, x.grad];
            },
            () => [tensor([1, 2, 0]), tensor([1, -1, 0])]
        )
        .equal(
            () => {
                let x = new Parameter(tensor(4));
                let z = x.abs();
                let result = z.forward();
                z.backward(tensor(1));
                return [result, x.grad];
            },
            () => [tensor(4), tensor(1)]
        )
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, -1],
                        [0, -3],
                    ])
                );
                let z = x.abs();
                let result = z.forward();
                z.backward(
                    tensor([
                        [1, 1],
                        [1, 1],
                    ])
                );
                return [result, x.grad];
            },
            () => [
                tensor([
                    [1, 1],
                    [0, 3],
                ]),
                tensor([
                    [1, -1],
                    [0, -1],
                ]),
            ]
        )
        .equal(
            () => {
                let x = new Parameter(tensor([1, -2, 3]));
                let z = x.abs();
                let result = z.forward();
                z.backward(tensor(2));
                return [result, x.grad];
            },
            () => [tensor([1, 2, 3]), tensor([2, -2, 2])]
        )
        .equal(
            () => {
                let x = new Parameter(tensor([1, -2, 3]));
                let z = x.abs().mul(tensor(2));
                let result = z.forward();
                z.backward(tensor([1, 1, 1]));
                return [result, x.grad];
            },
            () => [tensor([2, 4, 6]), tensor([2, -2, 2])]
        )
        .equal(
            () => {
                let x = new Parameter(
                    tensor([
                        [1, -2],
                        [0, 3],
                    ])
                );
                let z = x.abs();
                z.backward(new Constant(tensor(1)));
                let grad = x.grad as Symbolic;
                x.grad = undefined;
                grad.backward();
                x.mul(0).backward();
                if (!x.grad) throw new Error('Gradient not set');
                return [z.forward(), grad.forward(), x.grad as Tensor];
            },
            () => [
                tensor([
                    [1, 2],
                    [0, 3],
                ]),
                tensor([
                    [1, -1],
                    [0, 1],
                ]),
                tensor([
                    [0, 0],
                    [0, 0],
                ]),
            ]
        );

if (0) {
    // Define ReLU activation using existing operations
    function relu(x: Symbolic): Symbolic {
        return x.add(x.abs()).div(2);
    }

    // Define mean function for MSE loss
    function mean(x: Symbolic): Symbolic {
        return x.sum().div(x.size);
    }

    // Generate training data
    let x_data = arange(-Math.PI, Math.PI, (Math.PI - -Math.PI) / 100).unsqueeze(1);
    let y_data = x_data.map(Math.sin);

    // Define symbolic input and output
    let x = new Constant(x_data);
    let y = new Constant(y_data);

    // Define network parameters
    const hidden_dim = 8;
    let W1 = new Parameter(randn(1, hidden_dim));
    let b1 = new Parameter(zeros(hidden_dim));
    let W2 = new Parameter(randn(hidden_dim, 1));
    let b2 = new Parameter(zeros(1));

    // Define the two-layer DNN: xW + b
    let hidden = x.matmul(W1).add(b1);
    let hidden_activated = relu(hidden);
    let output = hidden_activated.matmul(W2).add(b2);

    // Define loss (Mean Squared Error)
    let loss = mean(output.sub(y).pow(2));

    // Training loop
    const learning_rate = 0.05;
    const epochs = 100;
    for (let epoch = 0; epoch <= epochs; epoch++) {
        // Forward pass to compute loss
        let current_loss = loss.forward();

        // Backward pass to compute gradients
        loss.backward();

        // Update parameters using gradient descent
        for (let param of [W1, b1, W2, b2]) {
            param.value.sub(param.grad?.mul(learning_rate) ?? 0, param.value);
            delete param.grad;
        }

        // Report loss every 10 epochs
        if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}, Loss: ${current_loss}`);
        }
    }

    {
        let x = new Parameter(tensor([1, 2, 3])); // Shape [3]
        let loss = x.mul(2); // Shape [3], e.g., [2, 4, 6]
        loss.backward(); // No gradient provided
        console.log(x.grad); // Check if gradients are computed
    }

    {
        let W = new Parameter(tensor([1, 2, 3])); // Shape [3]
        let x = new Constant(tensor([4, 5, 6])); // Shape [3]
        let output = W.mul(x); // Shape [3]
        let loss = output.sum().div(3).expand([1]).sum(); // Shape [1]
        loss.backward();
        console.log(W.grad); // Should be [4/3, 5/3, 6/3], but if zero, the bug is here
    }
}
