import { arange } from './arange.ts';
import { broadcast } from './broadcast.ts';
import { empty } from './empty.ts';
import { Tensor, tensor } from './tensor.ts';
import { _value, map, reduce } from './tensorfn.ts';
import { TestSuite } from './tester.ts';

export function add(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _add.for(x.ndim)(x, y, out);
    return out;
}

export function sub(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _sub.for(x.ndim)(x, y, out);
    return out;
}

export function mul(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _mul.for(x.ndim)(x, y, out);
    return out;
}

export function div(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _div.for(x.ndim)(x, y, out);
    return out;
}

export function pow(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _pow.for(x.ndim)(x, y, out);
    return out;
}

export function log(x: Tensor, out?: Tensor | null): Tensor {
    out ??= empty(x.shape);
    _log.for(x.ndim)(x, out);
    return out;
}

export function reciprocal(x: Tensor, out?: Tensor | null): Tensor {
    out ??= empty(x.shape);
    _reciprocal.for(x.ndim)(x, out);
    return out;
}

export function neg(x: Tensor, out?: Tensor | null): Tensor {
    out ??= empty(x.shape);
    _neg.for(x.ndim)(x, out);
    return out;
}

export function abs(x: Tensor, out?: Tensor | null): Tensor {
    out ??= empty(x.shape);
    _abs.for(x.ndim)(x, out);
    return out;
}

export function sign(x: Tensor, out?: Tensor | null): Tensor {
    out ??= empty(x.shape);
    _sign.for(x.ndim)(x, out);
    return out;
}

export function dot(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    out ??= empty();
    if (x.ndim != 1 || y.ndim != 1) throw new Error('non-1d tensors');
    if (x.size != y.size) throw new Error('inconsistent size');

    let sum = 0;
    for (let i = 0, dim = x.shape[0]; i < dim; i++) {
        sum += x.data[i * x.strides[0] + x.offset] * y.data[i * y.strides[0] + y.offset];
    }
    out.data[out.offset] = sum;
    return out;
}

export function sum(x: Tensor, axis?: number | number[] | null, keepdim = false, out?: Tensor | null): Tensor {
    let fn = _sum.for(x.ndim, axis);
    if (!out) {
        let shape = x.shape.slice();
        for (let axis of fn.axes) shape[axis] = 1;
        out = empty(shape);
    }
    fn(x, out, 0);
    if (!keepdim) out = out.squeeze(fn.axes);
    return out;
}

const _add = map(({ x }) => `${x[0].value} + ${x[1].value}`, 2);
const _sub = map(({ x }) => `${x[0].value} - ${x[1].value}`, 2);
const _mul = map(({ x }) => `${x[0].value} * ${x[1].value}`, 2);
const _div = map(({ x }) => `${x[0].value} / ${x[1].value}`, 2);

const _pow = map(({ x }) => `${x[0].value} ** ${x[1].value}`, 2);

const _log = map(({ x }) => `Math.log(${x[0].value})`, 1);
const _reciprocal = map(({ x }) => `1 / ${x[0].value}`, 1);
const _neg = map(({ x }) => `-${x[0].value}`, 1);
const _abs = map(({ x }) => `Math.abs(${x[0].value})`, 1);

const _sign = map(({ x }) => `${x[0].value} == 0 ? 0 : ${x[0].value} > 0 ? 1 : -1`, 1);

const _sum = reduce(({ x, [_value]: value }) => `${value} + ${x[0].value}`, 1);

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () => dot(tensor([2, 3]), tensor([2, 1])),
            () => tensor(7)
        )
        .equal(
            () => add(tensor([1, 2, 3]), tensor(2)),
            () => tensor([3, 4, 5])
        )
        .equal(
            () => tensor([1]).sum(),
            () => tensor(1)
        )
        .equal(
            () => tensor([1]).sum([], true),
            () => tensor([1])
        )
        .equal(
            () =>
                arange(2 * 3 * 4 * 5)
                    .reshape(2, 3, 4, 5)
                    .sum([0, -1]),
            () =>
                tensor([
                    [320, 370, 420, 470],
                    [520, 570, 620, 670],
                    [720, 770, 820, 870],
                ])
        );
