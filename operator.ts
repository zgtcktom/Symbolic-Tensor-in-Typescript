import { broadcast } from './broadcast.ts';
import { empty } from './empty.ts';
import { Tensor, tensor } from './tensor.ts';
import { map } from './tensorfn.ts';
import { TestSuite } from './tester.ts';

export function gt(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _gt.for(x.ndim)(x, y, out);
    return out;
}

export function ge(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _ge.for(x.ndim)(x, y, out);
    return out;
}

export function lt(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _lt.for(x.ndim)(x, y, out);
    return out;
}

export function le(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _le.for(x.ndim)(x, y, out);
    return out;
}

export function eq(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _eq.for(x.ndim)(x, y, out);
    return out;
}

export function ne(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _ne.for(x.ndim)(x, y, out);
    return out;
}

export function and(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _and.for(x.ndim)(x, y, out);
    return out;
}

export function or(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    [x, y] = broadcast(x, y);
    out ??= empty(x.shape);
    _or.for(x.ndim)(x, y, out);
    return out;
}

const _gt = map(({ x: [x, y] }) => `${x.value} > ${y.value}`, 2);

const _ge = map(({ x: [x, y] }) => `${x.value} >= ${y.value}`, 2);

const _lt = map(({ x: [x, y] }) => `${x.value} < ${y.value}`, 2);

const _le = map(({ x: [x, y] }) => `${x.value} <= ${y.value}`, 2);

const _eq = map(({ x: [x, y] }) => `${x.value} == ${y.value}`, 2);

const _ne = map(({ x: [x, y] }) => `${x.value} != ${y.value}`, 2);

const _and = map(({ x: [x, y] }) => `${x.value} && ${y.value}`, 2);

const _or = map(({ x: [x, y] }) => `${x.value} || ${y.value}`, 2);

export const test = (suite: TestSuite) =>
    suite.equal(
        () =>
            gt(
                tensor([
                    [1, 2],
                    [3, 4],
                ]),
                tensor([
                    [1, 1],
                    [4, 4],
                ])
            ),
        () =>
            tensor([
                [false, true],
                [false, false],
            ])
    );
