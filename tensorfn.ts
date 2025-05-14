import { tensor } from './tensor.ts';
import { empty } from './empty.ts';
import { arange } from './arange.ts';
import { _axes } from './tensorlike.ts';

export class TensorFn {
    render: RenderFn;
    nx: number;
    args: symbol[];
    bindings?: Record<symbol, any>;
    vars: Record<symbol, string>;
    cache: Map<string, BoundTensorFn>;
    returnOp?: (scope: Scope) => string;
    constructor(render: RenderFn, nx: number, args: symbol[] = [], bindings?: Record<symbol, any>) {
        this.render = render;
        this.nx = nx;
        this.args = args;
        this.bindings = bindings;

        this.vars = {};
        this.cache = new Map();

        this.var(...args);
    }

    for(ndim: number, axis?: number | number[] | null): BoundTensorFn {
        let axes = _axes(axis, ndim);

        let key = `${ndim}:${axes.join(',')}`;
        if (this.cache.has(key)) {
            return this.cache.get(key)!;
        }

        let { nx, args, bindings, vars } = this;

        let scope = Object.assign(getScope(ndim, nx), vars);

        let _args = [...scope.x.map(x => x.name), ...args.map(key => vars[key])];
        let fn = new Function(..._args, this.body(ndim, axes, scope));
        if (bindings) {
            let _bindings: { [key: string]: any } = {};
            for (let key of Object.getOwnPropertySymbols(bindings)) {
                _bindings[vars[key]] = bindings[key];
            }
            fn = partial(fn, _args, _bindings);
        }

        let bound = fn as BoundTensorFn;
        bound.axes = axes;
        this.cache.set(key, bound);
        return bound;
    }

    body(ndim: number, axes: number[], scope: Scope = Object.assign(getScope(ndim, this.nx), this.vars)): string {
        let { nx } = this;

        let { i, x, shape } = scope;
        let indices = reorder(axes, ndim);

        let body = '';
        for (let n = ndim - 1; n >= 0; n--) {
            let axis = indices[n];

            body = this.render(body, scope, n, ndim, axes);
            body = `for (let ${i[axis]} = 0; ${i[axis]} < ${shape[axis]}; ${i[axis]}++) {` + body;
            for (let j = 0; j < nx; j++) {
                body += `${x[j].offset} += ${x[j].strides[axis]};`;
            }
            body += `}`;
            for (let j = 0; j < nx; j++) {
                body += `${x[j].offset} -= ${x[j].strides[axis]} * ${shape[axis]};`;
            }
        }
        body = init(scope) + this.render(body, scope, -1, ndim, axes);
        body += `return ${this.returnOp?.(scope)};`;

        return body;
    }

    var(...keys: symbol[]): this {
        let { vars } = this;
        for (let i = 0, n = keys.length; i < n; i++) {
            let key = keys[i];
            let baseName = `_${key.description ?? 'arg'}`;
            let name = baseName;
            let suffix = 0;
            while (Object.values(vars).includes(name)) {
                name = `${baseName}${++suffix}`;
            }
            vars[key] = name;
        }
        return this;
    }

    return(op: (scope: Scope) => string): this {
        this.returnOp = op;
        return this;
    }
}

interface BoundTensorFn extends Function {
    axes: number[];
}

interface RenderFn {
    (body: string, scope: Scope, n: number, ndim: number, axes: number[]): string;
}

interface Scope {
    x: TensorVariable[];
    i: string[];
    shape: string[];
    [key: symbol]: string;
}

function reorder(axes: number[], ndim: number): number[] {
    let indices: number[] = Array(ndim);
    let n = axes.length;
    if (n == ndim) {
        for (let i = 0; i < ndim; i++) indices[i] = axes[i];
    } else {
        let pos = 0;
        for (let i = 0, j = 0; i < ndim; i++) {
            if (j < n && i == axes[j]) j++;
            else indices[pos++] = i;
        }

        for (let i = 0; i < n; i++) indices[pos++] = axes[i];
    }
    return indices;
}

function partial(fn: Function, args: string[], bindings: { [key: string]: any }): Function {
    let unbound: string[] = [];
    let bound: string[] = [];
    let values: any[] = [];
    for (let i = 0, n = args.length; i < n; i++) {
        let arg = args[i];
        if (Object.hasOwn(bindings, arg)) {
            bound.push(arg);
            values.push(bindings[arg]);
        } else {
            unbound.push(arg);
        }
    }

    let fnArg = 'fn';
    while (args.includes(fnArg)) fnArg += '_';

    args = ['this', ...args];
    let body = `return function(${unbound.join(',')}){return ${fnArg}.call(${args.join(',')})}`;
    let boundfn = new Function(fnArg, ...bound, body)(fn, ...values);
    boundfn.toString = () => {
        return `(function(${bound.join(',')}){return function(${unbound.join(',')}){return (${fn.toString()}).call(${args.join(',')})}})(${values.join(',')})`;
    };
    return boundfn;
}

class TensorVariable {
    name: string;
    data: string;
    strides: string[];
    offset: string;
    value: string;
    constructor(name: string, data: string, strides: string[], offset: string) {
        this.name = name;
        this.data = data;
        this.strides = strides;
        this.offset = offset;
        this.value = `${data}[${offset}]`;
    }

    toString() {
        return this.name;
    }
}

function getScope(ndim: number, nx: number): Scope {
    let x = Array(nx);
    for (let n = 0; n < nx; n++) {
        let strides = Array(ndim);
        for (let m = 0; m < ndim; m++) {
            strides[m] = `x_${n}_strides_${m}`;
        }
        x[n] = new TensorVariable(`x_${n}`, `x_${n}_data`, strides, `x_${n}_offset`);
    }

    let i = Array(ndim);
    let shape = Array(ndim);
    for (let n = 0; n < ndim; n++) {
        i[n] = `i_${n}`;
        shape[n] = `shape_${n}`;
    }

    return { x, i, shape };
}

function init({ x, shape }: Scope): string {
    let body = `let [${shape.join(', ')}] = ${x[0]}.shape;`;
    for (let n = 0, len = x.length; n < len; n++) {
        body += `let [${x[n].strides.join(', ')}] = ${x[n]}.strides;`;
        body += `let ${x[n].offset} = ${x[n]}.offset;`;
        body += `let ${x[n].data} = ${x[n]}.data;`;
    }
    return body;
}

export const _fn = Symbol('fn');
export const _value = Symbol('value');
export const _initial = Symbol('initial');

export function each(op: (scope: Scope) => string, nx: number, args: symbol[] = [], bindings?: { [key: symbol]: any }): TensorFn {
    return new TensorFn(
        (body, scope, i, ndim) => {
            if (i == ndim - 1) return `${op(scope)};${body}`;
            return body;
        },
        nx,
        args,
        bindings
    );
}

export function eachfn(fn: Function, nx: number): TensorFn {
    return each(
        ({ x, [_fn]: fn }) => {
            return `${fn}(${x.map(x => `${x.data}[${x.offset}]`)})`;
        },
        nx,
        [_fn],
        { [_fn]: fn }
    );
}

export function map(op: (scope: Scope) => string, nx: number, args: symbol[] = [], bindings?: { [key: symbol]: any }): TensorFn {
    return each(
        scope => {
            let { x } = scope;
            return `${x[nx].data}[${x[nx].offset}] = ${op({ ...scope, x: x.slice(0, -1) })}`;
        },
        nx + 1,
        args,
        bindings
    );
}

export function mapfn(fn: Function, nx: number): TensorFn {
    return map(
        ({ x, [_fn]: fn }) => {
            return `${fn}(${x.map(x => `${x.data}[${x.offset}]`)})`;
        },
        nx,
        [_fn],
        { [_fn]: fn }
    );
}

export function reduce(op: (scope: Scope) => string, nx: number, args: symbol[] = [], bindings?: { [key: symbol]: any }): TensorFn {
    return new TensorFn(
        (body, scope, i, ndim, axes) => {
            let { x, [_value]: value, [_initial]: initial } = scope;
            if (i == ndim - axes.length - 1) {
                return `let ${value} = ${initial};` + body + `;${x[nx].data}[${x[nx].offset}] = ${value};`;
            }
            if (i == ndim - 1) {
                return `${value} = ${op({ ...scope, x: x.slice(0, -1) })};` + body;
            }
            return body;
        },
        nx + 1,
        [_initial, ...args],
        bindings
    ).var(_value);
}

export function reducefn(fn: Function, nx: number): TensorFn {
    return reduce(
        ({ x, [_value]: value, [_fn]: fn }) => {
            return `${fn}(${[value, ...x.map(x => `${x.data}[${x.offset}]`)]})`;
        },
        nx,
        [_fn],
        { [_fn]: fn }
    );
}

export function accum(op: (scope: Scope) => string, nx: number, args: symbol[] = [], bindings?: { [key: symbol]: any }): TensorFn {
    return new TensorFn(
        (body, scope, i, ndim, axes) => {
            let { x, [_value]: value, [_initial]: initial } = scope;
            if (i == ndim - axes.length - 1) {
                return `let ${value} = ${initial};` + body;
            }
            if (i == ndim - 1) {
                return `${value} = ${op({ ...scope, x: x.slice(0, -1) })};${x[nx].data}[${x[nx].offset}] = ${value};` + body;
            }
            return body;
        },
        nx + 1,
        [_initial, ...args],
        bindings
    ).var(_value);
}

export function accumfn(fn: Function, nx: number): TensorFn {
    return accum(
        ({ x, [_value]: value, [_fn]: fn }) => {
            return `${fn}(${[value, ...x.map(x => `${x.data}[${x.offset}]`)]})`;
        },
        nx,
        [_fn],
        { [_fn]: fn }
    );
}

export const test = (suite: any) =>
    suite
        .equal(
            () => {
                let x = arange(4 * 3).reshape(4, 3);
                let out = empty(3).reshape(1, 3);
                reducefn((a: any, c: any) => a + c, 1).for(x.ndim, 0)(x, out, 0);
                return out;
            },
            () => tensor([[18, 22, 26]])
        )
        .equal(
            () => {
                let x = arange(4 * 3).reshape(4, 3);
                let out = empty(4).reshape(4, 1);
                reducefn((a: any, c: any) => a + c, 1).for(x.ndim, -1)(x, out, 0);
                return out;
            },
            () => tensor([[3], [12], [21], [30]])
        )
        .equal(
            () => {
                let x = arange(4 * 3).reshape(4, 3);
                let out = empty(1).reshape(1, 1);
                reducefn((a: any, c: any) => a + c, 1).for(x.ndim)(x, out, 0);
                return out;
            },
            () => tensor([[66]])
        )
        .equal(
            () => {
                let x = arange(4 * 3).reshape(4, 3);
                let y = arange(3).expand(x.shape);
                let data: any[] = [];
                eachfn((x: any, y: any) => data.push([x, y]), 2).for(x.ndim)(x, y);
                return tensor(data);
            },
            () =>
                tensor([
                    [0, 0],
                    [1, 1],
                    [2, 2],
                    [3, 0],
                    [4, 1],
                    [5, 2],
                    [6, 0],
                    [7, 1],
                    [8, 2],
                    [9, 0],
                    [10, 1],
                    [11, 2],
                ])
        )
        .equal(
            () => {
                let x = arange(4 * 3).reshape(4, 3);
                let y = arange(3).expand(x.shape);
                let out = empty(4, 3);
                mapfn((a: number, b: number) => a * b, 2).for(x.ndim)(x, y, out);
                return out;
            },
            () =>
                tensor([
                    [0, 1, 4],
                    [0, 4, 10],
                    [0, 7, 16],
                    [0, 10, 22],
                ])
        );

function batch(fn: Function) {}
