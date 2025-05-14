import { empty } from './empty.ts';
import { Tensor } from './tensor.ts';
import { reducefn } from './tensorfn.ts';

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

const _sum = reducefn((a: any, b: any) => a + b, 1);
