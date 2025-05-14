import { arange } from './arange.ts';
import { broadcastShape } from './broadcast.ts';
import { empty, ones } from './empty.ts';
import { dot } from './math.ts';
import { tensor, Tensor } from './tensor.ts';
import { each } from './tensorfn.ts';
import { TestSuite } from './tester.ts';

export function matmul(x: Tensor, y: Tensor, out?: Tensor | null): Tensor {
    x = tensor(x);
    y = tensor(y);

    if (x.ndim == 1 && y.ndim == 1) return dot(x, y, out);

    if (x.ndim == 2 && y.ndim == 2) {
        let m = x.shape[x.ndim - 2];
        let n = x.shape[x.ndim - 1];
        let p = y.shape[y.ndim - 2];
        let q = y.shape[y.ndim - 1];

        if (n !== p) throw new Error(`incompatible shapes: ${x.shape} and ${y.shape}`);
        out ??= empty(m, q);
        return matmul2d(x, y, out, x.offset, y.offset, out.offset);
    }

    if (x.ndim == 1) return matmul(x.unsqueeze(0), y, out?.unsqueeze(-2)).squeeze(-2);

    if (y.ndim == 1) return matmul(x, y.unsqueeze(1), out?.unsqueeze(-1)).squeeze(-1);

    if (x.ndim == 0 || y.ndim == 0) {
        throw new Error('incompatible 0d tensors');
    }

    let xbatch = x.shape.slice(0, -2);
    let ybatch = y.shape.slice(0, -2);

    let batch = broadcastShape([xbatch, ybatch]);
    x = x.expand(...batch, ...x.shape.slice(-2));
    y = y.expand(...batch, ...y.shape.slice(-2));

    let m = x.shape[x.ndim - 2];
    let n = x.shape[x.ndim - 1];
    let p = y.shape[y.ndim - 2];
    let q = y.shape[y.ndim - 1];

    if (n !== p) throw new Error(`incompatible shapes: ${x.shape} and ${y.shape}`);

    out ??= empty(...batch, m, q);

    if (batch.length == 1) {
        return matmul2d_batch(x, y, out, x.offset, y.offset, out.offset);
    }
    _matmul.for(batch.length)(x, y, out);
    return out;
}

function matmul2d_batch(x: Tensor, y: Tensor, out: Tensor, xoffset: number, yoffset: number, outoffset: number): Tensor {
    // let xn = x.ndim,
    //     yn = y.ndim;

    let xs = x.shape,
        ys = y.shape;

    let xt = x.strides,
        yt = y.strides;

    let xi = 1,
        xj = 2;
    let yi = 1,
        yj = 2;

    let m = xs[xi];
    let n = xs[xj];
    // let p = ys[yi]; // assuming n === p
    // if (n !== p) throw new Error(`incompatible shapes: ${x.shape} and ${y.shape}`);
    let q = ys[yj];

    // let on = out.ndim;
    let ot = out.strides;
    let oi = 1,
        oj = 2;

    let batch = xs[0];
    let xstride = xt[0];
    let ystride = yt[0];
    let ostride = ot[0];

    let xu = xt[xi],
        xv = xt[xj];
    let yu = yt[yi],
        yv = yt[yj];
    let ou = ot[oi],
        ov = ot[oj];

    let xd = x.data,
        yd = y.data,
        od = out.data;

    for (let _ = 0; _ < batch; _++) {
        for (let i = 0; i < m; i++) {
            let x_row = xoffset + i * xu;
            let out_row = outoffset + i * ou;
            for (let j = 0; j < q; j++) {
                let y_col = yoffset + j * yv;
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += xd[x_row + k * xv] * yd[y_col + k * yu];
                }
                od[out_row + j * ov] = sum;
            }
        }
        xoffset += xstride;
        yoffset += ystride;
        outoffset += ostride;
    }
    return out;
}

function matmul2d(x: Tensor, y: Tensor, out: Tensor, xoffset: number, yoffset: number, outoffset: number): Tensor {
    let xn = x.ndim,
        yn = y.ndim;

    let xs = x.shape,
        ys = y.shape;

    let xt = x.strides,
        yt = y.strides;

    let xi = xn - 2,
        xj = xn - 1;
    let yi = yn - 2,
        yj = yn - 1;

    let m = xs[xi];
    let n = xs[xj];
    // let p = ys[yi]; // assuming n === p
    // if (n !== p) throw new Error(`incompatible shapes: ${x.shape} and ${y.shape}`);
    let q = ys[yj];

    let on = out.ndim;
    let ot = out.strides;
    let oi = on - 2,
        oj = on - 1;

    let xu = xt[xi],
        xv = xt[xj];
    let yu = yt[yi],
        yv = yt[yj];
    let ou = ot[oi],
        ov = ot[oj];

    let xd = x.data,
        yd = y.data,
        od = out.data;

    for (let i = 0; i < m; i++) {
        let x_row = xoffset + i * xu;
        let out_row = outoffset + i * ou;
        for (let j = 0; j < q; j++) {
            let y_col = yoffset + j * yv;
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += xd[x_row + k * xv] * yd[y_col + k * yu];
            }
            od[out_row + j * ov] = sum;
        }
    }
    return out;
}

const _matmul2d = Symbol('matmul2d');
// usage of batch mode
let _matmul = each(
    ({ x, [_matmul2d]: matmul2d }) => {
        return `${matmul2d}(${x[0]}, ${x[1]}, ${x[2]}, ${x[0].offset}, ${x[1].offset}, ${x[2].offset})`;
    },
    3,
    [_matmul2d],
    { [_matmul2d]: matmul2d }
);

// let _matmul = each(({ x }) => {
//     return `(${matmul2d.toString()})(${x[0]}, ${x[1]}, ${x[2]}, ${x[0].offset}, ${x[1].offset}, ${x[2].offset})`;
// }, 3);

// let x = arange(10 * 3 * 4).reshape(10, 3, 4);
// let y = arange(4 * 5).reshape(4, 5);
// let test_a = () => matmul(x, y);
// test_a();
// console.time();
// for (let i = 0; i < 1000000; i++) {
//     (() => test_a())();
// }
// console.timeEnd();

// console.log(_matmul.for(2).toString());

const _batchify: any[] = [];
const _batchifyfor: any[][] = [];
const _fn = Symbol('fn');
const _views = Symbol('views');
function batchify(fn: Function, xs: Tensor[], ndim: number) {
    let views = Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
        let x = xs[i];
        views[i] = new Tensor(x.data, x.shape.slice(ndim), x.strides.slice(ndim), x.offset);
    }
    _batchify[xs.length] ??= each(
        ({ x, [_fn]: fn, [_views]: views }) => `${[...x.map((x, i) => `${views}[${i}].offset = ${x.offset}`), `${fn}(${x.map((_, i) => `${views}[${i}]`).join(',')})`].join(';')}`,
        xs.length,
        [_fn, _views]
    );

    _batchifyfor[xs.length] ??= [];
    _batchifyfor[xs.length][ndim] ??= _batchify[xs.length].for(ndim);

    _batchifyfor[xs.length][ndim](...xs, fn, views);
}

// batchify((x: Tensor, y: Tensor) => console.log(x, y), [arange(2 * 3 * 4).reshape(2, 3, 4), arange(2 * 3 * 2 * 1).reshape(2, 3, 2, 1)], 2);

export const test = (suite: TestSuite) =>
    suite
        .equal(
            () =>
                matmul(
                    tensor([
                        [2, 3],
                        [2, 4],
                    ]),
                    tensor([
                        [2, 1],
                        [2, 6],
                    ])
                ),
            () =>
                tensor([
                    [10, 20],
                    [12, 26],
                ])
        )
        .equal(
            () => matmul(tensor([2, 4]), tensor([2, 6])),
            () => tensor(28)
        )
        .equal(
            () =>
                matmul(
                    tensor([2, 4]),
                    tensor([
                        [2, 6],
                        [3, 2],
                    ])
                ),
            () => tensor([16, 20])
        )
        .equal(
            () =>
                matmul(
                    tensor([
                        [2, 6],
                        [3, 2],
                    ]),
                    tensor([2, 4])
                ),
            () => tensor([28, 14])
        )
        .equal(
            () => matmul(arange(10 * 3 * 4).reshape(10, 3, 4), arange(4 * 5).reshape(4, 5)),
            () =>
                tensor([
                    [
                        [70, 76, 82, 88, 94],
                        [190, 212, 234, 256, 278],
                        [310, 348, 386, 424, 462],
                    ],
                    [
                        [430, 484, 538, 592, 646],
                        [550, 620, 690, 760, 830],
                        [670, 756, 842, 928, 1014],
                    ],
                    [
                        [790, 892, 994, 1096, 1198],
                        [910, 1028, 1146, 1264, 1382],
                        [1030, 1164, 1298, 1432, 1566],
                    ],
                    [
                        [1150, 1300, 1450, 1600, 1750],
                        [1270, 1436, 1602, 1768, 1934],
                        [1390, 1572, 1754, 1936, 2118],
                    ],
                    [
                        [1510, 1708, 1906, 2104, 2302],
                        [1630, 1844, 2058, 2272, 2486],
                        [1750, 1980, 2210, 2440, 2670],
                    ],
                    [
                        [1870, 2116, 2362, 2608, 2854],
                        [1990, 2252, 2514, 2776, 3038],
                        [2110, 2388, 2666, 2944, 3222],
                    ],
                    [
                        [2230, 2524, 2818, 3112, 3406],
                        [2350, 2660, 2970, 3280, 3590],
                        [2470, 2796, 3122, 3448, 3774],
                    ],
                    [
                        [2590, 2932, 3274, 3616, 3958],
                        [2710, 3068, 3426, 3784, 4142],
                        [2830, 3204, 3578, 3952, 4326],
                    ],
                    [
                        [2950, 3340, 3730, 4120, 4510],
                        [3070, 3476, 3882, 4288, 4694],
                        [3190, 3612, 4034, 4456, 4878],
                    ],
                    [
                        [3310, 3748, 4186, 4624, 5062],
                        [3430, 3884, 4338, 4792, 5246],
                        [3550, 4020, 4490, 4960, 5430],
                    ],
                ])
        )
        .equal(
            () => matmul(ones(3), ones(2, 3, 4)).shape,
            () => [2, 4]
        )
        .equal(
            () => matmul(arange(2 * 1 * 3 * 4).reshape(2, 1, 3, 4), arange(1 * 2 * 4 * 5).reshape(1, 2, 4, 5)).shape,
            () => [2, 2, 3, 5]
        )
        .equal(
            () => matmul(tensor([1, 2, 3]), ones(2, 3, 4)),
            () =>
                tensor([
                    [6, 6, 6, 6],
                    [6, 6, 6, 6],
                ])
        )
        .equal(
            () => matmul(ones(2, 3, 4), tensor([1, 2, 3, 4])),
            () =>
                tensor([
                    [10, 10, 10],
                    [10, 10, 10],
                ])
        )
        .equal(
            () => {
                const out = empty(2, 2);
                matmul(
                    tensor([
                        [1, 2],
                        [3, 4],
                    ]),
                    tensor([
                        [2, 0],
                        [1, 3],
                    ]),
                    out
                );
                return out;
            },
            () =>
                tensor([
                    [4, 6],
                    [10, 12],
                ])
        );
