// import { arange } from './arange.ts';
// import { tensor, Tensor } from './tensor.ts';
// import { _axis } from './tensorlike.ts';
// import { TestSuite } from './tester.ts';

// export function split<T>(x: Tensor<T>, size: number | number[], axis: number = 0): Tensor<T>[] {
//     let { data, shape, strides, offset, ndim } = x;
//     axis = _axis(axis, ndim);
//     let stride = strides[axis];

//     if (Array.isArray(size)) {
//         let n = size.length;
//         let tensors: Tensor<T>[] = Array(n);
//         for (let i = 0; i < n; i++) {
//             let _size = size[i];
//             tensors[i] = new Tensor(data, shape.with(axis, _size), strides, offset);
//             offset += stride * _size;
//         }
//         return tensors;
//     }

//     let dim = shape[axis];
//     if (dim == 0) return [x];

//     let n = (dim / size) | 0;
//     let last = dim % size;

//     let tensors: Tensor<T>[] = Array(last ? n + 1 : n);

//     shape = shape.with(axis, size);
//     for (let i = 0; i < n; i++) {
//         tensors[i] = new Tensor(data, shape, strides, offset);
//         offset += stride * size;
//     }

//     if (last) tensors[n] = new Tensor(data, shape.with(axis, last), strides, offset);

//     return tensors;
// }

// export const test = (suite: TestSuite) =>
//     suite
//         .equal(
//             () => split(arange(0), 2),
//             () => [tensor([])]
//         )
//         .equal(
//             () => split(arange(10).reshape(5, 2), 2),
//             () => [
//                 tensor([
//                     [0, 1],
//                     [2, 3],
//                 ]),
//                 tensor([
//                     [4, 5],
//                     [6, 7],
//                 ]),
//                 tensor([[8, 9]]),
//             ]
//         )
//         .equal(
//             () => split(arange(10).reshape(5, 2), [1, 4]),
//             () => [
//                 tensor([[0, 1]]),
//                 tensor([
//                     [2, 3],
//                     [4, 5],
//                     [6, 7],
//                     [8, 9],
//                 ]),
//             ]
//         );
