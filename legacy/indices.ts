// import { tensor, Tensor } from './tensor.ts';

// export function indices(shape:number[], sparse = false) {
//     let ndim = shape.length;

//     if (sparse) {
//         let indices = Array(ndim);
//         for (let axis = 0; axis < ndim; axis++) {
//             let dim = shape[axis];
//             let data = Array(dim);
//             let _shape = Array(ndim).fill(1);
//             for (let i = 0; i < dim; i++) data[i] = i;
//             _shape[axis] = dim;
//             indices[axis] = new Tensor(data, _shape);
//         }

//         return indices;
//     }

//     let size = 1;
//     for (let axis = 0; axis < ndim; axis++) size *= shape[axis];

//     let data = Array(ndim * size);
//     let indices = Array(ndim).fill(0);

//     for (let i = 0; i < size; i++) {
//         for (let k = 0; k < ndim; k++) {
//             data[k * size + i] = indices[k];
//         }

//         let idx = ndim - 1;
//         while (idx >= 0 && indices[idx] + 1 >= shape[idx]) {
//             indices[idx--] = 0;
//         }

//         if (idx >= 0) indices[idx]++;
//     }

//     return new Tensor(data, shape.toSpliced(0, 0, ndim));
// }

// export const test = suite =>
//     suite
//         .equal(
//             () => indices([2, 3, 4], true),
//             () => [tensor([[[0]], [[1]]]), tensor([[[0], [1], [2]]]), tensor([[[0, 1, 2, 3]]])]
//         )
//         .equal(
//             () => indices([2, 3]),
//             () =>
//                 tensor([
//                     [
//                         [0, 0, 0],
//                         [1, 1, 1],
//                     ],
//                     [
//                         [0, 1, 2],
//                         [0, 1, 2],
//                     ],
//                 ])
//         )
//         .equal(
//             () => indices([2, 3, 4]),
//             () =>
//                 tensor([
//                     [
//                         [
//                             [0, 0, 0, 0],
//                             [0, 0, 0, 0],
//                             [0, 0, 0, 0],
//                         ],
//                         [
//                             [1, 1, 1, 1],
//                             [1, 1, 1, 1],
//                             [1, 1, 1, 1],
//                         ],
//                     ],
//                     [
//                         [
//                             [0, 0, 0, 0],
//                             [1, 1, 1, 1],
//                             [2, 2, 2, 2],
//                         ],
//                         [
//                             [0, 0, 0, 0],
//                             [1, 1, 1, 1],
//                             [2, 2, 2, 2],
//                         ],
//                     ],
//                     [
//                         [
//                             [0, 1, 2, 3],
//                             [0, 1, 2, 3],
//                             [0, 1, 2, 3],
//                         ],
//                         [
//                             [0, 1, 2, 3],
//                             [0, 1, 2, 3],
//                             [0, 1, 2, 3],
//                         ],
//                     ],
//                 ])
//         );
