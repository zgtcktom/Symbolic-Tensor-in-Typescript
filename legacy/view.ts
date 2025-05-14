// import { arange } from './arange.ts';
// import { broadcast } from './broadcast.ts';
// import { indices } from './indices.ts';
// import { Slice, slice } from './slice.ts';
// import { stack } from './stack.ts';
// import { Tensor, tensor } from './tensor.ts';
// import { TensorLike } from './tensorlike.ts';

// export class IndexedView {
//     at(...index: (number | number[] | string | Slice | any | null)[]): TensorLike {
//         throw new Error('Method not implemented.');
//     }

//     constructor(tensor: Tensor, index: (Tensor | null)[]) {
//         let adjacent = true;
//         let last = -1;
//         for (let i = 0; i < index.length; i++) {
//             if (index[i]) {
//                 if (last >= 0 && last + 1 != i) {
//                     adjacent = false;
//                     break;
//                 }
//                 last = i;
//             }
//         }
//         let start, end;
//         if (!adjacent) {
//             let front = [];
//             let back = [];
//             for (let axis = 0; axis < tensor.ndim; axis++) {
//                 (axis < index.length && index[axis] ? front : back).push(axis);
//             }
//             tensor = tensor.transpose(...front, ...back);
//             // console.log('transpose', tensor.shape);
//             let index_ = [];
//             for (let i = 0; i < front.length; i++) {
//                 index_.push(index[front[i]]);
//             }
//             index = index_;
//             start = 0;
//             end = front.length;
//         } else {
//             for (let i = 0; i < index.length; i++) {
//                 if (index[i]) {
//                     start = i;
//                     break;
//                 }
//             }
//             end = last + 1;
//         }

//         for (let i = 0; i < index.length; i++) {
//             if (index[i] == null) {
//                 index[i] = new Tensor([slice()]);
//             }
//         }
//         let indices = stack(broadcast(...index), -1);

//         // console.log(indices.shape, indices, start, end);
//         let { shape } = tensor;

//         shape = [...shape.slice(0, start), ...indices.shape.slice(0, -1), ...shape.slice(end)];
//         this.shape = shape;
//         this.ndim = shape.length;

//         this.tensor = tensor;
//         this.start = start;
//         this.end = start + indices.shape.length - 1;
//         indices = indices.reshape(-1, indices.shape[indices.shape.length - 1]).array();
//         this.indices = indices;
//         // console.log(tensor.shape);
//     }

//     set(tensor) {
//         tensor = tensor.expand(this.shape);
//         let indices_ = indices(this.shape.slice(this.start, this.end))
//             .reshape(this.end - this.start, -1)
//             .transpose()
//             .array();
//         // console.log('indices_.shape', indices_.shape);

//         for (let i = 0; i < this.indices.length; i++) {
//             let index = this.indices[i];
//             // console.log(tensor.shape, this.tensor.shape, index);

//             this.tensor.at(...index).set(tensor.at(...Array(this.start).fill(slice()), ...indices_[i]));
//         }
//         return this;
//     }
// }

// export const test = suite =>
//     suite
//         .equal(
//             () => {
//                 let x = arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5);
//                 new IndexedView(
//                     x,
//                     [
//                         null,
//                         [[[0], [1]]],
//                         [
//                             [[1], [3]],
//                             [[1], [2]],
//                             [[-1], [-2]],
//                         ],
//                     ].map(x => (x != null ? tensor(x) : x))
//                 ).set(tensor(1));
//                 return x;
//             },
//             () =>
//                 tensor([
//                     [
//                         [
//                             [0, 1, 2, 3, 4],
//                             [1, 1, 1, 1, 1],
//                             [10, 11, 12, 13, 14],
//                             [1, 1, 1, 1, 1],
//                         ],
//                         [
//                             [20, 21, 22, 23, 24],
//                             [25, 26, 27, 28, 29],
//                             [1, 1, 1, 1, 1],
//                             [1, 1, 1, 1, 1],
//                         ],
//                         [
//                             [40, 41, 42, 43, 44],
//                             [45, 46, 47, 48, 49],
//                             [50, 51, 52, 53, 54],
//                             [55, 56, 57, 58, 59],
//                         ],
//                     ],
//                     [
//                         [
//                             [60, 61, 62, 63, 64],
//                             [1, 1, 1, 1, 1],
//                             [70, 71, 72, 73, 74],
//                             [1, 1, 1, 1, 1],
//                         ],
//                         [
//                             [80, 81, 82, 83, 84],
//                             [85, 86, 87, 88, 89],
//                             [1, 1, 1, 1, 1],
//                             [1, 1, 1, 1, 1],
//                         ],
//                         [
//                             [100, 101, 102, 103, 104],
//                             [105, 106, 107, 108, 109],
//                             [110, 111, 112, 113, 114],
//                             [115, 116, 117, 118, 119],
//                         ],
//                     ],
//                 ])
//         )
//         .equal(
//             () => {
//                 let x = arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5);
//                 new IndexedView(
//                     x,
//                     [
//                         null,
//                         [[[0], [1]]],
//                         null,
//                         [
//                             [[1], [3]],
//                             [[1], [2]],
//                             [[-1], [-2]],
//                         ],
//                     ].map(x => (x != null ? tensor(x) : x))
//                 ).set(tensor(1));
//                 return x;
//             },
//             () =>
//                 tensor([
//                     [
//                         [
//                             [0, 1, 2, 3, 1],
//                             [5, 1, 7, 8, 1],
//                             [10, 1, 12, 13, 1],
//                             [15, 1, 17, 18, 1],
//                         ],
//                         [
//                             [20, 21, 1, 1, 24],
//                             [25, 26, 1, 1, 29],
//                             [30, 31, 1, 1, 34],
//                             [35, 36, 1, 1, 39],
//                         ],
//                         [
//                             [40, 41, 42, 43, 44],
//                             [45, 46, 47, 48, 49],
//                             [50, 51, 52, 53, 54],
//                             [55, 56, 57, 58, 59],
//                         ],
//                     ],
//                     [
//                         [
//                             [60, 1, 62, 63, 1],
//                             [65, 1, 67, 68, 1],
//                             [70, 1, 72, 73, 1],
//                             [75, 1, 77, 78, 1],
//                         ],
//                         [
//                             [80, 81, 1, 1, 84],
//                             [85, 86, 1, 1, 89],
//                             [90, 91, 1, 1, 94],
//                             [95, 96, 1, 1, 99],
//                         ],
//                         [
//                             [100, 101, 102, 103, 104],
//                             [105, 106, 107, 108, 109],
//                             [110, 111, 112, 113, 114],
//                             [115, 116, 117, 118, 119],
//                         ],
//                     ],
//                 ])
//         )
//         .equal(
//             () => {
//                 let x = arange(3 * 3 * 3).reshape(3, 3, 3);
//                 new IndexedView(
//                     x,
//                     [null, [0, -1], null].map(x => (x != null ? tensor(x) : x))
//                 ).set(
//                     tensor([
//                         [1, 2, 3],
//                         [-1, -2, -3],
//                     ])
//                 );
//                 return x;
//             },
//             () =>
//                 tensor([
//                     [
//                         [1, 2, 3],
//                         [3, 4, 5],
//                         [-1, -2, -3],
//                     ],
//                     [
//                         [1, 2, 3],
//                         [12, 13, 14],
//                         [-1, -2, -3],
//                     ],
//                     [
//                         [1, 2, 3],
//                         [21, 22, 23],
//                         [-1, -2, -3],
//                     ],
//                 ])
//         );
