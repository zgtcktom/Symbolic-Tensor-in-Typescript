// export class OffsetIterator {
//     constructor(shape, strides, offset) {
//         // one-time use
//         this.shape = shape;
//         this.strides = strides;
//         this.offset = offset;

//         let ndim = shape.length;
//         this.ndim = ndim;
//         this.index = Array(ndim).fill(0);

//         this.value = offset;
//         this.done = false;
//     }

//     [Symbol.iterator]() {
//         return this;
//     }

//     next() {
//         if (this.done) return { value: undefined, done: true };
//         let { ndim, value } = this;

//         if (ndim == 0) {
//             this.done = true;
//             return { value, done: false };
//         }

//         let { index, shape, strides } = this;

//         let offset = value;
//         let axis = ndim - 1;
//         index[axis] += 1;
//         offset += strides[axis];

//         for (; index[axis] >= shape[axis]; axis--) {
//             if (axis == 0) {
//                 this.done = true;
//                 return { value, done: false };
//             }
//             index[axis] = 0;
//             index[axis - 1] += 1;
//             offset += strides[axis - 1] - shape[axis] * strides[axis];
//         }

//         this.value = offset;
//         return { value, done: false };
//     }
// }
