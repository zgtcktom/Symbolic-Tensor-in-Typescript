// import { OffsetIterator } from './offsetiterator.mjs';
// import { Slice } from './slice.mjs';
// import { tensor, Tensor } from './tensor.mjs';

// export class FlatIterator extends OffsetIterator {
//     constructor(base) {
//         let { shape, strides, offset } = base;
//         super(shape, strides, offset);
//         this.base = base;
//     }

//     next() {
//         let { value, done } = super.next();

//         if (done) return { value: undefined, done: true };
//         return { value: this.base.data[value], done: false };
//     }

//     get(index) {
//         if (typeof index == 'number') {
//             return this.base.item(index);
//         }

//         let indices, shape;
//         if (index instanceof Slice) {
//             indices = index.indices(this.base.size);
//             shape = [indices.length];
//         } else {
//             index = tensor(index);
//             indices = index.flat;
//             shape = index.shape;
//         }

//         let size = 1;
//         for (let axis = 0, ndim = shape.length; axis < ndim; axis++) size *= shape[axis];

//         let data = Array(size);
//         let i = 0;
//         for (let index of indices) {
//             data[i++] = this.base.item(index);
//         }

//         return new Tensor(data, shape);
//     }

//     set(index, value) {
//         if (typeof index == 'number') {
//             return this.base.itemset(index, value);
//         }

//         let indices;
//         if (index instanceof Slice) {
//             indices = index.indices(this.base.size);
//         } else {
//             indices = tensor(index).flat;
//         }

//         for (let index of indices) {
//             this.base.itemset(index, value);
//         }
//     }
// }
