const MAX_LENGTH = 2 ** 32 - 1;

export class Slice {
    start: number | null;
    stop: number | null;
    step: number | null;

    constructor(start: number | null, stop: number | null, step: number | null) {
        if (step === 0) throw new Error('step cannot be zero');
        this.start = start;
        this.stop = stop;
        this.step = step;
    }

    indices(length: number): Range {
        if (length < 0) throw new Error('length should not be negative');
        let { start, stop, step } = this;

        step ??= 1;
        if (step == 0) throw new Error('slice step cannot be zero');
        if (step < -MAX_LENGTH) step = -MAX_LENGTH;

        start ??= step < 0 ? MAX_LENGTH : 0;
        stop ??= step < 0 ? -MAX_LENGTH - 1 : MAX_LENGTH;

        if (start < 0) {
            start += length;
            if (start < 0) start = step < 0 ? -1 : 0;
        } else if (start >= length) {
            start = step < 0 ? length - 1 : length;
        }

        if (stop < 0) {
            stop += length;
            if (stop < 0) stop = step < 0 ? -1 : 0;
        } else if (stop >= length) {
            stop = step < 0 ? length - 1 : length;
        }

        return new Range(start, stop, step);
    }

    toString() {
        let { start, stop, step } = this;
        return `slice(${start}, ${stop}, ${step})`;
    }
}

export class Range {
    start: number;
    stop: number;
    step: number;
    length: number;

    constructor(start: number, stop: number, step: number) {
        this.start = start;
        this.stop = stop;
        this.step = step;
        this.length = Math.max(0, Math.ceil((stop - start) / step));
    }

    at(index: number): number;
    at(index: Slice): Range;
    at(index: number | Slice): number | Range {
        let { start, step, length } = this;
        if (typeof index == 'number') {
            if (index < -length || index >= length) throw new Error('index out of range');
            if (index < 0) index += length;
            return start + index * step;
        }
        let { start: _start, stop: _stop, step: _step } = index.indices(length);
        return new Range(start + _start * step, start + _stop * step, _step * step);
    }

    *[Symbol.iterator]() {
        let { start, step, length } = this;
        for (let i = 0; i < length; i++) {
            yield start + i * step;
        }
    }

    toString() {
        let { start, stop, step } = this;
        return `range(${start}, ${stop}, ${step})`;
    }
}

const colon = new Slice(null, null, null);

export function slice(start?: number | null, stop?: number | null, step?: number | null): Slice;
export function slice(arg: string): Slice;
export function slice(args: (number | undefined | null)[]): Slice;
export function slice(start?: string | number | null | (number | undefined | null)[], stop?: number | null, step?: number | null): Slice {
    if (typeof start == 'string') {
        if (start == ':') return colon;

        let elements = start.split(':');
        if (elements.length > 3) throw new Error('invalid syntax');

        let [_start, _stop, _step] = elements;
        start = _start ? +_start : null;
        stop = _stop ? +_stop : null;
        step = _step ? +_step : null;
    } else {
        if (Array.isArray(start)) [start, stop, step] = start;
        start ??= null;
        stop ??= null;
        step ??= null;
    }
    return new Slice(start, stop, step);
}

export const test = (suite: any) =>
    suite
        .equal(
            () => [...slice().indices(5)],
            () => [0, 1, 2, 3, 4]
        )
        .equal(
            () => [...new Slice(1, 4, 1).indices(5)],
            () => [1, 2, 3]
        )
        .equal(
            () => [...new Slice(1, 5, 2).indices(5)],
            () => [1, 3]
        )
        .equal(
            () => [...new Slice(3, 7, 2).indices(5)],
            () => [3]
        )
        .equal(
            () => [...new Slice(10, 0, 1).indices(5)],
            () => []
        )
        .equal(
            () => [...new Slice(10, 0, -1).indices(5)],
            () => [4, 3, 2, 1]
        )
        .equal(
            () => new Range(0, 5, 1).at(2),
            () => 2
        )
        .equal(
            () => new Range(0, 5, 1).at(-1),
            () => 4
        )
        .equal(
            () => [...slice('1:4:1').indices(5)],
            () => [1, 2, 3]
        )
        .equal(
            () => [...slice([1, 4, 1]).indices(5)],
            () => [1, 2, 3]
        );
