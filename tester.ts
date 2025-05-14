import { Tensor } from './tensor.ts';

function compare(a: any, b: any, depth = Infinity): boolean {
    if (a === b) return true;
    if (a instanceof Tensor && b instanceof Tensor) return a.equal(b);
    if (!Array.isArray(a) || !Array.isArray(b) || a.length != b.length) return false;

    for (let i = 0, n = a.length; i < n; i++) {
        if (a[i] === b[i]) continue;
        if (depth > 0 && compare(a[i], b[i], depth - 1)) continue;
        return false;
    }
    return true;
}

export abstract class TestRun {
    name: string;
    constructor(name: string) {
        this.name = name;
    }

    abstract run(verbose: boolean, prefix: string, indent: number): boolean;

    abstract assert(callable: () => boolean): this;

    abstract equal(xfn: () => any, yfn: () => any): this;
}

const indentation = '  ';

export class TestCase extends TestRun {
    callable?: () => boolean;
    constructor(name: string) {
        super(name);
    }

    run(verbose = true, prefix = '', indent = 0): boolean {
        let ok = false;

        let start = performance.now();
        try {
            if (this.callable?.()) ok = true;
        } catch (error) {
            if (verbose) console.error(error);
        }
        let delta = ((performance.now() - start) / 1000).toFixed(4);

        let status = ok ? '\x1b[32mOK\x1b[0m' : '\x1b[31mNot OK\x1b[0m';
        if (prefix) prefix += ':';
        if (verbose) console.log(`${indentation.repeat(indent)}[${status}] ${prefix}${this.name} in ${delta}s`);

        return ok;
    }

    assert(callable: () => boolean): this {
        this.callable = callable;
        return this;
    }

    equal(xfn: () => any, yfn: () => any): this {
        return this.assert(() => compare(xfn(), yfn()));
    }
}

export class TestSuite extends TestRun {
    cases: TestRun[];
    constructor(name: string) {
        super(name);
        this.cases = [];
    }

    run(verbose = true, prefix = '', indent = 0): boolean {
        if (prefix) prefix += ':';

        let total = this.cases.length;

        if (verbose) console.log(`${indentation.repeat(indent)}Running ${prefix}${this.name} (${total} test cases)`);

        let start = performance.now();
        let ok = 0;

        for (let i = 0; i < total; i++) {
            ok += this.cases[i].run(verbose, this.name, indent + 1) ? 1 : 0;
        }

        this.cases.length = 0;
        let delta = ((performance.now() - start) / 1000).toFixed(4);
        let msg = ok == total ? '\x1b[32mOK\x1b[0m' : '\x1b[31mNot OK\x1b[0m';
        if (verbose) console.log(`${indentation.repeat(indent)}[${msg}] ${ok}/${total} (${prefix}${this.name}) in ${delta}s`);

        return ok == total;
    }

    add(testRun: TestRun) {
        this.cases.push(testRun);
        return this;
    }

    assert(callable: () => boolean): this {
        let name = `#${this.cases.length}`;
        return this.add(new TestCase(name).assert(callable));
    }

    equal(xfn: () => any, yfn: () => any): this {
        let name = `#${this.cases.length}`;
        return this.add(new TestCase(name).equal(xfn, yfn));
    }
}

export const test = (suite: TestSuite) =>
    suite
        .assert(() => {
            return true;
        })
        .equal(
            () => [[3]],
            () => [[3]]
        );

const DEBUG = process?.argv.includes('--debug') || process?.argv.includes('-d');

if (DEBUG) {
    const testSuite = new TestSuite('main');

    let ready = false;
    process.on('beforeExit', async () => {
        if (ready) return;
        ready = true;

        const fs = await import('fs');

        let ext = '.ts';
        for (let file of fs.readdirSync('.')) {
            if (file != 'test' + ext && file.endsWith(ext)) {
                let { test } = await import(`./${file}`);
                if (test) {
                    file = file.slice(0, -ext.length);
                    let suite = new TestSuite(file);
                    test(suite);
                    testSuite.add(suite);
                }
            }
        }
    });

    process.on('exit', () => {
        testSuite.run();
    });
}
