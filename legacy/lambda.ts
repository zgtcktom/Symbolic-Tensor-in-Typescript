import { TestSuite } from '../tester';

let x = (a: any, b: any, c: any) => a + b + c;
console.log(x.toString());

function lambda(arrowfn: Function) {
    let script = arrowfn.toString();

    if (script.startsWith('function') || !script.includes('=>')) {
        throw new Error('only arrow function is accepted');
    }
    let param = script.slice(0, script.indexOf('=>')).trim();
    if (param.includes('=')) {
        throw new Error('default argument is not supported');
    }
    let params: string[];
    if (param[0] == '(') {
        params = param
            .slice(1, -1)
            .split(',')
            .map(x => x.trim());
    } else {
        params = [param];
    }

    let body = script.slice(script.indexOf('=>') + 2).trim();
    if (body[0] == '{') {
        body = body.slice(1, -1).trim();
    }

    console.log(params, body);
}

console.log(lambda((a: any, b: any, c: any) => a + b + c));
console.log(
    lambda((a: any, b: any, c: any) => {
        return a + b + c;
    })
);

console.log(lambda(x => x));
