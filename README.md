# Symbolic Tensor Library in TypeScript

A TypeScript library for symbolic tensor operations, integrated with the `Tensor` class to enable automatic differentiation. Ideal for machine learning and scientific computing applications **(for educational use)**.

---

## Example: Neural Network Training

Below is an example of training a two-layer neural network to approximate the sine function using the Symbolic Tensor Library.

```typescript
// Define ReLU activation using existing operations
function relu(x: Symbolic): Symbolic {
    return x.add(x.abs()).div(2);
}

// Define mean function for MSE loss
function mean(x: Symbolic): Symbolic {
    return x.sum().div(x.size);
}

// Generate training data
let x_data = arange(-Math.PI, Math.PI, (Math.PI - -Math.PI) / 100).unsqueeze(1);
let y_data = x_data.map(Math.sin);

// Define symbolic input and output
let x = new Constant(x_data);
let y = new Constant(y_data);

// Define network parameters
const hidden_dim = 8;
let W1 = new Parameter(randn(1, hidden_dim));
let b1 = new Parameter(zeros(hidden_dim));
let W2 = new Parameter(randn(hidden_dim, 1));
let b2 = new Parameter(zeros(1));

// Define the two-layer DNN: xW + b
let hidden = x.matmul(W1).add(b1);
let hidden_activated = relu(hidden);
let output = hidden_activated.matmul(W2).add(b2);

// Define loss (Mean Squared Error)
let loss = mean(output.sub(y).pow(2));

// Training loop
const learning_rate = 0.05;
const epochs = 100;
for (let epoch = 0; epoch <= epochs; epoch++) {
    // Forward pass to compute loss
    let current_loss = loss.forward();

    // Backward pass to compute gradients
    loss.backward();

    // Update parameters using gradient descent
    for (let param of [W1, b1, W2, b2]) {
        param.value.sub(param.grad?.mul(learning_rate) ?? 0, param.value);
        delete param.grad;
    }

    // Report loss every 10 epochs
    if (epoch % 10 === 0) {
        console.log(`Epoch ${epoch}, Loss: ${current_loss}`);
    }
```

### Expected Output
```
Epoch 0, Loss: 13.604405718794625
Epoch 10, Loss: 0.1876167459766211
Epoch 20, Loss: 0.16583175100808945
Epoch 30, Loss: 0.14914665676599206
Epoch 40, Loss: 0.1357156601304027
Epoch 50, Loss: 0.12456694490427764
Epoch 60, Loss: 0.11491488716601057
Epoch 70, Loss: 0.10656432627782479
Epoch 80, Loss: 0.09928596500946857
Epoch 90, Loss: 0.09287564744943592
Epoch 100, Loss: 0.08721357332818439
```

### Key Steps
- **Model**: Two-layer network with ReLU activation.
- **Loss**: Mean squared error.
- **Training**: Gradient descent using automatic differentiation via `backward()`.

**Note**: This library is intended for **educational purposes**, demonstrating symbolic tensor operations and automatic differentiation. It is not optimized for production or performance-critical applications.

---

## Overview

The `Symbolic` class extends `TensorLike<number>`, providing a framework for symbolic tensor operations. Each operation, such as addition or multiplication, is represented by a subclass of `Symbolic`, which implements:
- `forward()`: Computes the operation's result.
- `backward()`: Propagates gradients for automatic differentiation.

The `symbolic()` utility converts inputs (numbers, `Tensor` objects, etc.) into `Symbolic` instances, typically as `Constant` objects.

This library is well-suited for:
- Building computation graphs for automatic differentiation.
- Performing tensor operations symbolically.
- Computing higher-order derivatives.

---

## Core API

### `symbolic(x: any): Symbolic`
- **Description**: Converts an input to a `Symbolic` object. If the input is already `Symbolic`, it is returned unchanged; otherwise, it is wrapped as a `Constant`.
- **Signature**: `symbolic(x: any): Symbolic`
- **Example**:
  ```typescript
  let x = symbolic(tensor([1, 2, 3])); // Creates a Constant
  let y = symbolic(5); // Creates a Constant with value 5
  ```

### `Symbolic`
- **Description**: Base class for symbolic tensor operations, providing methods for common tensor manipulations.

#### Methods
- **`copy(): Symbolic`**
  - Creates a symbolic copy of the tensor.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.copy(); // Symbolic copy of x
    ```

- **`contiguous(): Symbolic`**
  - Ensures the tensor is contiguous in memory symbolically.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.contiguous();
    ```

- **`flatten(start: number = 0, end: number = -1): Symbolic`**
  - Flattens the tensor between the specified axes.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2], [3, 4]]));
    let y = x.flatten(); // Results in [1, 2, 3, 4]
    ```

- **`ravel(): Symbolic`**
  - Returns a flattened version of the tensor.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2], [3, 4]]));
    let y = x.ravel(); // Results in [1, 2, 3, 4]
    ```

- **`expand(...shape: number[] | [number[]]): Symbolic`**
  - Expands the tensor to the specified shape via broadcasting.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.expand([2, 3]); // Results in [[1, 2, 3], [1, 2, 3]]
    ```

- **`reshape(...shape: number[] | [number[]]): Symbolic`**
  - Reshapes the tensor to the specified shape.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3, 4]));
    let y = x.reshape([2, 2]); // Results in [[1, 2], [3, 4]]
    ```

- **`transpose(...axes: number[] | [number[]]): Symbolic`**
  - Transposes the tensor according to the specified axes.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2], [3, 4]]));
    let y = x.transpose(); // Results in [[1, 3], [2, 4]]
    ```

- **`unsqueeze(axis: number): Symbolic`**
  - Adds a dimension of size 1 at the specified axis.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.unsqueeze(0); // Results in [[1, 2, 3]]
    ```

- **`squeeze(axis?: number | number[]): Symbolic`**
  - Removes dimensions of size 1 from the tensor.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[[1], [2]], [[3], [4]]]));
    let y = x.squeeze(2); // Results in [[1, 2], [3, 4]]
    ```

- **`at(...index: (Index | number[] | string)[]): Symbolic`**
  - Accesses tensor elements at the specified indices.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2, 3], [4, 5, 6]]));
    let y = x.at(0, 1); // Results in 2
    ```

- **`sum(axis?: number | number[], keepdim: boolean = false): Symbolic`**
  - Computes the sum along the specified axes.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2], [3, 4]]));
    let y = x.sum(0); // Results in [4, 6]
    ```

- **`add(x: any): Symbolic`**
  - Adds another tensor or scalar.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.add(2); // Results in [3, 4, 5]
    ```

- **`sub(x: any): Symbolic`**
  - Subtracts another tensor or scalar.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([3, 4, 5]));
    let y = x.sub(2); // Results in [1, 2, 3]
    ```

- **`mul(x: any): Symbolic`**
  - Multiplies by another tensor or scalar.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.mul(2); // Results in [2, 4, 6]
    ```

- **`div(x: any): Symbolic`**
  - Divides by another tensor or scalar.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([2, 4, 6]));
    let y = x.div(2); // Results in [1, 2, 3]
    ```

- **`reciprocal(): Symbolic`**
  - Computes the reciprocal.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 4]));
    let y = x.reciprocal(); // Results in [1, 0.5, 0.25]
    ```

- **`neg(): Symbolic`**
  - Negates the tensor.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, -2, 3]));
    let y = x.neg(); // Results in [-1, 2, -3]
    ```

- **`pow(x: any): Symbolic`**
  - Raises to the power of another tensor or scalar.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([2, 3, 4]));
    let y = x.pow(2); // Results in [4, 9, 16]
    ```

- **`log(): Symbolic`**
  - Computes the natural logarithm.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.log(); // Results in [0, log(2), log(3)]
    ```

- **`abs(): Symbolic`**
  - Computes the absolute value.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([-1, 2, -3]));
    let y = x.abs(); // Results in [1, 2, 3]
    ```

- **`sign(): Symbolic`**
  - Computes the sign.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([-1, 0, 3]));
    let y = x.sign(); // Results in [-1, 0, 1]
    ```

- **`dot(x: any): Symbolic`**
  - Computes the dot product.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([1, 2, 3]));
    let y = x.dot(tensor([4, 5, 6])); // Results in 32
    ```

- **`matmul(x: any): Symbolic`**
  - Performs matrix multiplication.
  - **Example**:
    ```typescript
    let x = new Parameter(tensor([[1, 2], [3, 4]]));
    let y = x.matmul(tensor([[5, 6], [7, 8]])); // Results in [[19, 22], [43, 50]]
    ```

---

## Subclasses of `Symbolic`

- **`Constant`**
  - Represents a constant tensor that does not require gradients.
  - **Constructor**: `new Constant(value: Tensor)`
  - **Example**:
    ```typescript
    let c = new Constant(tensor([1, 2, 3]));
    ```

- **`Parameter`**
  - Represents a tensor that requires gradients, typically for trainable parameters.
  - **Constructor**: `new Parameter(value: Tensor)`
  - **Example**:
    ```typescript
    let p = new Parameter(tensor([1, 2, 3]));
    p.backward(); // Accumulates gradients
    ```

- **Other Subclasses**: Operations like `Add`, `Mul`, `Reshape` are implemented as subclasses. Refer to the source code for details.

---

## Computing Second Derivatives

The library supports higher-order derivatives by treating gradients as `Symbolic` objects. After the first `backward()` call, the gradient (`grad`) can be used for further differentiation.

### Example: Second Derivative

```typescript
let x = new Parameter(tensor([[1, 2], [3, 4]]));
let z = x.pow(3); // z = x^3
z.backward(new Constant(tensor(1))); // First derivative: dz/dx = 3x^2
let grad = x.grad as Symbolic;
delete x.grad; // Reset gradient
grad.backward(); // Second derivative: d(dz/dx)/dx = 6x
console.log(x.grad); // Outputs tensor([[6, 12], [18, 24]])
```

- **Steps**:
  1. Compute `z = x^3`.
  2. First `backward()` computes `dz/dx = 3x^2`.
  3. Treat `grad` as `Symbolic` and call `backward()` to compute `d(3x^2)/dx = 6x`.

This method extends to higher-order derivatives.

---

## Usage Notes

- **Integration with `Tensor`**: `forward()` returns a `Tensor`, ensuring compatibility.
- **Automatic Differentiation**: Use `backward()` to compute gradients.
- **Flexibility**: Methods accept scalars, tensors, or `Symbolic` objects, with `symbolic()` handling conversions.
- **Educational Focus**: Designed for learning symbolic operations and autodiff, not for production.

---