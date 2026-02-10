# Micrograd with NumPy Arrays

A minimal automatic differentiation engine inspired by Karpathy's micrograd,
extended to support `numpy.ndarray`, broadcasting, matrix multiplication, and vectorized neural networks.

## Features

- Scalar + `numpy.ndarray` autodiff
- Broadcasting with correct backward (`unbroadcast`)
- **Matrix multiplication (`@`) with full backward support**
  - Vector dot product
  - Matrix–matrix multiplication
  - Batched matrix multiplication (NumPy-style broadcasting)
- Custom operators: `add`, `mul`, `pow`, `tanh`, `sum`, `reshape`, `stack`, `matmul`
- Vectorized linear layers (`x @ W + b`)
- Simple MLP built on top of the engine
- SGD optimizer

## Operators

| Operator | Description |
|---|---|
| `+`, `-`, `*`, `/` | Elementwise arithmetic with broadcasting |
| `**` | Power (scalar exponent only) |
| `@` | Matrix multiplication (supports batched) |
| `tanh()` | Elementwise activation |
| `sum()` | Reduce all elements to scalar |
| `reshape(*shape)` | Reshape tensor, gradient flows back correctly |
| `stack(list)` | Stack a list of `Value` objects along a new axis |

## Matrix Multiplication

Supports the `@` operator (`__matmul__`) with full backward propagation.

Supported cases:
- Vector dot product: `(n,) @ (n,) → scalar`
- Matrix–vector: `(m, n) @ (n,) → (m,)`
- Matrix–matrix: `(m, n) @ (n, p) → (m, p)`
- Batched matmul with broadcasting: `(..., m, n) @ (..., n, p) → (..., m, p)`


## Example: Dot Product

```python
x = Value(np.array([1.0, 2.0]))
w = Value(np.array([3.0, 4.0]))

y = x @ w
y.backward_all()

print(y.data)   # -> 11.0
print(x.grad)   # -> array([3., 4.])
print(w.grad)   # -> array([1., 2.])
```

## Example: MLP

```python
# Define a 3-layer MLP: input=3, hidden=4, output=2
mlp = MLP(3, [4, 2])

# Batch input
x = Value(np.random.randn(5, 3))

# Forward pass (output is reshaped to (N, out))
y = mlp(x)

# Simple loss
loss = y.sum()
loss.backward_all()

print(y.data.shape)  # -> (5, 2)
```

## Example: Training Loop

```python
import numpy as np
from engine import Value, MLP, Optimizer

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

model = MLP(2, [4, 1])
opt = Optimizer(model.parameters())

for epoch in range(3000):
    opt.zero_grad()
    pred = model(Value(X))
    diff = pred - Value(y)
    loss = (diff * diff).sum()
    loss.backward_all()
    opt.update(lr=0.05)
```

## Classes

### `Value`
The core autograd node. Wraps a scalar or `numpy.ndarray` and records operations
for automatic differentiation.

```python
Value(data, _children=(), _op='')
```

- `data`: scalar or `np.ndarray`
- `grad`: gradient of the same shape as `data`, accumulated during `backward_all()`
- `backward_all()`: runs backpropagation from this node through the full computation graph

### `LinearLayer`
A fully connected layer: `y = x @ W + b`

```python
layer = LinearLayer(input_num=4, output_num=8)
y = layer(x)
```

### `MLP`
A multi-layer perceptron with `tanh` activations on all hidden layers.
The final output is reshaped to `(N, out)` for consistent batch handling.

```python
mlp = MLP(input_num=3, layer_num=[8, 8, 1])
```

### `Optimizer`
Vanilla SGD optimizer.

```python
opt = Optimizer(model.parameters())
opt.zero_grad()   # clear gradients before each step
opt.update(lr)    # apply gradient update
```
