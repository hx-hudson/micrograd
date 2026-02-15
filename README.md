# Micrograd with NumPy Arrays

A minimal automatic differentiation engine inspired by Karpathy's micrograd,
extended to support `numpy.ndarray`, broadcasting, matrix multiplication,
convolutions, and vectorized neural networks — including a CNN trained on MNIST.

## Features

- Scalar + `numpy.ndarray` autodiff
- Broadcasting with correct backward (`unbroadcast`)
- **Matrix multiplication (`@`) with full backward support**
  - Vector dot product
  - Matrix–matrix multiplication
  - Batched matrix multiplication (NumPy-style broadcasting)
- Custom operators: `add`, `mul`, `pow`, `tanh`, `relu`, `exp`, `log`, `sum`, `reshape`, `stack`, `matmul`
- 2D convolution (`conv2d`) with stride and padding support
- Max pooling (`max_pooling`)
- Cross-entropy loss with numerically stable softmax
- Vectorized linear layers (`x @ W + b`)
- Convolutional layer (`Conv2dLayer`)
- Simple MLP and CNN built on top of the engine
- SGD optimizer
- MNIST training script

## Operators

| Operator | Description |
|---|---|
| `+`, `-`, `*`, `/` | Elementwise arithmetic with broadcasting |
| `**` | Power (scalar exponent only) |
| `@` | Matrix multiplication (supports batched) |
| `tanh()` | Elementwise tanh activation |
| `relu()` | Elementwise ReLU activation |
| `exp()` | Elementwise exponential |
| `log(eps=1e-12)` | Elementwise natural log (numerically stable) |
| `sum()` | Reduce all elements to scalar |
| `reshape(*shape)` | Reshape tensor, gradient flows back correctly |
| `stack(list)` | Stack a list of `Value` objects along a new axis |

## Standalone Functions

| Function | Description |
|---|---|
| `conv2d(x, w, b=None, stride=1, padding=0)` | 2D convolution with optional bias, stride, and zero-padding |
| `max_pooling(x, stride)` | Max pooling with full backward pass |
| `cross_entropy(x, y)` | Softmax + cross-entropy loss (numerically stable) |

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

## Example: Training Loop (XOR)

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

## Example: CNN on MNIST

```python
from model import CNN
from engine import cross_entropy, Optimizer

model = CNN()
optimizer = Optimizer(model.parameters())

# X_train: (60000, 1, 28, 28) float32, normalized
# y_train: (60000,) int

logits = model.forward(X_batch)          # (batch, 10)
loss = cross_entropy(logits, y_batch)
loss.backward_all()
optimizer.update(lr=0.01)
```

Run the full training script:

```bash
python train_on_MNIST.py
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

### `Neuron`
A single neuron with randomized weights and a tanh activation.

```python
neuron = Neuron(input_num=4)
y = neuron(x)   # output is tanh(x @ w + b)
```

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

### `Conv2dLayer`
A 2D convolutional layer wrapping the `conv2d` function.

```python
Conv2dLayer(in_channel, out_channel, kernel_size, bias=True, stride=1, padding=0)
```

```python
layer = Conv2dLayer(1, 8, 3, padding=1)
y = layer(x)   # x: (batch, 1, H, W) -> y: (batch, 8, H, W)
```

### `CNN`
A small LeNet-style CNN for MNIST classification defined in `model.py`.

```
input:     (batch, 1, 28, 28)
conv1:     (batch, 4, 28, 28)   [3×3, padding=1]
max_pool:  (batch, 4, 14, 14)   [stride=2]
conv2:     (batch, 8, 14, 14)   [3×3, padding=1]
max_pool:  (batch, 8, 7, 7)     [stride=2]
linear:    (batch, 10)          [392 → 10]
```

```python
from model import CNN
model = CNN()
logits = model(x)   # (batch, 10)
```

### `Optimizer`
Vanilla SGD optimizer.

```python
opt = Optimizer(model.parameters())
opt.zero_grad()   # clear gradients before each step
opt.update(lr)    # apply gradient update
```