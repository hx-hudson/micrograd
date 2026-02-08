# Micrograd with NumPy Arrays

A minimal automatic differentiation engine inspired by Karpathy's micrograd,
extended to support `numpy.ndarray`, broadcasting, and simple neural networks.

## Features
- Scalar + `numpy.ndarray` autodiff
- Broadcasting with correct backward (`unbroadcast`)
- Custom operators: add, mul, pow, tanh, sum, stack
- Simple MLP built on top of the engine
- SGD optimizer

## Example

```python
x = Value(np.array([1.0, 2.0]))
w = Value(np.array([3.0, 4.0]))
y = (x * w).sum()
y.backward_all()
print(x.grad)  # -> array([3., 4.])