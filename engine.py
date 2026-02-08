import random
import numpy as np


def unbroadcast(grad, shape):
    if isinstance(grad, (int, float)):
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (grad_dim, shape_dim) in enumerate(zip(grad.shape, shape)):
        if shape_dim == 1 and not grad_dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

def stack(x:list):
    data = [value.data for value in x]
    data = np.array(data)
    out = Value(data, x, "stack")
    def backward():
        for i,value in enumerate(x):
            value.grad += out.grad[i]
    out._backward = backward
    return out

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None

        if isinstance(self.data,np.ndarray):
            self.grad = np.zeros_like(data, dtype=float)
        else:
            self.grad = 0.0

    def __add__(self, other):
        other = Value(other) if isinstance(other, (int, float,np.ndarray,np.number)) else other
        x = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += unbroadcast(x.grad, np.shape(self.data))
            other.grad += unbroadcast(x.grad, np.shape(other.data))
        x._backward = backward
        return x

    def __mul__(self, other):
        other =  Value(other) if isinstance(other,(int,float,np.ndarray,np.number)) else other
        x = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += unbroadcast(other.data*x.grad, np.shape(self.data))
            other.grad += unbroadcast(self.data*x.grad, np.shape(other.data))
        x._backward = backward
        return x

    def __pow__(self, power):
        assert isinstance(power, (int, float))
        x = Value(self.data ** power, (self,), '**')
        def backward():
            self.grad += unbroadcast(power*(self.data**(power - 1))*x.grad, np.shape(self.data))
        x._backward = backward
        return x

    def tanh(self):
        e2x = np.exp(2*self.data)
        x = 1 - 2/(e2x+1)
        x = Value(x, (self,), 'tanh')
        def backward():
            self.grad += unbroadcast((1 - x.data**2)*x.grad, np.shape(self.data))
        x._backward = backward
        return x

    def __rtruediv__(self, other):
        return Value(other)*self**-1

    def __radd__(self, other):
        return self+other

    def __rmul__(self, other):
        return self*other

    def __neg__(self):
        return self*-1

    def __sub__(self, other):
        return self+(-other)

    def __truediv__(self, other):
        return self*other**-1

    def backward_all(self):
        visited = set()
        node = []
        def topo(x):
            if x not in visited:
                visited.add(x)
                for n in x._children:
                    topo(n)
                node.append(x)
        topo(self)
        self.grad = 1.0
        for v in reversed(node):
            v._backward()

    def sum(self):
        x = Value(self.data.sum(),(self,),"sum")
        def backward():
            self.grad +=  np.ones_like(self.data, dtype=float) * x.grad
        x._backward = backward
        return x

    def __repr__(self):
        return f"Data:{self.data}"

class Neuron:
    def __init__(self, input_num):
        self.ws = Value(np.random.randn(input_num))
        self.bias = Value(random.uniform(-1.0,1.0))

    def forward(self, input_x):
        result = (input_x * self.ws).sum() + self.bias
        return result.tanh()

    def __call__(self, input_x):
        return self.forward(input_x)

    def parameters(self):
        return [self.ws, self.bias]

class Layer:
    def __init__(self, input_num, output_num):
        self.neurons = [Neuron(input_num) for i in range(output_num)]

    def forward(self, input_x):
        result = [neuron(input_x) for neuron in self.neurons]
        result = stack(result)
        return result

    def __call__(self, input_x):
        return self.forward(input_x)

    def parameters(self):
        return [ws for neuron in self.neurons for ws in neuron.parameters()]

class MLP:
    def __init__(self, input_num, layer_num:list):
        length = len(layer_num)
        sizes = [input_num] + list(layer_num)
        self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(length)]

    def forward(self, input_x):
        result = input_x
        for layer in self.layers:
            result = layer(result)
        return result

    def __call__(self, input_x):
        return self.forward(input_x)

    def parameters(self):
        return [ws for layer in self.layers for ws in layer.parameters()]

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            if isinstance(parameter.data,np.ndarray):
                parameter.grad = np.zeros_like(parameter.grad)
            else:
                parameter.grad  = 0.0

    def update(self, lr):
        for parameter in self.parameters:
            parameter.data -= lr*parameter.grad

