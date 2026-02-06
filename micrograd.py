import random
import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._children = set(_children)
        self._op = _op
        self.backward = lambda: None
        self.grad = 0.0

    def __add__(self, other):
        other = Value(other) if isinstance(other, (int, float)) else other
        x = Value(self.data + other.data, (self, other), '+')
        def backward():
            self.grad += x.grad
            other.grad += x.grad
        x.backward = backward
        return x

    def __mul__(self, other):
        other =  Value(other) if isinstance(other,(int,float)) else other
        x = Value(self.data * other.data, (self, other), '*')
        def backward():
            self.grad += other.data*x.grad
            other.grad += self.data*x.grad
        x.backward = backward
        return x

    def __pow__(self, power):
        assert isinstance(power, int)
        x = Value(pow(self.data, power), (self,), '**')
        def backward():
            self.grad += power*(self.data**(power - 1))*x.grad
        x.backward = backward
        return x

    def tanh(self):
        e2x = math.exp(2*self.data)
        x = 1 - 2/(e2x+1)
        x = Value(x, (self,), 'tanh')
        def backward():
            self.grad += (1 - x.data**2)*x.grad
        x.backward = backward
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
            v.backward()

    def __repr__(self):
        return f"Data:{self.data}"

class Neuron:
    def __init__(self, input_num):
        self.ws = [Value(random.uniform(-1.0,1.0)) for x in range(input_num)]
        self.bias = Value(random.uniform(-1.0,1.0))

    def forward(self, input_x):
        result = sum((x * y for x, y in zip(self.ws, input_x)),self.bias)
        return result

    def __call__(self, input_x):
        return self.forward(input_x)

    def parameters(self):
        return self.ws + [self.bias]

class Layer:
    def __init__(self, input_num, output_num):
        self.neurons = [Neuron(input_num) for i in range(output_num)]

    def forward(self, input_x):
        result = [neuron(input_x) for neuron in self.neurons]
        return result[0] if len(result)==1 else result

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
            parameter.grad = 0.0

    def update(self, lr):
        for parameter in self.parameters:
            parameter.data -= lr*parameter.grad

