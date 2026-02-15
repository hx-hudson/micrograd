import random
import numpy as np


def unbroadcast(grad, shape):
    if not isinstance(grad, np.ndarray):
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

def conv2d(x, w, b=None, stride=1, padding=0):
    p = padding
    if b is not None:
        bias = Value(b) if not isinstance(b, Value) else b
        assert bias.data.ndim == 1
    x = Value(x) if not isinstance(x, Value) else x
    weight = Value(w) if not isinstance(w, Value) else w
    assert x.data.ndim == 4 and weight.data.ndim == 4
    assert weight.data.shape[1] == x.data.shape[1]

    x_padding = np.pad(x.data, ((0,0),(0,0),(p,p),(p,p))) if p!=0 else x.data
    x_padding_shape = np.shape(x_padding)
    w_shape = np.shape(weight.data)

    first_dim_length = int((x_padding_shape[-2] - w_shape[-2])//stride) + 1
    second_dim_length = int((x_padding_shape[-1] - w_shape[-1])//stride) + 1
    w_out_channel = w_shape[0]
    y_data = np.zeros((x_padding_shape[0],w_out_channel,first_dim_length,second_dim_length))
    for c in range(w_out_channel):
        sub_weight = weight.data[c,...]
        for i in range(first_dim_length):
            for j in range(second_dim_length):
                sub_x = x_padding[..., i*stride : i*stride + w_shape[-2], j*stride : j*stride + w_shape[-1]]
                sub_sum = (sub_x * sub_weight).sum(axis=(1, 2, 3))
                y_data[:, c, i, j] = sub_sum
    if b is not None:
        y = Value(y_data+bias.data[None, :, None, None], (weight, x, bias), "conv2d")
    else:
        y = Value(y_data, (weight, x), "conv2d")


    def backward():
        padding_grad = np.zeros_like(x_padding)
        for c2 in range(w_out_channel):
            sub_w_data = weight.data[c2, ...]
            for i2 in range(first_dim_length):
                for j2 in range(second_dim_length):
                    grad = y.grad[:, c2, i2, j2]
                    sub_x2 = x_padding[:, :, i2*stride : i2*stride + w_shape[-2], j2*stride : j2*stride + w_shape[-1]]
                    weight.grad[c2, ...] += (sub_x2 * grad[:, None, None, None]).sum(axis=0)
                    padding_grad[:, :, i2*stride : i2*stride + w_shape[-2], j2*stride : j2*stride + w_shape[-1]] += sub_w_data[None, ...] * grad[:, None, None, None]
        if b is not None:
            bias.grad += y.grad.sum(axis=(0,2,3))
        x.grad += padding_grad[:, :, p:-p, p:-p] if p!=0 else padding_grad

    y._backward = backward
    return y

def max_pooling(x, stride):
    # If multiple elements share the maximum value,
    # the full upstream gradient is propagated to each of them.
    x = Value(x) if not isinstance(x, Value) else x
    batch, channel, x_length, x_width = np.shape(x.data)
    y_length, y_width = (x_length//stride, x_width//stride)
    y_data = np.zeros((batch, channel, y_length, y_width))
    mask = np.zeros_like(x.data, dtype=bool) # use bool to save memory

    for i in range(y_length):
        for j in range(y_width):
            sub_data = x.data[:, :, i*stride:i*stride+stride, j*stride:j*stride+stride]
            y_data[:, :, i, j] = np.max(sub_data,axis=(2,3))
            sub_y_data = y_data[:, :, i, j]
            y_data_expanded = sub_y_data[:,:,None,None].repeat(stride, axis=2).repeat(stride, axis=3)
            mask[:, :, i*stride:i*stride+stride, j*stride:j*stride+stride] = (y_data_expanded == sub_data)
    y = Value(y_data, (x,), "max_pool")
    def backward():
        y_grad_expanded = y.grad.repeat(stride, axis=2).repeat(stride,axis=3)
        x.grad += mask * y_grad_expanded
    y._backward = backward
    return y

def cross_entropy(x, y):
    x = Value(x) if not isinstance(x, Value) else x
    batch, class_num = np.shape(x.data)
    y = np.asarray(y, dtype=int)

    # prevent overflow
    x_data = x.data
    data_shifted = x_data - np.max(x_data, axis=1, keepdims=True)
    # softmax
    data_exp = np.exp(data_shifted)
    data_softmax = data_exp / np.sum(data_exp, axis=1, keepdims=True)

    y_prob = data_softmax[np.arange(batch), y]
    loss_data = -np.mean(np.log(y_prob+1e-12))
    loss = Value(loss_data, (x,), "cross_entropy")
    def backward():
        p = data_softmax.copy()
        p[np.arange(batch), y] -= 1.0
        x.grad += p / batch * loss.grad

    loss._backward = backward
    return loss

class Conv2dLayer:
    def __init__(self, in_channel, out_channel, kernel_size, bias=True, stride=1, padding=0):
        self.weight = Value(0.01 * np.random.randn(out_channel,in_channel,kernel_size,kernel_size))
        if bias:
            self.bias = Value(np.zeros(out_channel))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        y = conv2d(x, self.weight, self.bias, self.stride, self.padding)
        return y

    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]

    def __call__(self, x):
        return self.forward(x)

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

    def relu(self):
        y_data = np.maximum(0, self.data)
        y = Value(y_data, (self,), "relu")
        def backward():
            self.grad += (self.data > 0) * y.grad
        y._backward = backward
        return y

    def exp(self):
        y_data = np.exp(self.data)
        y = Value(y_data, (self,), "exp")
        def backward():
            self.grad += y_data * y.grad
        y._backward = backward
        return y

    def log(self, eps = 1e-12):
        y_data = np.log(self.data)
        y = Value(y_data, (self,), "log")
        def backward():
            self.grad += y.grad / (self.data + eps)
        y._backward = backward
        return y

    def __matmul__(self, other):
        other = Value(other) if isinstance(other, (int, float, np.ndarray, np.number)) else other
        x = Value(self.data @ other.data, (self, other), "@")
        def backward():
            self_dim = np.ndim(self.data)
            other_dim = np.ndim(other.data)
            if self_dim == 1 and other_dim ==1:
                self.grad += x.grad * other.data
                other.grad += x.grad * self.data
                return

            self_matrix, other_matrix, x_matrix = self.data, other.data, x.grad
            if self_dim == 1:
                self_matrix = self.data[..., None, :]
                x_matrix = x.grad[..., None, :]
            if other_dim == 1:
                other_matrix = other.data[..., :, None]
                x_matrix = x.grad[..., :, None]

            d_self = np.matmul(x_matrix, np.swapaxes(other_matrix, -1, -2))
            d_other = np.matmul(np.swapaxes(self_matrix, -1, -2), x_matrix)

            if self_dim == 1:
                d_self = np.squeeze(d_self, axis=-2)
            if other_dim == 1:
                d_other = np.squeeze(d_other, axis=-1)

            self.grad += unbroadcast(d_self, np.shape(self.data))
            other.grad += unbroadcast(d_other, np.shape(other.data))
        x._backward = backward
        return x

    def __rmatmul__(self, other):
        return Value(other) @ self

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
        if isinstance(self.data, np.ndarray):
            self.grad = np.ones_like(self.data, dtype=float)
        else:
            self.grad = 1.0
        for v in reversed(node):
            v._backward()

    def sum(self):
        x = Value(self.data.sum(),(self,),"sum")
        def backward():
            self.grad +=  np.ones_like(self.data, dtype=float) * x.grad
        x._backward = backward
        return x

    def reshape(self, *shape):
        x = Value(self.data.reshape(*shape), (self,), 'reshape')
        def backward():
            self.grad += x.grad.reshape(self.data.shape)
        x._backward = backward
        return x

    def __repr__(self):
        return f"Data:{self.data}"

class Neuron:
    def __init__(self, input_num):
        self.ws = Value(np.random.randn(input_num))
        self.bias = Value(random.uniform(-1.0,1.0))

    def forward(self, x):
        result = x @ self.ws + self.bias
        return result.tanh()

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.ws, self.bias]

class LinearLayer:
    def __init__(self, input_num, output_num):
        self.weights = Value(0.01 * np.random.randn(input_num, output_num))
        self.bias = Value(np.zeros(output_num))

    def forward(self, x):
        y = x @ self.weights + self.bias
        return y

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [self.weights, self.bias]

class MLP:
    def __init__(self, input_num, layer_num:list):
        length = len(layer_num)
        sizes = [input_num] + list(layer_num)
        self.layers = [LinearLayer(sizes[i],sizes[i+1]) for i in range(length)]

    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i<len(self.layers)-1:
                y = y.tanh()
        return y.reshape(-1, 1)

    def __call__(self, x):
        return self.forward(x)

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

