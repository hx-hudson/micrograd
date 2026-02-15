from engine import Conv2dLayer, max_pooling, LinearLayer

class CNN:
    def __init__(self):
        """
        input: (batch, 1, 28, 28)
        conv1: (batch, 4, 28, 28)
        max_pool: (batch, 4, 14, 14)
        conv2: (batch, 8, 14, 14)
        max_pool: (batch, 8, 7, 7)
        linear: (batch, 392)
        """
        self.conv1 = Conv2dLayer(1, 4, 3, padding=1)
        self.conv2 = Conv2dLayer(4, 8, 3, padding=1)
        self.linear = LinearLayer(392, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = y.relu()
        y = max_pooling(y, 2)
        y = self.conv2(y)
        y = y.relu()
        y = max_pooling(y, 2)
        batch = y.data.shape[0]
        y = y.reshape(batch, -1)
        y = self.linear(y)
        return y

    def parameters(self):
        return self.conv1.parameters()+self.conv2.parameters()+self.linear.parameters()

    def __call__(self, x):
        return self.forward(x)
