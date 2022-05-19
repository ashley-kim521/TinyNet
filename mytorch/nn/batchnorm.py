from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module
from mytorch.nn import functional as F

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        requires_grad = x.requires_grad
        flag = 0
        mean = x.data.sum(axis=0, keepdims = True)/len(x.data)
        xmu = x.data - mean 
        var = xmu * xmu
        variance = var.sum (axis=0,keepdims = True)/(len(var))
        value = (1 / (variance + self.eps.data))
        normalized = xmu * value.__pow__(0.5) # np.sqrt(value) 

        if (flag == 0):
            self.running_mean = mean
            self.running_var = variance
            flag = flag + 1
        else:    
            if (self.is_train == True):
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            else:
                xmu = x.data
                value = self.running_var+self.epsilon
                normalized = (xmu - self.running_mean) /value.__pow__(0.5) #np.sqrt(self.running_var+self.epsilon)
        y = normalized*self.gamma.data + self.beta.data
        y = Tensor(y, requires_grad=requires_grad,
            is_leaf=not requires_grad)
        return y