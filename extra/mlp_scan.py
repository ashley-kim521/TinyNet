import numpy as np

from mytorch.nn.activations import ReLU
from mytorch.nn.conv import Conv1d, Flatten
from mytorch.nn.linear import Linear
from mytorch.nn.sequential import Sequential
from mytorch.tensor import Tensor


class CNN_SimpleScanningMLP:
    """CNN as a Simple Scanning MLP

    Complete the indicated methods to convert the linear MLP described
    in the assignment sheet into a Simple 1D CNN.
    """
    def __init__(self):
        # TODO: Initialize Conv1d layers with appropriate params (this is the hard part)
        # For reference, here's the arguments for Conv1d:
        #            Conv1d(in_channel, out_channel, kernel_size, stride)
        self.conv1 = Conv1d(24, 8, 8, 4)
        self.conv2 = Conv1d(8, 16, 1, 1)
        self.conv3 = Conv1d(16, 4, 1, 1)
     
        # TODO: Initialize Sequential object with layers based on the MLP architecture.
        # Note: Besides using Conv1d instead of Linear, there is a slight difference in layers.
        #       What's the difference and why?
        self.layers = Sequential(self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten())

    def init_weights(self, weights):
        """Converts the given 3 weight matrices of the linear MLP into the weights of the Conv layers.
        Args:
            weights (tuple(np.array)): shapes ((8, 192), (16, 8), (4, 16))
                                       Think of each as a Linear.weight.data, shaped (out_features, in_features)
        """
        # TODO: Convert the linear weight arrays into Conv1d weight tensors
        # Make sure to not add nodes to the comp graph!
        w1, w2, w3 = weights # Here, we've unpacked them into separate arrays for you.
        
        # Assume the Conv1d weight tensors are already initialized with the params that you specified in __init__().
        # Your job now is to replace those weights with the MLP's weights.
        
        # Tip: You can automatically retrieve the Conv1d params like so:
        #      ex) self.conv1.out_channel, self.conv1.kernel_size, self.conv1.in_channel

        # Set the weight tensors with your converted MLP weights
        # 1. Reshape each Linear weight matrix into (out_channel, kernel_size, in_channel).
        # 2. Transpose the axes back into these shapes: (out_channel, in_channel, kernel_size). 
        #       (Note: .T will not work for this step)

        w1 = np.reshape(w1, (8, 8, 24))
        temp = np.transpose(w1, (0, 2, 1))
        requires_grad = self.conv1.weight.requires_grad
        self.conv1.weight = Tensor(temp, requires_grad=requires_grad, is_leaf=True, is_parameter=True)

        w2 = np.reshape(w2, (16, 1, 8))
        temp = np.transpose(w2, (0, 2, 1))
        requires_grad = self.conv2.weight.requires_grad
        self.conv2.weight = Tensor(temp, requires_grad=requires_grad, is_leaf=True, is_parameter=True)

        w3 = np.reshape(w3, (4, 1, 16))
        temp = np.transpose(w3, (0, 2, 1))
        requires_grad = self.conv3.weight.requires_grad
        self.conv3.weight = Tensor(temp, requires_grad=requires_grad, is_leaf=True, is_parameter=True)

    def forward(self, x):
        """Do not modify this method

        Args:
            x (Tensor): (batch_size, in_channel, in_features)
        Returns:
            Tensor: (batch_size, out_channel, out_features)
        """
        return self.layers(x)

    def __call__(self, x):
        """Do not modify this method"""
        return self.forward(x)


class CNN_DistributedScanningMLP:
    """CNN as a Distributed Scanning MLP

    Complete the indicated methods to convert the linear MLP described
    in the assignment sheet into a Distributed 1D CNN."""
    def __init__(self):
        # TODO: Initialize Conv1d layers
        # For reference, here's the arguments for Conv1d:
        #            Conv1d(in_channel, out_channel, kernel_size, stride)
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8, 4, 2, 1)

        # TODO: Initialize Sequential object
        self.layers = Sequential(self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten())

    def __call__(self, x):
        """Do not modify this method"""
        return self.forward(x)

    def init_weights(self, weights):
        """Use the 3 weight matrices of linear MLP to init the weights of the CNN.
        Args:
            weights (tuple(np.array)): shapes ((8, 192), (16, 8), (4, 16))
                                       Think of each as a Linear.weight.data, shaped (out_features, in_features)
        """
        w1, w2, w3 = weights

        # TODO: Convert the linear weights into Conv1d weights
        # Make sure to not add nodes to the comp graph!

        temp = np.array([w1[0, :48], w1[1, :48]]) #in_channel, out_channel: 24,2
        temp = np.reshape(temp, (2, 2, 24)) 
        temp = np.transpose(temp, (0, 2, 1))
        requires_grad = self.conv1.weight.requires_grad
        self.conv1.weight = Tensor(temp, requires_grad = requires_grad, is_leaf=True, is_parameter=True)

        #for all filters, slice 0 to 4
        temp = np.array([w2[0, 0:2*2], w2[1, 0:4], w2[2, 0:4], w2[3, 0:4], w2[4, 0:4], #in_channel, out_channel: 2, 8
                                                  w2[5, 0:4], w2[6, 0:4], w2[7, 0:4]])
        temp = np.reshape(temp, (8, 2, 2))
        temp = np.transpose(temp, (0, 2, 1))
        requires_grad = self.conv2.weight.requires_grad
        self.conv2.weight = Tensor(temp, requires_grad = requires_grad, is_leaf=True, is_parameter=True)

        #slice all of them --> no need to slice: in_channel, out_channel: 8, 4
        temp = np.reshape(w3, (4, 2, 8)) 
        temp = np.transpose(temp, (0, 2, 1))
        requires_grad = self.conv3.weight.requires_grad
        self.conv3.weight = Tensor(temp, requires_grad = requires_grad, is_leaf=True, is_parameter=True)


    def forward(self, x):
        """Already completed for you.
        Args:
            x (Tensor): (batch_size, in_channel, in_features)
        Returns:
            out (Tensor): (batch_size, out_channel, out_features)
        """
        return self.layers(x)
