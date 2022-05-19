import os
import sys
import numpy as np

from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.conv import Conv1d, Flatten
from mytorch.nn.functional import get_conv1d_output_size
from mytorch.nn.linear import Linear
from mytorch.nn.module import Module
from mytorch.nn.sequential import Sequential


class CNN(Module):
    """A simple convolutional neural network.
    """
    def __init__(self):
        super().__init__()
        
        # You'll need these constants for the first layer
        first_input_size = 60 # The width of the input to the first convolutional layer
        first_in_channel = 24 # The number of channels input into the first layer

        # TODO: initialize all layers EXCEPT the last linear layer
        layers = [
            Conv1d(24, 56, 5, stride=1),
            Tanh(),
            Conv1d(56,28,6, stride=2),
            ReLU(),
            Conv1d(28,14,2, stride=2),
            Sigmoid(),
            Flatten()
        ]
        
        # TODO: Iterate through the conv layers and calculate the final output size
        final_size = get_final_conv_output_size(layers, first_input_size)

        # TODO: Append the linear layer with the correct size onto `layers`
        layers.append(Linear(final_size*14,10))

        # TODO: Put the layers into a Sequential
        self.layers = Sequential(layers[0], layers[1], layers[2], layers[3], layers[4], layers[5], layers[6], layers[7])

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channels, input_size)
        Return:
            out (np.array): (batch_size, out_feature)
        """
        # Already completed for you. Passes data through all layers in order.
        return self.layers(x)


def get_final_conv_output_size(layers, input_size):
    """Calculates how the last dimension of the data will change throughout a CNN model
    
    Note that this is the final output size BEFORE the flatten.
    Args:
        layers (list(Module)): List of Conv1d layers, activations, and flatten layers
        input_size (int): input_size of x, the input data 
    """
    # Hint, you may find the function `isinstance()` to be useful.
    for i in layers:
        if isinstance(i,Conv1d):
            output_size = get_conv1d_output_size(input_size, i.kernel_size, i.stride) 
            input_size = output_size
    
    return output_size
    
