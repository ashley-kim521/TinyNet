import numpy as np

# import mytorch.tensor as tensor
from mytorch import tensor
from mytorch.autograd_engine import Function

def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                                    is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T), 

class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                                                 is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data),

"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""
class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis = axis, keepdims = keepdims), \
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        #print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        if a.data.shape != b.data.shape:
            raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + -b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * -grad_output.data

        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        
        # calculate gradient of output w.r.t. each input
        grad_a = grad_output.data * (b.data)
        grad_b = (a.data) * grad_output.data
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.dot(a.data, b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        
        a, b = ctx.saved_tensors
        
        # calculate gradient of output w.r.t. each input
        grad_a = np.dot(grad_output.data, (b.data.T))
        grad_b = np.dot((a.data.T), grad_output.data)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / (b.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        grad_a = grad_output.data * (1/b.data)
        grad_b = (-a.data/b.data) * (grad_output.data/b.data)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)

class ReLu(Function):
    @staticmethod
    def forward(ctx, y):
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(y)
        requires_grad = y.requires_grad
        maxi = y.data * (y.data > 0) 
        c = tensor.Tensor(maxi, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        q = grad_output.data * (y.data > 0) 
        return tensor.Tensor(q),

class NEG(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        requires_grad = x.requires_grad
        c = tensor.Tensor(-x.data, requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        # calculate gradient of output w.r.t. each input
        q = (-x.data) * grad_output.data
        return tensor.Tensor(q), 

class EXP(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        requires_grad = x.requires_grad
        c = tensor.Tensor(np.exp(x.data), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        # calculate gradient of output w.r.t. each input
        q = np.exp(x.data) * grad_output.data
        return tensor.Tensor(q), 

class POW(Function):
    @staticmethod
    def forward(ctx, x,y):
        ctx.save_for_backward(x)
        ctx.constant = y
        requires_grad = x.requires_grad 
        c = tensor.Tensor(np.power(x.data, y), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        y = ctx.constant
        # calculate gradient of output w.r.t. each input
        q = grad_output.data * y * (x.data ** (y - 1))
        return tensor.Tensor(q), 

class cross_entropy(Function):
    @staticmethod
    def forward(ctx, predicted, target):
        ctx.save_for_backward(predicted, target)

        requires_grad = predicted.requires_grad or target.requires_grad

        batch_size, num_classes = predicted.shape
        ones = np.ones(predicted.shape) 
        one_hot = to_one_hot(target, num_classes) #one hot of y 
    
        x_n = np.sum(np.exp(predicted.data), axis=1, keepdims = True)
        x_c = np.exp(predicted.data)
        logsoftmax = np.log(x_c/x_n)

        Neg_one_hot = np.negative(one_hot.data)
        logsoftmax = tensor.Tensor(logsoftmax)
        multiplied = logsoftmax.data*Neg_one_hot.data

        NLLLoss = np.sum(multiplied) 
        divided = NLLLoss / batch_size

        c = tensor.Tensor(divided, requires_grad=requires_grad,
                                is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        one_hot = to_one_hot(b, a.shape[1]) #one hot of y 
        batch_size = a.shape[0]

        grad = np.exp(a.data) / np.sum(np.exp(a.data), axis=-1, keepdims=True)
        numbers = np.arange(batch_size)
        grad[numbers, b.data] -= 1
        grad = grad/batch_size
    
        return tensor.Tensor(grad), 


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad = True)

class Dropout(Function):
    @staticmethod
    def forward(ctx, x, p=0.5, is_train=False):
        """Forward pass for dropout layer.

        Args:
            ctx (ContextManager): For saving variables between forward and backward passes.
            x (Tensor): Data tensor to perform dropout on
            p (float): The probability of dropping a neuron output.
                       (i.e. 0.2 -> 20% chance of dropping)
            is_train (bool, optional): If true, then the Dropout module that called this
                                       is in training mode (`<dropout_layer>.is_train == True`).
                                       
                                       Remember that Dropout operates differently during train
                                       and eval mode. During train it drops certain neuron outputs.
                                       During eval, it should NOT drop any outputs and return the input
                                       as is. This will also affect backprop correspondingly.
        """
        requires_grad = x.requires_grad
        U = np.random.binomial(1, 1-p, size=x.shape)
        if is_train==True:
            H = x.data* U * (1/(1-p))
            c = tensor.Tensor(H, requires_grad=requires_grad,
                is_leaf=not requires_grad)

            temp = U * (1/(1-p))
            h = tensor.Tensor(temp, requires_grad=requires_grad,
                is_leaf=not requires_grad)
            ctx.save_for_backward(h)

            return c
        else:
            h = tensor.Tensor(U, requires_grad=requires_grad,
                is_leaf=not requires_grad)
            ctx.save_for_backward(h)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        u = ctx.saved_tensors[0]
        output = u.data * grad_output.data 
        return tensor.Tensor(output), 

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad),
    
class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad),


def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.
        
        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    size_ = ((input_size - kernel_size)//stride)+1
    return size_
    
class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.
        
        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.
        
        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution
        
        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape #2,6,61
        out_channel, _, kernel_size = weight.shape #11,6, 6
        
        # TODO: Save relevant variables for backward pass
        requires_grad = x.requires_grad
        ctx.save_for_backward(x, weight, bias)
        ctx.constant = stride

        #Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride) #14

        # TODO: Initialize output with correct size

        out = np.zeros((batch_size, out_channel, output_size)) #2,11,14

        # TODO: Calculate the Conv1d output.
        for batch in range(batch_size): #2
            for p in range(out_channel): #11
                for q in range(output_size): #14
                    temp = x.data[batch, :, q*stride:q*stride+kernel_size] * weight.data[p]
                    temp = np.sum(temp) + bias.data[p]
                    out[batch, p, q] = temp

        # TODO: Put output into tensor with correct settings and return 
        out = tensor.Tensor(out, requires_grad=requires_grad, is_leaf=not requires_grad)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        batch_size, in_channel, input_size = x.shape 
        out_channel, _, kernel_size = weight.shape 
        stride = ctx.constant
        output_size = get_conv1d_output_size(input_size, kernel_size, stride) #14
        #grad_output size: (batch_size, out_channel, output_size)  (2,11,14)

        dx = np.zeros(x.shape) #2,6,61
        dweight= np.zeros(weight.shape) #11,6,6
        dbias = np.zeros(bias.shape) #11,

        for batch in range(batch_size): #2
            for c in range(out_channel): #11
                for q in range(output_size): #14
                    dweight[c] += grad_output.data[batch, c, q] * x.data[batch, :, q*stride:q*stride+kernel_size] 
                    dx[batch, :, q*stride:q*stride+kernel_size] += (grad_output.data[batch, c, q] * weight.data[c,:,:])

            dbias += np.sum(grad_output.data[batch, :, :], axis=1)
 
        return tensor.Tensor(dx), tensor.Tensor(dweight), tensor.Tensor(dbias), None

class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        value = x.data[indices]
        ctx.save_for_backward(x)
        ctx.constant = indices
        requires_grad = x.requires_grad
        output = tensor.Tensor(value, requires_grad=requires_grad, is_leaf=not requires_grad)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        x = ctx.saved_tensors[0]
        indices = ctx.constant
        output = np.zeros(x.shape)
        output[indices] = grad_output.data
        
        return tensor.Tensor(output),
    
class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            dim (int): The dimension along which we concatenate our tensors
            seq (list of tensors): list of tensors we wish to concatenate
        '''
        *seq, dim = args

        # Save inputs to access later in backward pass.
        ctx.constant = dim
        ctx.seq = seq
        counter = 0
        requires_grad = False

        list_ = []
        for i in seq:
            shape = i.data.shape
            list_.append(i.data)
            if counter > 0:
                for q in range(len(shape)):
                    if q == dim:
                        pass
                    else:
                        # print(q)
                        assert shape[q] == temp[q] or shape[q] == 0
            requires_grad = i.requires_grad or requires_grad
            counter = counter + 1
            temp = i.data.shape
        
        c = tensor.Tensor(np.concatenate(list_, axis=dim), requires_grad=requires_grad,
                                           is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx,grad_output):
        dim = ctx.constant 
        seq = ctx.seq
        counter = 0
        list_ = []
        for i in seq:
            shape = i.data.shape
            slicing = shape[dim]
            indices = []
            for q in range(counter,counter+slicing):
                indices = np.append(indices, int(q))
            returning = np.take(grad_output.data, [indices], axis = dim) 
            counter = counter + slicing

            returning = returning.reshape(shape)
            tensor_output = tensor.Tensor(returning)
            list_.append(tensor_output)

        return list_