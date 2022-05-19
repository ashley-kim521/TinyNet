# TinyNet
Development of my own version of the popular deep learning library PyTorch. A key part of TinyNet is the implementation of Autograd, which is a library for Automatic Differentiation. Autograd is used to automatically calculate the derivative / gradient of any computable function. This includes NNs, which are just big, big functions. It turns out that backpropagation is actually a special case of automatic differentiation. They both do the same thing: calculate partial derivatives.

With autograd, you only need to implement derivatives for simple operations and functions. With those simple operations, you can run most common DL components. The rest of the code is typical Object-Oriented Programing (OOP).


# How Forward Pass Works
During forward propagation, autograd automatically constructs a computational graph. It does this in the background, while the function is being run.
<img width="539" alt="Screen Shot 2022-05-18 at 11 30 34 PM" src="https://user-images.githubusercontent.com/75964687/169198440-7e54793b-fab0-4fc7-aa98-ca6c0a1cde47.png">

The computational graph tracks how elementary operations modify data throughout the function. Starting from the left, you can see how the input variables a and b are first multiplied together,
resulting in a temporary output Op:Mult. Then, a is added to this temporary output, creating d (the
middle node).
Nodes are added to the graph whenever an operation occurs. In practice, we do this by calling the .apply()
method of the operation (which is a subclass of autograd engine.Function). Calling .apply() on the
subclass implicitly calls Function.apply(), which does the following:

# How Backward Pass Works


The structure of the code is the following:
mytorch........................................................................MyTorch library
nn....................................................................Neural Net-related files
activations.py
batchnorm.py
functional.py
linear.py
loss.py
module.py
sequential.py
optim ................................................................. Optimizer-related files
optimizer.py
sgd.py
autograd engine.py.....................................................Autograd main code
tensor.py.....................................................................Tensor object
mnist.py.....................................................Running MLP on MNIST
