# TinyNet
Development of my own version of the popular deep learning library PyTorch. A key part of TinyNet is the implementation of Autograd, which is a library for Automatic Differentiation. Autograd is used to automatically calculate the derivative / gradient of any computable function. This includes NNs, which are just big, big functions. It turns out that backpropagation is actually a special case of automatic differentiation. They both do the same thing: calculate partial derivatives.

With autograd, you only need to implement derivatives for simple operations and functions. With those simple operations, you can run most common DL components. The rest of the code is typical Object-Oriented Programing (OOP).

## How Forward Pass Works
During forward propagation, autograd automatically constructs a computational graph. It does this in the background, while the function is being run.

<img width="539" alt="Screen Shot 2022-05-18 at 11 30 34 PM" src="https://user-images.githubusercontent.com/75964687/169198440-7e54793b-fab0-4fc7-aa98-ca6c0a1cde47.png">

The computational graph tracks how elementary operations modify data throughout the function. Starting from the left, you can see how the input variables a and b are first multiplied together, resulting in a temporary output Op:Mult. Then, a is added to this temporary output, creating d (the middle node). Nodes are added to the graph whenever an operation occurs. In practice, we do this by calling the .apply() method of the operation (which is a subclass of autograd engine.Function). Calling .apply() on the subclass implicitly calls Function.apply(), which does the following:
1. Create a node object for the operation’s output
2. Run the operation on the input(s) to get an output tensor
3. Store information on the node, which links it to the comp graph
4. Store the node on the output tensor
5. Return the output tensor

**But what information is stored on the node, and how does it link the node to the comp graph?** 
The node stores two things: the operation that created the node’s data and a record of the node’s “parents”. Recall that each tensor stores its own node. So when making the record of the current node’s parents, we
can usually just grab the nodes from the input tensors. But also recall that we only create nodes when an operation is called. Op:Mult was the very first operation, so its input tensors a and b didn’t even have nodes initialized for them yet.

To solve this issue, Function.apply() also checks if any parents need to have their nodes created for them. If so, it creates the appropriate type of node for that parent (we’ll introduce node types during backward), and then adds that node to its list of parents. Effectively, this connects the current node to its parent nodes in the graph.

**To recap, whenever we have an operation on a tensor in the comp graph, we create a node, get the output of the operation, link the node to the graph, and store the node on the output tensor.**

**But why are we making this graph?**
Remember: we’re doing all of this to calculate gradients. Specifically, the partial gradients of the loss w.r.t. each gradient-enabled tensor (any tensor with requires grad==True). Think of the graph as a trail of breadcrumbs that keeps track of where we’ve been. It’s used to retrace our steps back through the graph during backprop. This is where ”backpropagation” gets its name: both autograd/backprop traverse graphs backwards while calculating gradients.

## How Backward Pass Works


## Code Structure
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
