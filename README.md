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
<img width="539" alt="Screen Shot 2022-05-18 at 11 30 34 PM" src="https://user-images.githubusercontent.com/75964687/169199189-241e6bf6-bbdf-4d3d-ab6e-e49e46d672c3.png">

In the backward pass, starting at the final node (e), autograd traverses the graph in reverse by performing a recursive **Depth-First Search** (DFS).

Why a DFS? Because it turns out that every computable function can be decomposed into a Directed Acyclic Graph (DAG). A reverse-order DFS on a DAG guarantees at least one valid path for traversing the entire graph in linear time (this is because a reverse-order DFS on a DAG is essentially a Topological Sort). Doing this in reverse is just much more efficient than doing it forwards.

At each recursive call, Autograd calculates one gradient for each input. Each gradient is then passed onto its respective parent, but only if that parent is “gradient-enabled” (requires grad==True). If the parent isn’t, the parent does not get its own gradient/recursive call. Note: constants like the 3 node are not gradient-enabled by default. Eventually, there will be a gradient-enabled node that has no more gradient-enabled parents. For nodes like
these, all we do is store the gradient they just received in their tensor’s .grad.

<img width="851" alt="Screen Shot 2022-05-18 at 11 40 17 PM" src="https://user-images.githubusercontent.com/75964687/169199461-5e092a18-9818-48b7-b5ae-68100eff6421.png">


## Let's verify it with real Torch!
<img width="579" alt="Screen Shot 2022-05-18 at 11 44 34 PM" src="https://user-images.githubusercontent.com/75964687/169199897-74f5b933-b640-425c-913b-dbd57fc7e46f.png">

**Equivalent symbolic math for each of the two gradients**

<img width="523" alt="Screen Shot 2022-05-18 at 11 44 43 PM" src="https://user-images.githubusercontent.com/75964687/169199963-b8d415f8-cf5e-4319-9f19-fb1fe8222190.png">

<img width="584" alt="Screen Shot 2022-05-18 at 11 44 47 PM" src="https://user-images.githubusercontent.com/75964687/169199976-f7c4950e-206c-49d0-ba84-3d3912e8634b.png">


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
