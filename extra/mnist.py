import numpy as np
import mytorch.nn.loss as loss
import mytorch.optim.sgd as sgd
import mytorch.nn.sequential as sequential
import mytorch.nn.linear as linear
import mytorch.nn.activations as activations
import mytorch.tensor as tensor
import mytorch.nn.functional as F

# TODO: Import any mytorch packages you need (XELoss, SGD, etc)

# NOTE: Batch size pre-set to 100. Shouldn't need to change.
BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    """Initialize objects and start training
    You won't need to call this function yourself.
    (Data is provided by autograder)
    
    Args:
        train_x (np.array): training data (55000, 784) 
        train_y (np.array): training labels (55000,) 
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        val_accuracies (list(float)): List of accuracies per validation round
                                      (num_epochs,)
    """
    # TODO: Initialize an MLP, optimizer, and criterion
    criterion = loss.CrossEntropyLoss()
    model = sequential.Sequential(linear.Linear(784, 20), activations.ReLU(), linear.Linear(20,10))
    optimizer = sgd.SGD(model.parameters(), lr=0.1)

    # TODO: Call training routine (make sure to write it below)
    val_accuracies = train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3)
    return val_accuracies

def split_data_into_batches(x, y):
    batch_data = np.array_split(x, BATCH_SIZE)
    batch_labels = np.array_split(y, BATCH_SIZE)
    # requires_grad = False
    # data = tensor.Tensor(batch_data, requires_grad = requires_grad, is_leaf=not requires_grad)
    # label = tensor.Tensor(batch_labels, requires_grad = requires_grad, is_leaf=not requires_grad)
    # batch_data = [x[i:i+BATCH_SIZE] for i in range(0, len(x), BATCH_SIZE)]
    # batch_labels = [y[i:i+BATCH_SIZE] for i in range(0, len(y), BATCH_SIZE)]
    # requires_grad = False
    # data = tensor.Tensor(batch_data, requires_grad = requires_grad, is_leaf=not requires_grad)
    # label = tensor.Tensor(batch_labels, requires_grad = requires_grad, is_leaf=not requires_grad)
    return zip(batch_data, batch_labels)

def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=3):
    """Training routine that runs for `num_epochs` epochs.
    Returns:
        val_accuracies (list): (num_epochs,)
    """
    val_accuracies = []
    model.train()
    for i in range(num_epochs):
        index = [i for i in range(len(train_x))]
        np.random.shuffle(index)
        train_x_new  = train_x[index, :]
        train_y_new = train_y[index,]

        batches = split_data_into_batches(train_x_new, train_y_new)
        requires_grad = False
        # acc = 0
        # tot = 0
        for i, (data, label) in enumerate(batches):
            x = tensor.Tensor(data, requires_grad = requires_grad, is_leaf=not requires_grad)
            y = tensor.Tensor(label, requires_grad = requires_grad, is_leaf=not requires_grad)
            optimizer.zero_grad() # clear any previous gradients
            out = model.forward(x)
            #print(out.data[0])
            # acc += (np.argmax(out.data, axis=1) == label).sum()
            # tot += len(label)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step() # update weights with new gradients
            if (i % 100 == 0):
                accuracy = validate(model, val_x, val_y)
                val_accuracies.append(accuracy)
                model.train()
    return val_accuracies

def validate(model, val_x, val_y):
    """Validation routine, tests on val data, scores accuracy
    Relevant Args:
        val_x (np.array): validation data (5000, 784)
        val_y (np.array): validation labels (5000,)
    Returns:
        float: Accuracy = correct / total
    """
    model.eval()
    batches = split_data_into_batches(val_x, val_y)
    num_correct = 0
    requires_grad = False
    for (batch_data, batch_labels) in batches:
        x = tensor.Tensor(batch_data, requires_grad = requires_grad, is_leaf=not requires_grad)
        y = tensor.Tensor(batch_labels, requires_grad = requires_grad, is_leaf=not requires_grad)
        out = model.forward(x)
        for i in range(50):
            pred_idx = np.argmax(out.data[i])
            true_idx = batch_labels[i]
            if pred_idx == true_idx:
                num_correct += 1
    return (num_correct / 5000)

