from mytorch import tensor
import numpy as np
import mytorch.nn.functional as F

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    
    temp = []
    sorted_tensor = []
    counter = 0
    sorted_indices = []

    for i in sequence:
        time_step, feature = i.data.shape
        temp.append((time_step, counter))
        counter = counter + 1
    temp.sort(reverse=True) 

    for i in temp:
        sorted_indices.append(i[1])

    for j in range(len(temp)):
        sorted_tensor.append(sequence[sorted_indices[j]])

    final_tensor = []
    batch_size = []
    counter = 0
    total_length = sum(len(x) for x in sorted_tensor) 

    for z in range(sorted_tensor[0].shape[0]):
        counter = 0
        for p in range(len(sorted_tensor)):
            try:
                value = sorted_tensor[p][z]
                counter = counter + 1
                final_tensor.append(value)
            except IndexError:
                pass
        batch_size.append(counter)
    batch_size = np.asarray(batch_size)
    sorted_indices = np.asarray(sorted_indices)

    packed_tensor = F.Cat.apply(*final_tensor, 0)
    packed_tensor = F.Reshape.apply(packed_tensor, (total_length, feature))
    packed_object = PackedSequence(packed_tensor, sorted_indices, batch_size) 

    return packed_object

def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices

    data, indices, batch = (ps.data, ps.sorted_indices, ps.batch_sizes) 
    shape = []
    counter = 0
    temp = 0
    reset = 0
    for i in batch:
        if temp > i:
            shape.append(counter)
            if reset > 1:
                while reset != 1:
                    shape.append(counter)
                    reset = reset - 1
            reset = 0
        else: 
            reset = reset + 1
        temp = i
        counter = counter + 1
    shape.append(counter)
    shape.sort(reverse=True) 

    index = []
    temp = []
    counter = 0
    for i in shape:
        for j in range(i):
            new_index = (np.sum(batch[:j])) + counter
            temp.append(new_index)
        index.append(temp)
        temp = []
        counter = counter + 1

    unpacked = []
    for i in range(len(indices)):
        current = data[index[i]]
        unpacked.append(current)
    
    final = []
    for i in indices:
        final.append(unpacked[i])
    return final


