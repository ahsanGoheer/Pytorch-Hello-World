import torch 
import numpy as np 

#Tensor Conversions.

data = [[1,2],[3,4]]

numpy_array= np.array(data) #Converting to numpy array.


tensor = torch.tensor(data) #Converting List to Tensor.


tensor_from_np=torch.from_numpy(numpy_array) # Converting numpy array to tensor.

print(tensor)
print(tensor_from_np)


## Converting Tensors from Other Tensors.

tensor_replicated = torch.ones_like(tensor) # Tensor with all ones.

print(tensor_replicated)

tensor_replicated_rand=torch.rand_like(tensor_from_np,dtype=torch.float) #Tensor with random values.

print(tensor_replicated_rand)


## Generating Tensors based on shape.

ones_shape= (2,2)

ones_tensor = torch.ones(ones_shape)

rand_shape=(4,6)

rand_tensor = torch.rand(rand_shape)

print(ones_tensor)
print(rand_tensor)


## Tensor Attributes 

print(ones_tensor.shape)
print(ones_tensor.dtype)
print(ones_tensor.device)


## Moving Tensor to GPU

if torch.cuda.is_available():
    tensor=ones_tensor.to('cuda')

print(tensor.device)

## Tensor and numpy array on cpu share memory locations.

tensor_from_np.add_(1) # Adds a number to each element of the tensor.

print(tensor_from_np)
print(numpy_array)