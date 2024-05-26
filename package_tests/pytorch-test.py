# Python program using PyTorch 
# for defining tensors, fitting a 
# two-layer network to random 
# data, and calculating the loss 

import torch 

dtype = torch.float
#device = torch.device("cpu") 
#device = torch.device("cuda:0") #Uncomment this to run on GPU 
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Running on GPUs")
else:
    device = torch.device("cpu")
    print("Running on CPU")


# N is batch size; D_in is input dimension; 
# H is hidden dimension; D_out is output dimension. 
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data 
x = torch.randn(N, D_in, device=device, dtype=dtype) 
y = torch.randn(N, D_out, device=device, dtype=dtype) 

# Randomly initialize weights 
w1 = torch.randn(D_in, H, device=device, dtype=dtype) 
w2 = torch.randn(H, D_out, device=device, dtype=dtype) 

learning_rate = 1e-6
for t in range(500): 
    # Forward pass: compute predicted y 
    h = x.mm(w1) 
    h_relu = h.clamp(min=0) 
    y_pred = h_relu.mm(w2) 

    # Compute and print loss 
    loss = (y_pred - y).pow(2).sum().item() 
    print(t, loss) 

    # Backprop to compute gradients of
