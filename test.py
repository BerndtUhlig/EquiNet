import torch
import torch.nn as nn
from sympy import multinomial_coefficients
import numpy as np
from sympy import multinomial_coefficients
from torch.utils.data import TensorDataset, DataLoader

from EquiNet import *
from torch.optim import Adam
#
# weightTensor = torch.randn(5)
# print(weightTensor)
# weightList = weightTensor.tolist()

# t1 = torch.tensor([1,2,3])
# t2 = torch.tensor([3,4,5])
# t3 = [t1,t2]
# t4 = torch.vstack(t3)
#
# A = torch.tensor([[1.,2.,3.],[2.,1.,3.],[3.,2.,1.],[3.,1.,2.],[1.,3.,2.],[2.,3.,1.]])
# reference = A[0]
# indices = np.array([[np.where(reference == element)[0][0] for element in row] for row in A])
# print(indices)
# print(A)
# A_linv = torch.tensor(np.linalg.inv((A.T @ A).numpy()) @ A.T.numpy())

# print(add)
# A_linv += add
#
# print(A_linv @ A)

# A = torch.tensor([[1.,2.,3.],[2.,1.,3.],[3.,2.,1.],[3.,1.,2.],[1.,3.,2.],[2.,3.,1.]])
#
# val = PermutationClosedStructure(10, [9,1])
# print(len(val.weightMatrix))
# print(val.weightMatrix)

def argmax_one_hot(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    idx = x.argmax(dim=dim, keepdim=True)
    return torch.zeros_like(x,dtype=torch.float32).scatter_(dim,idx,1.0)

def run_experiment(model, data, num_epochs=10000, lr=1e-3, device="cpu", size=10):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        model.train()

        for xb, yb in data:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            lossval = criterion(logits, yb)
            lossval.backward()
            optimizer.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                pred = model(xb)
                acc = (pred == yb.to(device)).float().mean()
                print(f"epoch {epoch:03d}  loss {lossval.item()}  acc {acc:.2%}")

    #
    model.eval()
    with torch.no_grad():
        vec = torch.randn(size)



        vec[0] = 10
        vec[1] = -2
        ver = argmax_one_hot(vec, 0)
        print(f"Value before Permutation: {vec}")
        print(f"Correct one hot for argmax: {ver}")
        val = model(vec)
        print(f"Output: {val}")
        vec[0] = -2
        vec[1] = 10
        print(f"Value after Permutation: {vec}")
        val2 = model(vec)
        print(f"Output: {val2}")

device = "cuda" if torch.cuda.is_available() else "cpu"

d = 100       # dimensionality
N = 1000  # samples
x = torch.randn(N, d)     # e.g. N(0,1) inputs
y = argmax_one_hot(x,1) # integer labels: argmax index
print(x)
print(y)



layer1 = PermutationClosedLayer(d ,2*d ,None,False,2)
layer2 = PermutationClosedLayer(2*d ,d ,layer1,False,2)
seq = nn.Sequential(layer1,layer2,nn.Softmax()).to(device)
seq_verify = nn.Sequential(nn.Linear(d ,2*d ),nn.Linear(2*d ,d),nn.Softmax()).to(device)

torch.manual_seed(42)


num_epochs = 15000
batch_size = 16
train_batches = [(x.to(device), y.to(device))]
train_set = TensorDataset(x, y)

run_experiment(seq,train_batches,num_epochs, 1e-3, device,d )
torch.manual_seed(42)
run_experiment(seq_verify,train_batches,num_epochs,1e-3,device,d)
# torch.manual_seed(42)
# d = 100       # dimensionality
# N = 10000              # samples
# x = torch.randn(N, d)     # e.g. N(0,1) inputs
# y = argmax_one_hot(x,1) # integer labels: argmax index
# train_batches = [(x.to(device), y.to(device))]
# run_experiment(seq_verify,train_batches,num_epochs,1e-3,device)
