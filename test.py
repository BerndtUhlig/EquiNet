import torch
import torch.nn as nn
from sympy import multinomial_coefficients
import numpy as np
from sympy import multinomial_coefficients


from EquiNet import *
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

layer1 = PermutationClosedLayer(5,10,None,False,2)
layer2 = PermutationClosedLayer(10,15,layer1,False,2)
layer3 = PermutationClosedLayer(15,5,layer2,False,2)
seq = nn.Sequential(layer1,layer2,layer3)
print(layer1.weights)
print(layer2.running_weight_matrix)
seq.eval()
with torch.no_grad():
    val = seq(torch.tensor([1.,2.,3.,4.,5.]))
    print(val)
    val2 = seq(torch.tensor([1.,2.,3.,5.,4.]))
    print(val2)

# pcs = PermutationClosedStructureInverse(A)
# add = torch.zeros_like(pcs.weightMatrix)
# # print(A_linv)
# add[0] += torch.tensor([0.1,-0.1,0.2,0.2,0.1,-0.1])
# add[1] += torch.tensor([-0.1,0.1,-0.1,0.1,0.2,0.2])
# add[2] += torch.tensor([0.2,0.2,0.1,-0.1,-0.1,0.1])
#
# print(pcs.weightMatrix)
# print()
# print((pcs.weightMatrix + add) @ A)

# layer1 = PermutationClosedLayer(5,10,None,False,2)
# layer2 = PermutationClosedLayer(10,5,layer1,False,2)
# seq = nn.Sequential(layer1,nn.ReLU(),layer2)
# seq.eval()
# with torch.no_grad():
#     val = seq(torch.tensor([1.,2.,3.,4.,5.]))
#     print(val)
#     val2 = seq(torch.tensor([1.,2.,3.,5.,4.]))
#     print(val2)
#     val3 = seq(torch.tensor([3.,2.,1.,5.,4.]))
#     print(val3)