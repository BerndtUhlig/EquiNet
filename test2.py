import torch
from sympy.utilities.iterables import multiset_permutations
import numpy as np
import torch.nn.functional as F
#
# weightTensor = torch.randn(2,3,3)
# list_weights = weightTensor.tolist()
# print(list_weights)
# print(weightTensor)
# print(weightTensor[1][0])

# n_list = [5,1]
# weightTensor = torch.randn(2,3,4)
# weightList = weightTensor.tolist()
# new_tensor_data = []
# for i in range(len(n_list)):
#     for b in range(n_list[i]):
#         new_tensor_data.append(i)
# val = list(multiset_permutations(new_tensor_data))
# indices = torch.tensor(val)
# reference = weightList
# print(indices)
# print(weightTensor[indices])
#
# for row in indices:
#     for value in row:
#         print("hi")
indices = torch.tensor([[0, 1, 1, 1, 1, 2, 3, 3, 3, 3], [1, 0, 1, 1, 1, 3, 2, 3, 3, 3], [1, 1, 0, 1, 1, 3, 3, 2, 3, 3], [1, 1, 1, 0, 1, 3, 3, 3, 2, 3], [1, 1, 1, 1, 0, 3, 3, 3, 3, 2]])

rows, D = indices.shape
x = torch.randn(20,10,3)
x_rows, x_D,channels = x.shape
B = 20
k = 4
weight = torch.randn(k,3,3)
x_normal = torch.randn(20,10)
# indices : (rows, D) with values in 0..k-1
indices_exp = indices.expand(B, -1, -1)                   # (B, rows, D)
src         = x.unsqueeze(1)

mask = F.one_hot(indices, num_classes=k).float()

S = torch.einsum('bd,rdk->brk', x_normal, mask)











S = torch.einsum('bdc,rdk->brkc', x, mask)

s_transpose = S.T
# STEP B – multiply each group by its learnable scalar
counts = mask.sum(1)                       # (rows, k)

unsqueezed_counts =  counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
# broadcast counts to (B, rows, k, C) and divide
S_mean = S / unsqueezed_counts

# ── continue exactly as before ───────────────────────────────────────
weighted = torch.einsum('brkc,kcf->brkf', S_mean, weight)
# STEP C – add the k groups → same shape as old `result`
out = weighted.sum(2)  # (B, rows)
print(out)
y_plain = torch.einsum('bdc,rdcf->brf', x, weight[indices])
print(y_plain)
xmax = x.max(dim=1, keepdim=True)

src = src.expand(B, rows, x_D,channels)   # (B, rows, D)
neg_inf = torch.finfo(x.dtype).min
S_max = torch.full((B, rows,  k, channels), neg_inf, device=x.device)
S_max.scatter_reduce_(dim=2, index=indices_exp, src=src, reduce="amax")
S_max = S_max.sum(2)
print(S_max)
matrixSplits = [
    [[i for i in range(indices.size(1)) if indices[j][i] == k] for j in range(indices.size(0))]
    for k in
    range(k)
]
weightIndicesSplits = [torch.tensor(x) for x in matrixSplits]


check = (x[:, weightIndicesSplits[0]])
max_check = torch.max(check, dim=2)
result = (weight[0] * torch.max(check, dim=2).values)
for i in range(1, k):
    check2 = x[:, weightIndicesSplits[i]]
    result += weight[i] *  torch.max(check, dim=2).values
final = result
print(final)


# weights = normal_values[testval]
# print(weights)
#
# x = torch.randn(10,1,1)
# normal_x = x.flatten()
# indices = [
#     [[i for i in range(testval.size(1)) if testval[j][i] == k]for j in range(testval.size(0)) ]for k in range(4)
# ]
# indices0 = torch.tensor(indices[0])
# indices1 = torch.tensor(indices[1])
# indices2 = torch.tensor(indices[2])
# indices3 = torch.tensor(indices[3])
#
# checkResult = weights @ normal_x
#
# result0 = x[indices0] @ values[0]
# result1 = x[indices1] @ values[1]
# result2 = x[indices2] @ values[2]
# result3 = x[indices3] @ values[3]
# result = torch.hstack((result0,result1,result2,result3))
# result = torch.sum(result, dim=1)
# print(result.flatten())
# print(checkResult)
