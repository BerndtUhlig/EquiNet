import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.utilities.iterables import multiset_permutations
from sympy import multinomial_coefficients
import numpy as np
from typing import List
import math



class ConstantNode(nn.Module):

    def __init__(self,channels_in,channels_out, k: int):
        super().__init__()
        self.size = k
        self.value = nn.Parameter(torch.randn(1,channels_out,channels_in))

    def forward(self, x):
        return torch.full((self.size,),self.value.item()) @ x



class PermutationClosedStructure(nn.Module):

    def __init__(self, k: int, channels_in: int, channels_out: int,  pooling_function: str, n_list: List[int] = None):
        super().__init__()
        self.k = k
        self.pooling_function = pooling_function
        weightTensor = torch.randn(k,channels_out,channels_in)
        self.weightParameter = nn.Parameter(weightTensor)
        weightList = self.weightParameter.tolist()
        if n_list is None:
            val = list(multiset_permutations(weightList))
            val = [list(i) for i in val]
        else:
            new_tensor_data = []
            for i in range(len(n_list)):
                for b in range(n_list[i]):
                    new_tensor_data.append(i)
            val = list(multiset_permutations(new_tensor_data))
            self.indices = torch.tensor(val)
            #
            # mask = F.one_hot(self.indices, num_classes=self.k).float()
            # self.register_buffer("mask", mask)
            matrixSplits = [
            [[i for i in range(self.indices.size(1)) if self.indices[j][i] == k] for j in range(self.indices.size(0))] for k in
            range(self.k)
            ]
            self.weightIndicesSplits = [torch.tensor(x) for x in matrixSplits]

    def forward(self,x):
        samples, size, channels = x.shape
        rows, D = self.indices.shape
        if self.pooling_function == "mean":

            check = (x[:, self.weightIndicesSplits[0]])
            mean = torch.mean(check, dim=2)
            result = mean @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result += torch.mean(check, dim=2).values @ (self.weightParameter[i]).T
            final = result
            return final

            # S = torch.einsum('bdc,rdk->brkc', x, self.mask)
            #
            # counts = self.mask.sum(1)
            # unsqueezed_counts = counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
            # S_mean = S / unsqueezed_counts
            # # STEP B – multiply each group by its learnable scalar
            # weighted = torch.einsum('brkc,kcf->brkf', S_mean, self.weightParameter)
            #
            # # STEP C – add the k groups → same shape as old `result`
            # out = weighted.sum(2)  # (B, rows)

        elif self.pooling_function == "max":
            check = (x[:, self.weightIndicesSplits[0]])
            max = torch.max(check, dim=2).values
            result = max @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result += torch.max(check, dim=2).values @ (self.weightParameter[i]).T
            final = result
            return final

            # indices_exp = self.indices.expand(samples, -1, -1, channels)
            # neg_inf = torch.finfo(x.dtype).min
            # src = x.unsqueeze(1).expand(samples, rows, size, channels)
            # S_max = torch.full((samples, rows, self.k, channels), neg_inf, device=x.device)
            # S_max.scatter_reduce_(dim=2, index=indices_exp, src=src, reduce="amax")
            # S_max = S_max.sum(2)
            # return S_max
        else:
            # out = torch.einsum('bdc,rdcf->brf', x, self.weightParameter[self.indices])
            # return out
            check = (x[:, self.weightIndicesSplits[0]])
            sum = torch.sum(check, dim=2)
            result = sum @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result += torch.sum(check2, dim=2) @ (self.weightParameter[i]).T
            final = result
            return final







class PermutationClosedStructureInverse(nn.Module):

    def __init__(self,channels_in, channels_out,pooling_function: str, running_weight_matrix):
        super().__init__()
        self.pooling_function = pooling_function
        # This may be rather poor in performance => Potential for using "pseudo inverse"
        running_weight_matrix_check = running_weight_matrix.detach().numpy()
        transpose = running_weight_matrix.T.detach().numpy()
        transpose = transpose
        reference = np.unique(transpose[0])
        test =  transpose @ running_weight_matrix.detach().numpy()
        self.weightMatrix = []
        check = [[np.where(np.isclose(reference, element))[0][0] for element in row] for row in transpose]
        # for i in range(len(check)):
        #      for j in range(len(check[i])):
        #          if not check[i][j][0]:
        #              if check[i][j][0] == 0:
        #                  continue
        #              tst = check[i][j]
        #              print("OY")
        self.indices = torch.tensor(np.array(check))
        self.k = len(reference)
        weight_vals = torch.randn(self.k,channels_out,channels_in)
        self.weightParameter = nn.Parameter(weight_vals).to(torch.float32)

        # mask = F.one_hot(self.indices, num_classes=self.k).float()
        # self.register_buffer("mask", mask)
        matrixSplits = [
            [[i for i in range(self.indices.size(1)) if self.indices[j][i] == k] for j in range(self.indices.size(0))]
            for k in
            range(self.k)
        ]
        self.weightIndicesSplits = [torch.tensor(x) for x in matrixSplits]

    def forward(self,x):
        samples, size, channels = x.shape
        rows, D = self.indices.shape
        if self.pooling_function == "mean":

            check = (x[:, self.weightIndicesSplits[0]])
            mean = torch.mean(check, dim=2)
            result = mean @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result +=  torch.mean(check, dim=2).values @ (self.weightParameter[i]).T
            final = result
            return final

            # S = torch.einsum('bdc,rdk->brkc', x, self.mask)
            #
            # counts = self.mask.sum(1)
            # unsqueezed_counts = counts.unsqueeze(0).unsqueeze(-1).clamp(min=1)
            # S_mean = S / unsqueezed_counts
            # # STEP B – multiply each group by its learnable scalar
            # weighted = torch.einsum('brkc,kcf->brkf', S_mean, self.weightParameter)
            #
            # # STEP C – add the k groups → same shape as old `result`
            # out = weighted.sum(2)  # (B, rows)

        elif self.pooling_function == "max":
            check = (x[:, self.weightIndicesSplits[0]])
            max = torch.max(check, dim=2).values
            result = max @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result +=  torch.max(check, dim=2).values @ (self.weightParameter[i]).T
            final = result
            return final



            # indices_exp = self.indices.expand(samples, -1, -1, channels)
            # neg_inf = torch.finfo(x.dtype).min
            # src = x.unsqueeze(1).expand(samples, rows, size, channels)
            # S_max = torch.full((samples, rows, self.k, channels), neg_inf, device=x.device)
            # S_max.scatter_reduce_(dim=2, index=indices_exp, src=src, reduce="amax")
            # S_max = S_max.sum(2)
            # return S_max
        else:
            # out = torch.einsum('bdc,rdcf->brf', x, self.weightParameter[self.indices])
            # return out
            check = (x[:, self.weightIndicesSplits[0]])
            sum = torch.sum(check, dim=2)
            result = sum @ (self.weightParameter[0]).T
            for i in range(1, self.k):
                check2 = x[:, self.weightIndicesSplits[i]]
                result +=  torch.sum(check2, dim=2) @ (self.weightParameter[i]).T
            final = result
            return final


class PermutationClosedLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, channels_in: int, channels_out: int,pooling_function: str, predecessor, find_max_pcs: bool):
        super().__init__()
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.predecessor = predecessor
        self.pooling_function = pooling_function
        print("LAYER GENERATION STARTED")
        if input_size <= output_size:
            print("GENERATING PCS SMALL TO BIG")
            self.pcs_generation(input_size, output_size, channels_in, channels_out,pooling_function, find_max_pcs)

            sum = 0
            for i in range(len(self.PCSList)):
                if i == 0:
                    self.indices = self.PCSList[i].indices
                else:
                    sum += self.PCSList[i].k
                    pcsindex = self.PCSList[i].indices
                    self.indices = torch.vstack([self.indices, pcsindex + torch.full(pcsindex.size(),sum)])

            if len(self.constants) > 0:
                for i in range(len(self.constants)):
                    sum += 1
                    self.indices = torch.vstack([self.indices, torch.full((self.constants[i].size,),sum)])

        else:
            print("GENERATING PCS BIG TO SMALL")
            curr_pred = predecessor
            running_weight_matrix = None
            while (curr_pred is not None):
                if curr_pred.input_size > self.output_size:
                    if running_weight_matrix == None:
                        running_weight_matrix = curr_pred.indices
                        curr_pred = curr_pred.predecessor
                        continue
                    running_weight_matrix = running_weight_matrix @ curr_pred.indices
                    curr_pred = curr_pred.predecessor

                    continue
                else:
                    if running_weight_matrix == None:
                        running_weight_matrix = curr_pred.indices
                    else:
                        running_weight_matrix = running_weight_matrix @ curr_pred.indices
                    break
            assert (curr_pred is not None)

            difference = self.output_size - curr_pred.input_size
            pcsInv = PermutationClosedStructureInverse( channels_in,channels_out,pooling_function,running_weight_matrix)
            self.PCSList.append(pcsInv)
            self.weights = pcsInv.weightMatrix
            self.indices = pcsInv.indices
            if difference > 0:
                print("GENERATING PCS BIG TO SMALL => CONSTANT NODES")
                for i in range(difference):
                    constant = ConstantNode(input_size, channels_in, channels_out)
                    self.constants.append(constant)
                constant_matrix = torch.vstack([x.value for x in self.constants])
        if predecessor is not None:
            self.running_weight_matrix = self.indices @ predecessor.running_weight_matrix
        else:
            self.running_weight_matrix = self.indices

    def forward(self,x):
        #print("FORWARD CALLED")
        #self.regenerate_weight_matrix()
        values = torch.hstack([pcs(x) for pcs in self.PCSList])
        if len(self.constants) > 0:
            constant_matrix = torch.hstack([const(x) for const in self.constants])
            values = torch.vstack((values, constant_matrix))
        return values
    # 5!  > 5!/2! * 2! * 1! > 5!/3!*1!*1! > 5!/4! * 1!
    # 5! > 5!/2!*2!*1!

    def regenerate_weight_matrix(self):
        #print("REGENERATING WEIGHT MATRIX")
        self.weights = torch.vstack([x.weightMatrix for x in self.PCSList])
        if len(self.constants) > 0:
            constant_matrix = torch.vstack([x.value for x in self.constants])
            self.weights = torch.vstack((self.weights, constant_matrix))


    def create_maximum_pcs(self, input_size:int, output_size: int):
        list = []
        k = math.factorial(input_size)

        for i in range(input_size):
            list.append(1)

        runningindex = 0
        runningindexInv = input_size - 1
        while k > output_size:
            print(runningindex)
            print(runningindexInv)
            if runningindex >= runningindexInv:
                runningindex = 0
                for index in range(len(list)):
                    if list[index] > 0:
                        runningindexInv = index

            list[runningindex] += 1
            list[runningindexInv] -= 1
            print(list)

            k = math.factorial(input_size)
            ksmall = 1
            for item in list:
                ksmall = ksmall * math.factorial(item)
            k = k / ksmall
            print(k)
            runningindex += 1
            runningindexInv -= 1
            if list[0] == input_size - 1 and list[1] == 1:
                break
        newlist = [x for x in list if x > 0]
        return newlist, int(k)

    def pcs_generation(self, input_size: int, output_size: int, channels_in: int, channels_out: int, pooling_function:str, find_max_pcs: bool) -> None:
        # Current idea here is: More different weights = more expressiveness = better. Might need to ask about that
        # We therefore stack a bunch of PCS with 2 different weights on top of each other. 2 weights per PCS means the PCS will always be input_size in size
        # Several PCS with 2 different weights each should be better than one gigantic PCS with 3 different weights

        # find maxPCS fit (do this later)

        if not find_max_pcs:
            PCSamount = output_size/input_size
            for i in range(int(PCSamount)):
                self.PCSList.append(PermutationClosedStructure(2, channels_in, channels_out,pooling_function, [1,input_size - 1]))

            difference = output_size - int(PCSamount) * input_size
            if difference > 0:
                for i in range(difference):
                    constant = ConstantNode(channels_in, channels_out,input_size)
                    self.constants.append(constant)
        else:
         weights, pcs_size = self.create_maximum_pcs(input_size,output_size)
         difference = output_size - pcs_size
         self.PCSList.append(PermutationClosedStructure(len(weights), weights))
         if difference > 0:
                for i in range(int(difference)):
                    constant = ConstantNode(input_size)
                    self.constants.append(constant)