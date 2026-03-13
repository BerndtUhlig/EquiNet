import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.utilities.iterables import multiset_permutations
from sympy import multinomial_coefficients
from more_itertools import distinct_permutations
import numpy as np
from typing import List
import math



class ConstantNode(nn.Module):

    def __init__(self,channels_in,channels_out, k: int):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.size = k
        self.value = nn.Parameter(torch.randn(channels_out,channels_in))

    def forward(self, x):
        out = x.sum(dim=1)
        transpose = self.value.T
        value = out @ transpose
        dim = value.unsqueeze(dim=1)
        return dim



class PermutationClosedStructure(nn.Module):

    def __init__(self, k: int, channels_in: int, channels_out: int,  pooling_function: str, n_list: List[int] = None):
        super().__init__()
        self.k = k
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.Gamma = nn.Linear(channels_in,channels_out)
        self.Lambda = nn.Linear(channels_in,channels_out, bias=False)
        self.pooling_function = pooling_function
        new_tensor_data = []
        for i in range(len(n_list)):
            for b in range(n_list[i]):
                new_tensor_data.append(i)
        tuples = list(distinct_permutations(new_tensor_data))
        val = [list(data) for data in tuples]
        indices = torch.tensor(val)
        self.register_buffer('indices', indices)



    def forward(self,x):
        samples, size, _ = x.shape
        rows, D = self.indices.shape

        if self.pooling_function == "mean":

            xm = x.mean(1,keepdim=True)
            xm = self.Lambda(xm)
            x = self.Gamma(x)
            result = x - xm
            return result


        elif self.pooling_function == "max":

            xm, _ = x.max(1, keepdim=True)
            xm = self.Lambda(xm)
            x = self.Gamma(x)
            result = x - xm
            return result

        else:
            xm = x.sum(1, keepdim=True)
            xm = self.Lambda(xm)
            x = self.Gamma(x)
            result = x - xm
            return result








class PermutationClosedStructureInverse(nn.Module):

    def __init__(self,channels_in, channels_out,pooling_function: str, running_weight_matrix):
        super().__init__()
        self.pooling_function = pooling_function
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        transpose = running_weight_matrix.T
        rows, columns = transpose.shape
        state = []
        for row in running_weight_matrix:
            reference = list(np.unique(row))
            if reference == state:
                continue
            else:
                state = reference
                if len(reference) > 1:
                    self.PCSList.append(PermutationClosedStructure(2,channels_in,channels_out,pooling_function,[1,rows-1]))
                else:
                    self.constants.append(ConstantNode(channels_in,channels_out,1))

        sum = 0
        indices = []
        for i in range(len(self.PCSList)):
            if i == 0:
                indices = self.PCSList[i].indices.T
            else:
                sum += self.PCSList[i].k
                pcsindex = self.PCSList[i].indices.T
                indices = torch.hstack([indices, pcsindex + torch.full(pcsindex.size(), sum)])

        if len(self.constants) > 0:
            for i in range(len(self.constants)):
                sum += 1
                constindex = torch.full((rows,1), sum)
                indices = torch.hstack([indices, constindex])
        self.register_buffer("indices",indices)

    def forward(self,x):
        samples, points, channels = x.shape
        rows, columns = self.indices.shape
        result = torch.zeros((samples,rows,self.channels_out), device=x.device)
        bigChunks = torch.split(x,[len(self.PCSList)*rows,len(self.constants)],dim=1)
        chunkPCS = torch.chunk(bigChunks[0], len(self.PCSList), dim=1)
        for i in range(len(self.PCSList)):
            result += self.PCSList[i](chunkPCS[i])
        if len(self.constants) > 0:
            chunkConstants = torch.chunk(bigChunks[1], len(self.constants), dim=1)
            for k in range(len(self.constants)):
                result += self.constants[k](chunkConstants[k])
        return result




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
            self.indices = pcsInv.indices
            self.constants = pcsInv.constants
        if predecessor is not None:
            self.running_weight_matrix = self.indices @ predecessor.running_weight_matrix
        else:
            self.running_weight_matrix = self.indices

    def forward(self,x):


        values = torch.hstack([pcs(x) for pcs in self.PCSList])
        if len(self.constants) > 0 and self.input_size <= self.output_size:
            constant_matrix = torch.hstack([const(x) for const in self.constants])
            values = torch.hstack((values, constant_matrix))
        return values


    def regenerate_weight_matrix(self):
        #print("REGENERATING WEIGHT MATRIX")
        self.weights = torch.vstack([x.weightMatrix for x in self.PCSList])
        if len(self.constants) > 0:
            constant_matrix = torch.vstack([x.value for x in self.constants])
            self.weights = torch.vstack((self.weights, constant_matrix))

    def create_maximum_pcs(self,input_size: int, output_size: int, max_k: int = 3):
        lst = []
        lst.append(input_size - 1)
        lst.append(1)
        k = input_size
        new_k = input_size
        for i in range(input_size - 2):
            lst.append(0)
        lst_cpy = lst.copy()
        runningindex = 0
        runningindexInv = 1
        while new_k < output_size:
            k = new_k
            lst = lst_cpy.copy()
            print(runningindex)
            print(runningindexInv)
            lst_cpy[runningindex] -= 1
            lst_cpy[runningindexInv] += 1
            print(lst_cpy)
            new_k = math.factorial(input_size)
            ksmall = 1
            for item in lst_cpy:
                ksmall = ksmall * math.factorial(item)
            new_k = new_k / ksmall
            print(new_k)
            runningindex += 1
            if runningindex == runningindexInv:
                runningindex = 0
            if lst_cpy[runningindexInv] >= lst_cpy[runningindex]:
                if lst_cpy[runningindex] < lst_cpy[runningindexInv]:
                    runningindex = runningindexInv
                runningindexInv += 1
            if runningindexInv == len(lst_cpy):
                k = new_k
                break
        newlist = [x for x in lst if x > 0]
        return newlist, int(k)
    def pcs_generation(self, input_size: int, output_size: int, channels_in: int, channels_out: int, pooling_function:str, find_max_pcs: bool) -> None:
        # Current idea here is: More different weights = more expressiveness = better. Might need to ask about that
        # We therefore stack a bunch of PCS with 2 different weights on top of each other. 2 weights per PCS means the PCS will always be input_size in size
        # Several PCS with 2 different weights each should be better than one gigantic PCS with 3 different weights


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