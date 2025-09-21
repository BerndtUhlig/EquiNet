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

    def __init__(self, k: int, channels_in: int, channels_out: int, n_list: List[int] = None):
        super().__init__()
        self.k = k
        self.channels_in = channels_in
        self.channels_out = channels_out
        weights = torch.randn(k,channels_out,channels_in)
        self.weightParameter = torch.nn.Parameter(weights)
        new_tensor_data = []
        for i in range(len(n_list)):
            for b in range(n_list[i]):
                new_tensor_data.append(i)
        tuples = list(distinct_permutations(new_tensor_data))
        val = [list(data) for data in tuples]
        indices = torch.tensor(val) # (P, N)
        self.register_buffer("indices", indices)
        self.inputSize =  indices.size(1)
        self.outputSize =  indices.size(0)
        M = torch.stack([( self.indices == a).float() for a in range(k)], 0)
        self.register_buffer("M", M)

    def forward(self,x):
        B,L,C = x.shape
        assert C == self.channels_in
        y = x.new_zeros(B, self.outputSize, self.channels_out)
        for index in range(self.k):
            val = self.weightParameter[index].T
            W_k = x @ val
            index_m = self.M[index]
            sol = torch.einsum("BNC, PN-> BPC", W_k, index_m)
            y+=sol
        return y







class PermutationClosedStructureInverse(nn.Module):

    def __init__(self,channels_in, channels_out, running_weight_matrix):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        transpose = running_weight_matrix.T
        check =  transpose @ running_weight_matrix
        rows, columns = transpose.shape
        state = []
        for row in running_weight_matrix:
            reference = list(np.unique(row))
            if reference == state:
                continue
            else:
                state = reference
                if len(reference) > 1:
                    self.PCSList.append(PermutationClosedStructure(2,channels_in,channels_out,[1,rows-1]))
                else:
                    self.constants.append(ConstantNode(channels_in,channels_out,1))

        sum = 0
        for i in range(len(self.PCSList)):
            if i == 0:
                self.indices = self.PCSList[i].indices.T
            else:
                sum += self.PCSList[i].k
                pcsindex = self.PCSList[i].indices.T
                self.indices = torch.hstack([self.indices, pcsindex + torch.full(pcsindex.size(), sum)])

        if len(self.constants) > 0:
            for i in range(len(self.constants)):
                sum += 1
                constindex = torch.full((rows,1), sum)
                self.indices = torch.hstack([self.indices, constindex])

    def forward(self,x):
        samples, points, channels = x.shape
        rows, columns = self.indices.shape
        bigChunks = torch.split(x,[len(self.PCSList)*rows,len(self.constants)],dim=1)
        chunkPCS = torch.chunk(bigChunks[0], len(self.PCSList), dim=1)
        result = torch.zeros((samples,rows,self.channels_out), device=x.device)
        for i in range(len(self.PCSList)):
            result += self.PCSList[i](chunkPCS[i])
        if len(bigChunks[1][1]) > 0:
            chunkConstants = torch.chunk(bigChunks[1], len(self.constants), dim=1)
            for k in range(len(self.constants)):
                result += self.constants[k](chunkConstants[k])
        final = result
        return result




class PermutationClosedLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, channels_in: int, channels_out: int, predecessor, find_max_pcs: bool):
        super().__init__()
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.predecessor = predecessor
        print("LAYER GENERATION STARTED")
        if input_size <= output_size:
            print("GENERATING PCS SMALL TO BIG")
            self.pcs_generation(input_size, output_size, channels_in, channels_out, find_max_pcs)

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
            pcsInv = PermutationClosedStructureInverse( channels_in,channels_out,running_weight_matrix)
            self.PCSList.append(pcsInv)
            self.indices = pcsInv.indices
            self.constants = pcsInv.constants
        if predecessor is not None:
            self.running_weight_matrix = self.indices @ predecessor.running_weight_matrix
        else:
            self.running_weight_matrix = self.indices

    def forward(self,x):
        #print("FORWARD CALLED")
        #self.regenerate_weight_matrix()

        values = torch.hstack([pcs(x) for pcs in self.PCSList])
        if len(self.constants) > 0 and self.input_size <= self.output_size:
            constant_matrix = torch.hstack([const(x) for const in self.constants])
            values = torch.hstack((values, constant_matrix))
        return values
    # 5!  > 5!/2! * 2! * 1! > 5!/3!*1!*1! > 5!/4! * 1!
    # 5! > 5!/2!*2!*1!

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

    def pcs_generation(self, input_size: int, output_size: int, channels_in: int, channels_out: int, find_max_pcs: bool) -> None:
        # Current idea here is: More different weights = more expressiveness = better. Might need to ask about that
        # We therefore stack a bunch of PCS with 2 different weights on top of each other. 2 weights per PCS means the PCS will always be input_size in size
        # Several PCS with 2 different weights each should be better than one gigantic PCS with 3 different weights

        # find maxPCS fit (do this later)

        if not find_max_pcs:
            PCSamount = output_size/input_size
            for i in range(int(PCSamount)):
                self.PCSList.append(PermutationClosedStructure(2, channels_in, channels_out, [1,input_size - 1]))

            difference = output_size - int(PCSamount) * input_size
            if difference > 0:
                for i in range(difference):
                    constant = ConstantNode(channels_in, channels_out,input_size)
                    self.constants.append(constant)
        else:
         weights, pcs_size = self.create_maximum_pcs(input_size,output_size)
         difference = output_size - pcs_size
         self.PCSList.append(PermutationClosedStructure(len(weights), channels_in, channels_out, weights))
         while difference >= output_size:
             weights, pcs_size = self.create_maximum_pcs(input_size, output_size)
             difference = difference - pcs_size
             self.PCSList.append(PermutationClosedStructure(len(weights), channels_in, channels_out, weights))
         if difference > 0:
                for i in range(int(difference)):
                    constant = ConstantNode(input_size)
                    self.constants.append(constant)