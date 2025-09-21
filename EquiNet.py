import numpy
import torch
import torch.nn as nn
from sympy.utilities.iterables import multiset_permutations
from sympy import multinomial_coefficients
from more_itertools import distinct_permutations
import numpy as np
from typing import List
import math
import weakref


class ConstantNode(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.size = k
        self.value = nn.Parameter(torch.randn(1))

    def forward(self,x):
        matrix = torch.full((self.size,1),self.value.item())
        result = x @ matrix
        return result


class PermutationClosedStructure(nn.Module):

    def __init__(self, k: int, n_list: List[int] = None):
        super().__init__()
        self.k = k
        weightTensor = torch.randn(k)
        self.weightParameter = nn.Parameter(weightTensor)
        weightList = self.weightParameter.tolist()
        if n_list is None:
            val = list(distinct_permutations(weightList))
            val = [list(i) for i in val]
        else:
            new_tensor_data = []
            for i in range(len(n_list)):
                for b in range(n_list[i]):
                    new_tensor_data.append(i)
            tuples = list(distinct_permutations(new_tensor_data))
            val = [list(data) for data in tuples]
            self.indices = torch.tensor(val)

    def forward(self,x):
        matrix = (self.weightParameter[self.indices]).T
        result =x @ matrix
        return result



class PermutationClosedStructureInverse(nn.Module):

    def __init__(self, running_weight_matrix):
        super().__init__()
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
        weight_vals = torch.randn(self.k)
        self.weightParameter = nn.Parameter(weight_vals).to(torch.float32)
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self,x):
        return x @ (self.weightParameter[self.indices]).T



class PermutationClosedLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, predecessor,find_max_pcs: bool):
        super().__init__()
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.predecessor = weakref.ref(predecessor) if predecessor is not None else None
        print("LAYER GENERATION STARTED")
        if input_size <= output_size:
            print("GENERATING PCS SMALL TO BIG")
            self.pcs_generation(input_size, output_size, find_max_pcs)

            sum = 0
            for i in range(len(self.PCSList)):
                if i == 0:
                    self.indices = self.PCSList[i].indices
                else:
                    sum += self.PCSList[i].k
                    pcsindex = self.PCSList[i].indices
                    self.indices = torch.vstack([self.indices, pcsindex + torch.full(pcsindex.size(), sum)])

            if len(self.constants) > 0:
                for i in range(len(self.constants)):
                    sum += 1
                    self.indices = torch.vstack([self.indices, torch.full((self.constants[i].size,), sum)])

        else:
            print("GENERATING PCS BIG TO SMALL")
            curr_pred = predecessor
            running_weight_matrix = None
            while (curr_pred is not None):
                if curr_pred.input_size > self.output_size:
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
            pcsInv = PermutationClosedStructureInverse(running_weight_matrix)
            self.PCSList.append(pcsInv)
            self.weights = pcsInv.weightMatrix
            self.indices = pcsInv.indices
            if difference > 0:
                print("GENERATING PCS BIG TO SMALL => CONSTANT NODES")
                for i in range(difference):
                    constant = ConstantNode(input_size)
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

    def pcs_generation(self, input_size: int, output_size: int, find_max_pcs: bool) -> None:
        # Current idea here is: More different weights = more expressiveness = better. Might need to ask about that
        # We therefore stack a bunch of PCS with 2 different weights on top of each other. 2 weights per PCS means the PCS will always be input_size in size
        # Several PCS with 2 different weights each should be better than one gigantic PCS with 3 different weights


        if not find_max_pcs:
            PCSamount = output_size/input_size
            for i in range(int(PCSamount)):
                self.PCSList.append(PermutationClosedStructure(2, [1,input_size - 1]))

            difference = output_size - int(PCSamount) * input_size
            if difference > 0:
                for i in range(difference):
                    constant = ConstantNode(input_size)
                    self.constants.append(constant)
        else:
         weights, pcs_size = self.create_maximum_pcs(input_size,output_size)
         difference = output_size - pcs_size
         self.PCSList.append(PermutationClosedStructure(len(weights), weights))
         while difference >= output_size:
             weights, pcs_size = self.create_maximum_pcs(input_size, output_size)
             difference = difference - pcs_size
             self.PCSList.append(PermutationClosedStructure(len(weights), weights))
         if difference > 0:
                for i in range(int(difference)):
                    constant = ConstantNode(input_size)
                    self.constants.append(constant)