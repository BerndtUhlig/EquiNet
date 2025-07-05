import numpy
import torch
import torch.nn as nn
from sympy.utilities.iterables import multiset_permutations
from sympy import multinomial_coefficients
import numpy as np
from typing import List


class ConstantNode(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.value = nn.Parameter(torch.full((k,), torch.randn(1).item()))


class PermutationClosedStructure(nn.Module):

    def __init__(self, k: int, n_list: List[int] = None):
        super().__init__()
        self.k = k
        weightTensor = torch.randn(k)
        self.weightParameter = nn.Parameter(weightTensor)
        weightList = self.weightParameter.tolist()
        if n_list is None:
            val = list(multiset_permutations(weightList))
            val = [list(i) for i in val]
        else:
            new_tensor_data = []
            for i in range(len(n_list)):
                for b in range(n_list[i]):
                    new_tensor_data.append(weightList[i])
            val = list(multiset_permutations(new_tensor_data))
            val = [list(i) for i in val]

        reference = weightList
        self.indices = torch.tensor(
            np.array( [[np.where(np.isclose(reference, element))[0][0] for element in row] for row in val]))

    def forward(self):
        return self.weightParameter[self.indices]



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


    def forward(self):
        return self.weightParameter[self.indices]



class PermutationClosedLayer(nn.Module):

    def __init__(self, input_size: int, output_size: int, predecessor, last_layer, max_weights_PCS: int):
        super().__init__()
        self.PCSList = nn.ModuleList()
        self.constants = nn.ModuleList()
        self.input_size = input_size
        self.output_size = output_size
        self.predecessor = predecessor
        print("LAYER GENERATION STARTED")
        if input_size <= output_size:
            print("GENERATING PCS SMALL TO BIG")
            self.pcs_generation(input_size, output_size, max_weights_PCS)

            sum = 0
            for i in range(len(self.PCSList)):
                if i == 0:
                    self.indices = self.PCSList[i].indices
                else:
                    sum += self.PCSList[i].k
                    pcsindex = self.PCSList[i].indices
                    self.indices = torch.vstack([self.indices, pcsindex + torch.full(pcsindex.size(),sum)])

        else:
            print("GENERATING PCS BIG TO SMALL")
            curr_pred = predecessor
            while (curr_pred is not None):
                if curr_pred.input_size > self.output_size:
                    curr_pred = curr_pred.predecessor
                    continue
                else:
                    break
            assert (curr_pred is not None)
            running_weight_matrix = predecessor.running_weight_matrix
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
            numpy_check = predecessor.running_weight_matrix.detach().numpy()
            self.running_weight_matrix = self.indices @ predecessor.running_weight_matrix
        else:
            self.running_weight_matrix = self.indices

    def forward(self,x):
        #print("FORWARD CALLED")
        #self.regenerate_weight_matrix()
        weights = torch.vstack([x() for x in self.PCSList])
        if len(self.constants) > 0:
            constant_matrix = torch.vstack([x.value for x in self.constants])
            weights = torch.vstack((weights, constant_matrix))
        val = x @ weights.T
        return val

    # 5!  > 5!/2! * 2! * 1! > 5!/3!*1!*1! > 5!/4! * 1!
    # 5! > 5!/2!*2!*1!

    def regenerate_weight_matrix(self):
        #print("REGENERATING WEIGHT MATRIX")
        self.weights = torch.vstack([x.weightMatrix for x in self.PCSList])
        if len(self.constants) > 0:
            constant_matrix = torch.vstack([x.value for x in self.constants])
            self.weights = torch.vstack((self.weights, constant_matrix))


    def pcs_generation(self, input_size: int, output_size: int, max_weights_PCS: int) -> None:
        # Current idea here is: More different weights = more expressiveness = better. Might need to ask about that
        # We therefore stack a bunch of PCS with 2 different weights on top of each other. 2 weights per PCS means the PCS will always be input_size in size
        # Several PCS with 2 different weights each should be better than one gigantic PCS with 3 different weights
        maxW = max_weights_PCS
        if max_weights_PCS > input_size:
            maxW = input_size

        # find maxPCS fit (do this later)

        PCSamount = output_size / input_size
        for i in range(int(PCSamount)):
            self.PCSList.append(PermutationClosedStructure(maxW, [1,input_size - 1]))

        difference = output_size - int(PCSamount) * input_size
        if (difference > 0):
            for i in range(difference):
                constant = ConstantNode(input_size)
                self.constants.append(constant)
