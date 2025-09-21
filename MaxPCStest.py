
import math
from more_itertools import distinct_permutations


def create_maximum_pcs(input_size: int, output_size: int, max_k: int = 3):
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
            lst = lst_cpy.copy()
            break
    newlist = [x for x in lst if x > 0]
    return newlist, int(k)

lst, val = create_maximum_pcs(5,1000)
print(lst)
print(val)