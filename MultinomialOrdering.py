import numpy as np
import math


print(math.factorial(5))


# n = 128
# o = 350000
#
#
# list = []
# k = math.factorial(n)
#
# list.append(n-1)
# list.append(1)
#
# for i in range(n-2):
#     list.append(0)
#
# runningindex = 0
# runningindexInv = n-1
# save_k = 0
# while k < o:
#     print(runningindex)
#     print(runningindexInv)
#     if runningindex >= runningindexInv:
#         runningindex = 0
#         for index in range(len(list)):
#             if list[index] > 0:
#                 runningindexInv = index
#
#
#
#     list[runningindex]+=1
#     list[runningindexInv]-=1
#     print(list)
#
#     k = math.factorial(n)
#     ksmall = 1
#     for item in list:
#         ksmall = ksmall * math.factorial(item)
#     k = k/ksmall
#     print(k)
#     runningindex+=1
#     runningindexInv-=1
#     if list[0] == n-1 and list[1] == 1:
#         break
#
# print(list)
# print(k)
#
