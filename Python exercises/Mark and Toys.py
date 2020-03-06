#Mark and Jane are very happy after having their first child. Their son loves toys, so Mark wants to buy some.
#There are a number of different toys lying in front of him, tagged with their prices. Mark has only a certain amount to
#spend, and he wants to maximize the number of toys he buys with this money.

#Given a list of prices and an amount to spend, what is the maximum number of toys Mark can buy?

#!/bin/python

import math
import os
import random
import re
import sys

# Complete the maximumToys function below.
def maximumToys(prices, k):
    prices.sort()
    count = 0
    for item in prices:
        k=k-item
        if(k>=0):
            count = count+1
        else:
            break

    return count

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = raw_input().split()

    n = int(nk[0])

    k = int(nk[1])

    prices = map(int, raw_input().rstrip().split())

    result = maximumToys(prices, k)

    fptr.write(str(result) + '\n')

    fptr.close()
