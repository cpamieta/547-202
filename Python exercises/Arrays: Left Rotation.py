#A left rotation operation on an array shifts each of the array's elements unit to the left. 
#Given an array of integers and a number, , perform left rotations on the array. 
#Return the updated array to be printed as a single line of space-separated integers.

import math
import os
import random
import re
import sys

# Complete the rotLeft function below.
def rotLeft(a, d):
    arrayHolder =[None] * len(a)
    print(arrayHolder)

    index=0
    for move in a:
        newIndex = index - d

        
        if(newIndex<0):
            print(d)

            newIndex = len(a) + newIndex

            
        
        arrayHolder[newIndex] = move
        
        index = index+1
        
    return(arrayHolder)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nd = raw_input().split()

    n = int(nd[0])

    d = int(nd[1])

    a = map(int, raw_input().rstrip().split())

    result = rotLeft(a, d)

    fptr.write(' '.join(map(str, result)))
    fptr.write('\n')

    fptr.close()
