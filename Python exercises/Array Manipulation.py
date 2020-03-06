#Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value to each of the array 
#element between two given indices, inclusive. Once all operations have been performed, return the maximum value in your array. 

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the arrayManipulation function below.
def arrayManipulation(n, queries):
    arrayHolder =[0] * n
    x=0
    while(x < len(queries)):
        y=queries[x][0]
        z=queries[x][1]
        t=queries[x][2]
        while(y<=z):
            
            arrayHolder[y-1]=arrayHolder[y-1]+t
            y=y+1
            
        x=x+1
    return(max(arrayHolder))
        


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    queries = []

    for _ in range(m):
        queries.append(list(map(int, input().rstrip().split())))

    result = arrayManipulation(n, queries)

    fptr.write(str(result) + '\n')

    fptr.close()
