#You are given an unordered array consisting of consecutive integers [1, 2, 3, ..., n] without any duplicates. 
#You are allowed to swap any two elements. You need to find the minimum number of swaps required to sort the array in ascending order. 

import math
import os
import random
import re
import sys

# Complete the minimumSwaps function below.
def minimumSwaps(arr):

      #arrayHolder =[None] * len(a)
    count=0
    x=0
    index=1
    #for line in q:
    while(x<len(arr)):


        
        if(arr[x]== index):
            x = x+1
            index = index+1

            
            
        else:
            
            
            
            hold = arr[x]
            hold1 = arr[hold-1]
            arr[hold-1]=hold
            arr[x] = hold1
            count =count +1
    return(count)        
            



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    res = minimumSwaps(arr)

    fptr.write(str(res) + '\n')

    fptr.close()
