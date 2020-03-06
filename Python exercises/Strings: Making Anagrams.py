#Alice is taking a cryptography class and finding anagrams to be very useful.
#We consider two strings to be anagrams of each other if the first string's letters
#can be rearranged to form the second string. In other words, both strings must contain
#the same exact letters in the same exact frequency For example, bacdc and dcbac are anagrams, but bacdc and dcbad are not.

#Alice decides on an encryption scheme involving two large strings where encryption is dependent on the minimum number
#of character deletions required to make the two strings anagrams. Can you help her find this number? 

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the makeAnagram function below.
def makeAnagram(a, b):
    
    x=0
    y=0
    hashLista= {}
    hashListb= {}
    count=0
    for x in a:
        if(x not in hashLista.keys()):
            hashLista[x] = 1
        else:
            hashLista[x]=hashLista[x]+1

    for x in b:
        if(x not in hashListb.keys()):
            hashListb[x] = 1
        else:
            hashListb[x]=hashListb[x]+1
    
    for value in hashLista:
        if(value not in hashListb.keys()):
            count =count + hashLista[value]

        elif(hashLista[value]!=hashListb[value]):
            count =count + abs(hashLista[value] -hashListb.pop(value))

        else:
            hashListb.pop(value)


    for valueb in hashListb:
        count = count + hashListb[valueb]
    return count



if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    a = input()

    b = input()

    res = makeAnagram(a, b)

    fptr.write(str(res) + '\n')

    fptr.close()
