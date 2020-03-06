#Harold is a kidnapper who wrote a ransom note, but now he is worried it will be traced back to him through his handwriting.
#He found a magazine and wants to know if he can cut out whole words from it and use them to create an untraceable replica of his ransom note.
#The words in his note are case-sensitive and he must use only whole words available in the magazine. He cannot use substrings or
#concatenation to create the words he needs.

#Given the words in the magazine and the words in the ransom note,
#print Yes if he can replicate his ransom note exactly using whole words from the magazine; otherwise, print No.

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the checkMagazine function below.
def checkMagazine(magazine, note):
    replica = True
    magHash = {}
   
    for mag in magazine:
       
        if mag.strip() in magHash:
            magHash[mag.strip()] = magHash[mag.strip()]+1
        else:         
            magHash[mag.strip()]=1
    for notes in note:
        if notes in magHash:
            if(magHash[notes]>0):
                magHash[notes.strip()] = magHash[notes.strip()]-1          
            else:
                replica=False
                break
        else:
            replica=False
            break     

    if(replica):
        print("Yes")
       
    else:
        print("No")

if __name__ == '__main__':
    mn = input().split()

    m = int(mn[0])

    n = int(mn[1])

    magazine = input().rstrip().split()

    note = input().rstrip().split()

    checkMagazine(magazine, note)
