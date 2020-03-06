#Each time Sunny and Johnny take a trip to the Ice Cream Parlor, they pool their money to buy ice cream. 
#On any given day, the parlor offers a line of flavors. Each flavor has a cost associated with it.

#Given the value of and the of each flavor for trips to the Ice Cream Parlor, help Sunny and Johnny choose two distinct
#flavors such that they spend their entire pool of money during each visit. ID numbers are the 1- based index number
#associated with a . For each trip to the parlor, print the ID numbers for the two types of ice cream that Sunny and
#Johnny purchase as two space-separated integers on a new line. You must print the smaller ID first and the larger ID second.

#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the whatFlavors function below.

def whatFlavors(cost, money):
    x=0
    icecreamHash = {}


    for ice in cost:
        if(money-ice) in icecreamHash:
            print("{} {}".format(icecreamHash[money-ice]+1, x+1))
            return
        else:
            icecreamHash[ice] = x


            x=x+1

if __name__ == '__main__':
    t = int(input())

    for t_itr in range(t):
        money = int(input())

        n = int(input())

        cost = list(map(int, input().rstrip().split()))

        whatFlavors(cost, money)
