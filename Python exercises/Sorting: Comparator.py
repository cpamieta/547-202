#Given an array of Player objects, write a comparator that sorts them in order of decreasing score. 
#If or more players have the same score, sort those players alphabetically ascending by name. 
#To do this, you must create a Checker class that implements the Comparator interface, then write an int compare(Player a, Player b) method implementing the Comparator.

from functools import cmp_to_key
class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def __repr__(self):
        pass
    def comparator(a, b):
        val = b.score - a.score
        if val == 0:
            return -1 if a.name < b.name else 1
        return val

