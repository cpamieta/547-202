#The goal of Artificial Intelligence is to create a rational agent (Artificial Intelligence 1.1.4). 
#An agent gets input from the environment through sensors and acts on the environment with actuators.
#In this challenge, you will program a simple bot to perform the correct actions based on environmental input.

#Meet the bot MarkZoid. It's a cleaning bot whose sensor is a head mounted camera and whose actuators are the wheels beneath it.
#It's used to clean the floor.

#The bot here is positioned at the top left corner of a 5*5 grid. Your task is to move the bot to clean all the dirty cells.

#!/usr/bin/python

#Manhattan distance
#Travelling Salesman Problem (TSP):
#euclidean distance
import math  


def next_move(posr, posc, board):
    xMove= None
    yMove = None
    euclidean = 99999999999
    if(board[posr][posc] =="d"):
        print("CLEAN")
    else:
        x=0
        while(x<5):
            y=0
            while(y<5):
                #print(board[x][y])
                if(board[x][y]=='d'):
                    euclideanCalc = math.sqrt((posr-x)**2 + (posc-y)**2)
                    
                    if(euclidean>euclideanCalc ):
                        euclidean = euclideanCalc
                        xMove = x
                        yMove = y                                         
                
                y = y+1
                               
            x=x+1
        xDistance =  posr-xMove    
        yDistance = posc- yMove
        if(xDistance>0):
            print("UP")
            
        elif(xDistance<0):
            print("DOWN")
        elif(yDistance>0):
            print("LEFT")
            
        else:
            print("RIGHT")
            


if __name__ == "__main__":
    pos = [int(i) for i in input().strip().split()]
    board = [[j for j in input().strip()] for i in range(5)]
    next_move(pos[0], pos[1], board)
