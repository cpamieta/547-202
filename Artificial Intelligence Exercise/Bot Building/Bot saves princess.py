#Princess Peach is trapped in one of the four corners of a square grid.
#You are in the center of the grid and can move one step at a time in any of the four directions. Can you rescue the princess? 

#!/usr/bin/python

def displayPathtoPrincess(n,grid):

#print all the moves here
    #found = true
    #while(found):
    nn=n-1
    m=int(n/2)
    if(grid[0][0]== 'p'):
        x=0-m
        y=0-m
            
            
    elif(grid[nn][nn]== 'p'):
        x=nn-m
        y=nn-m
    elif(grid[nn][0]== 'p'):
        x=nn-m
        y=0-m            
    elif(grid[0][nn]== 'p'):
        x=0-m
        y=nn-m
            
            
    while(x!=0):
        if(x<0):
            print("LEFT")
            x=x+1
        else:
            print("RIGHT")
            x=x-1
            
    while(y !=0):
        if(y<0):
            print("UP")
            y=y+1
        else:
            print("DOWN")
            y=y-1

m = int(input())
grid = [] 
for i in range(0, m): 
    grid.append(input().strip())

displayPathtoPrincess(m,grid)
