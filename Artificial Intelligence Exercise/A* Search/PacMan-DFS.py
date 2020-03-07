#!/usr/bin/python

"""
    (0,1) - UP is inserted first
    (1,0) - LEFT is inserted second
    (1,2) - RIGHT is inserted third
    (2,1) - DOWN is inserted fourth (on top)

"""

def dfs( r, c, pacman_r, pacman_c, food_r, food_c, grid):
    stack, path = [[pacman_r,pacman_c]], []
    adjacency_matrix = {}
    x=0
    while(x<r):
        y=0
        while(y<c):
            if(grid[x][y]!='%'):
                moves = []
                #down
                if(x-1 >= 0 and grid[x-1][y]!='%'):
                    moves.append([x-1,y])                 
                if(x-1 >= 0 and grid[x-1][y]!='%'):
                    moves.append([x-1,y]) 
                if(y-1>=0 and grid[x][y-1]!='%'):
                    moves.append([x,y-1]) 
                if(y+1<c and grid[x][y+1]!='%'):
                    moves.append([x,y+1])                         
                if(x+1<r and grid[x+1][y]!='%'):
                    moves.append([x+1,y])
               
                   
                  
                adjacency_matrix[x,y] = moves                           
            y=y+1      
        x=x+1
    foodFound = True
    while stack and foodFound:
        vertex = stack.pop()
        
        if vertex in path:
            continue
        path.append(vertex)
        if(vertex[0]==food_r and vertex[1]==food_c):
            foodFound=False
            break
            
        for neighbor in adjacency_matrix[vertex[0],vertex[1]]:
            stack.append(neighbor)


    print(len(path))
    for f in path:
        print("{} {}".format(f[0] , f[1]))
    print(len(path)-1)
    for f in path:
        print("{} {}".format(f[0] , f[1]))
    #return r

    
    
    
pacman_r, pacman_c = [ int(i) for i in raw_input().strip().split() ]
food_r, food_c = [ int(i) for i in raw_input().strip().split() ]
r,c = [ int(i) for i in raw_input().strip().split() ]

grid = []
for i in xrange(0, r):
    grid.append(raw_input().strip())

dfs(r, c, pacman_r, pacman_c, food_r, food_c, grid)
