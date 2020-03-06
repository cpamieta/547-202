#In this version of "Bot saves princess", Princess Peach and bot's position are randomly set. Can you save the princess?

#

def nextMove(n,r,c,grid):
    nn=n-1
    m=int(n/2)
    x=0
    xx=0
    yy=0

    notFound=True
    while(notFound):
        y=0
        while(y<n):

            if(grid[x][y]== 'p'):
                notFound=False
                yy = x-r
                xx = y -c
          
            y=y+1                   
        x=x+1
    if(xx<0):
        return("LEFT")
    elif(xx>0):
        return("RIGHT")
       
            
    elif(yy<0):
        return("UP")
    elif(yy>0):
        return("DOWN")

n = int(input())
r,c = [int(i) for i in input().strip().split()]
grid = []
for i in range(0, n):
    grid.append(input())

print(nextMove(n,r,c,grid))
