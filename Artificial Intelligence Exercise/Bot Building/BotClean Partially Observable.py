#The game Bot Clean took place in a fully observable environment, i.e., the state of every cell was visible
#to the bot at all times. Let us consider a variation of it where the environment is partially observable.
#The bot has the same actuators and sensors. But the sensors visibility is confined to its 8 adjacent cells.

#!/usr/bin/python3
import math
def next_move(posx, posy, board):

    xMove= None
    yMove = None
    euclidean = 99999999999
    if(board[posx][posy] =="d"):
        print("CLEAN")
    else:
        c=[]
        z = []
       
        try:
            with open("aaa1.txt") as file:
                data = file.read()
                c.append(data.split(';'))
                t=0
                while(t<5):
                    temp = c[0][t].split(',')
                    z.append(temp)
                    t=t+1
                
                u=0
                while(u<5):
                    v=0
                    while(v<5):
                
                        if(board[u][v]=='b'):
                            z[u][v]='b'
                        elif(board[u][v]=='d'):                  
                            z[u][v]='d'
                        elif(board[u][v]=='-'):                  
                            z[u][v]='-'                             
                        v=v+1           
                    u =u+1
                
                board = z
                x=0
                while(x<5):
                    y=0
                    while(y<5):

                        if(y==4):
                            if(board[x][y]=='b'):                        
                                f.write('-;')
                            else:                       
                                f.write(str(board[x][y])+';')
                   
                        else:
                            if(board[x][y]=='b'):                        
                                f.write('-,')
                            else:                       
                                f.write(str(board[x][y])+',')  
                        y=y+1
                    x=x+1

                f.close()
        except:
            with open("aaa1.txt", "w") as f:
                x=0
                while(x<5):
                    y=0
                    while(y<5):
                        if(y==4):
                            if(board[x][y]=='b'):                        
                                f.write('-;')
                            else:                       
                                f.write(str(board[x][y])+';')
                    
                        else:
                            if(board[x][y]=='b'):                        
                                f.write('-,')
                            else:                       
                                f.write(str(board[x][y])+',')  
                        y=y+1
                    x=x+1

                f.close()
       
        x=0
        while(x<5):
            y=0
            while(y<5):
                #print(board[x][y])
                if(board[x][y]=='d'):
                    euclideanCalc = math.sqrt((posx-x)**2 + (posy-y)**2)
                    
                    if(euclidean>euclideanCalc ):
                        euclidean = euclideanCalc
                        xMove = x
                        yMove = y                                         
                
                y = y+1
                               
            x=x+1
        if(xMove != None):    
            xDistance =  posx-xMove    
            yDistance = posy- yMove
        else:
            x=0
            while(x<5):
                y=0
                while(y<5):
                    if(board[x][y]=='o'):          
                        euclideanCalc = math.sqrt((posx-x)**2 + (posy-y)**2)
                        if(euclidean>euclideanCalc ):
                            euclidean = euclideanCalc
                            xMove = x
                            yMove = y                                         
                
                    y = y+1
                               
                x=x+1
            
            xDistance =  posx-xMove    
            yDistance = posy- yMove

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
