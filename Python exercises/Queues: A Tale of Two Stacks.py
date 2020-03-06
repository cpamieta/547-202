#In this challenge, you must first implement a queue using two stacks. Then process queries, where each query is one of the following
from queue import Queue 

class MyQueue(object):
    #so the pop in list remove the item in the last index so acts like lilo, but for this
    #we want pop to remove the old item so the objext in index 0
    #so to do this with a list, we need two lists.
    #first list would be loaded the one new items are loaded.
    #as soon as someone calls pop or peak we want to than dump all the data into the other list. With that, the order would be switched so when you do pop on the other list the correct value gets poped


    def __init__(self):
        self.fifo= [] 
        self.lifo = []
            
    def peek(self):
        self.test()      
        return self.lifo[-1]        
        
    def pop(self):
        self.test()
        return self.lifo.pop()         
        
    def put(self, value):
        self.fifo.append(value)

    def test(self):
        if(len(self.lifo) ==0):
            if(len(self.fifo)>0):
                
                while(len(self.fifo)>0):
                    self.lifo.append(self.fifo.pop())                            
queue = MyQueue()
t = int(input())
for line in range(t):
    values = map(int, input().split())
    values = list(values)
    if values[0] == 1:
        queue.put(values[1])        
    elif values[0] == 2:
        queue.pop()
    else:
        print(queue.peek())

