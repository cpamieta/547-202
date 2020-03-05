#Given a large chunk of text, identify the most frequently occurring trigram in it. If there are multiple trigrams with the same frequency,
#then print the one which occurred first.
#Assume that trigrams are groups of three consecutive words in the same sentence which are separated by nothing but a single space and are 
#case insensitive. The size of the input will be less than 10 kilobytes.

#From Paragraphs to Sentences
import re
import sys

maps = { }
count = 0
totCount = 0
#stringInput= "While working at a horse riding camp several years ago I spent a good twenty minutes explaining to a group which consisted of twelve children and their young teacher the importance of horse safety before walking behind a horse and being kicked in the head. I recall only walking in a zigag back to the house with the muffled sounds of children screaming in the background before collapsing and waking up in hospital. While I was there with a fractured skull the teacher bought me in a get well soon card signed by all the children so I asked her out but she said no. I bought a real dinosaurs tooth fossil recently with invoice note of authenticity as it is something I have always wanted. There is a quarry a short drive away that my  yo son and I go to and explore sometimes. When we went there last I suggested we dig for fossils and miraculously found the dinosaur tooth thinking it would be a big deal to him but he stated No its just a rock. When I swore I was positive that it is was a saurischian tooth from the mesozoic era he replied that I had made that up and for me to throw it away. I cannot prove to him that it is a real dinosaur tooth without divulging the invoice and he is never seeing that as I would have to explain why I didnt buy a playstation instead of a million yo fossil. Occasionally he picks it up and gives me a disdaining look. Also I bought some NASA mission badges a while back off ebay. He asked me if they had been in space and I had to admit that they hadnt and he stated Well thats just weak then. While I was in a electronics store called Jaycar buying something with blinking lights a girl approached and asked me a question concerning which network cable would be suitable for her needs. Wanting to appear helpful I found a large selection of cables and listed the benefits of each. After she explained that the cable needed to be long enough to reach from her neighbours house to hers as her neighbour had offered to share their broadband I laughed and told her that was the stupidest thing I have ever heard and did not know if the store had cables that long so she asked Well can I speak to someone else then. I looked blank before realising that I was wearing a blue shirt the same colour as the staff that worked there and the whole time I had been helping her she had assumed that I was an employee. After explaining to her that I did not work there and denying that I had been pretending to do so I asked her out but she said no. While working at a horse riding camp several years ago I spent a good twenty minutes explaining to a group which consisted of twelve children and their young teacher the importance of horse safety before walking behind a horse and being kicked in the head. I recall only walking in a zigag back to the house with the muffled sounds of children screaming in the background before collapsing and waking up in hospital. While I was there with a fractured skull the teacher bought me in a get well soon card signed by all the children so I asked her out but she said no. While I was in a electronics store called Jaycar buying something with blinking lights a girl approached and asked me a question concerning which network cable would be suitable for her needs. Wanting to appear helpful I found a large selection of cables and listed the benefits of each. After she explained that the cable needed to be long enough to reach from her neighbours house to hers as her neighbour had offered to share their broadband I laughed and told her that was the stupidest thing I have ever heard and did not know if the store had cables that long so she asked Well can I speak to someone else then. I looked blank before realising that I was wearing a blue shirt the same colour as the staff that worked there and the whole time I had been helping her she had assumed that I was an employee. After explaining to her that I did not work there and denying that I had been pretending to do so I asked her out but she said no. I bought a real dinosaurs tooth fossil recently with invoice note of authenticity as it is something I have always wanted. There is a quarry a short drive away that my yo son and I go to and explore sometimes. When we went there last I suggested we dig for fossils and miraculously found the dinosaur tooth thinking it would be a big deal to him but he stated No its just a rock. When I swore I was positive that it is was a saurischian tooth from the mesozoic era he replied that I had made that up and for me to throw it away. I cannot prove to him that it is a real dinosaur tooth without divulging the invoice and he is never seeing that as I would have to explain why I didnt buy a playstation instead of a million yo fossil. Occasionally he picks it up and gives me a disdaining look. Also I bought some NASA mission badges a while back off ebay. He asked me if they had been in space and I had to admit that they hadnt and he stated Well thats just weak then. While I was in a electronics store called Jaycar buying something with blinking lights a girl approached and asked me a question concerning which network cable would be suitable for her needs. Wanting to appear helpful I found a large selection of cables and listed the benefits of each. After she explained that the cable needed to be long enough to reach from her neighbours house to hers as her neighbour had offered to share their broadband I laughed and told her that was the stupidest thing I have ever heard and did not know if the store had cables that long so she asked Well can I speak to someone else then. I looked blank before realising that I was wearing a blue shirt the same colour as the staff that worked there and the whole time I had been helping her she had assumed that I was an employee. After explaining to her that I did not work there and denying that I had been pretending to do so I asked her out but she said no."
#stringInput= "I came from the moon. He went to the other room. She went to the drawing room."

stringInput= sys.stdin.read()
quote=False  
stringInput=stringInput.rstrip('.').split(".")  
x=0

   
while(len(stringInput)>totCount):   
    newCount=0
    newCount2=0
    newCount3=0
    newString = stringInput[totCount].lstrip().split()
    key= ""
    x=x+len(newString)
    while(len(newString)>newCount):
        key =  "{} {}".format(key, newString[newCount])
        #I see total 6 still here
        
        if(newCount3==2):  
            if key.strip().lower() in maps:              
                maps[key.strip().lower()] = maps[key.strip().lower()]+1

            else:
                maps[key.strip().lower()]=1

            newCount = newCount2
            newCount2=newCount2+1
            key= ""
            newCount3=-1

        newCount3=newCount3+1    
        newCount=newCount+1
       
       
    totCount = totCount +1
       
finalValue= 0
finalname= ""
for x in maps:
    if(finalValue< maps[x] ):
        finalValue = maps[x]
        finalname = x
        
print(finalname)

   
