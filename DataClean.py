
# coding: utf-8

# In[ ]:

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
ranks1 = pd.read_csv('weekend1.csv', sep=",", header=1)
ranks2 = pd.read_csv('weekend2.csv', sep=",", header=1)
                                                          

data = pd.read_csv('ticketdata.txt', sep="\t", header=None)

data.columns = ["Title", "Price", "startTime", "endTime","link","subtitle","sold"]
data["# of ticket"] = 1
data["Shuttle Passes"] = 0
data["CAR CAMPING PASS"] = 0
data["VIP"] = 0
data["profit"] = -1000000.00
data["days till"] = -100
data["ticketPrice"] = data["Price"]
data["VIP Parking"] = 0
data["weekend"] = 0
data["PriceRange"] = 0
data["profitpercent"] = 0.00
data["profitloss"] = 0
data['Title'] = data['Title'].astype(str)
data['subtitle'] = data['subtitle'].astype(str)
data["trend"] = 0

#Used to search for key search words like VIP.
data["junk"] = data["Title"]
data['junk']=data['junk'].str.replace("Weekend 2", '')
data['junk']=data['junk'].str.replace("Weekend 1", '', case= False)
data['junk']=data['junk'].str.replace("Weekend 2", '', case= False)
data['junk']=data['junk'].str.replace("Weekend one", '', case= False)
data['junk']=data['junk'].str.replace("Weekend two", '', case= False)
data['junk']=data['junk'].str.replace("3-Day", '', case= False)
data['junk']=data['junk'].str.replace("3 DAY PASS", '', case= False)
data['junk']=data['junk'].str.replace("04", '', case= False)
data['junk']=data['junk'].str.replace("4/", '', case= False)
data['junk']=data['junk'].str.replace("2018", '')
data['junk']=data['junk'].str.replace("13", '')
data['junk']=data['junk'].str.replace("14", '')
data['junk']=data['junk'].str.replace("15", '')
data['junk']=data['junk'].str.replace("20", '')
data['junk']=data['junk'].str.replace("21", '')

data['junk']=data['junk'].str.replace("22", '')
data['junk']=data['junk'].str.replace("23", '')

x = 0
len(data)
date_format = "%Y-%m-%d"
week1 = '2018-04-13'
week2 = '2018-04-20'
while (x< len(data)):
    
    
    
    if(data['sold'][x].lower().find("endedwithsales")>=0):
        data['sold'][x] = 1
    else:
        data['sold'][x] = 0
       

        
        
        
    #searching for key word
    if(data["junk"][x].lower().find("vip parking")>=0):
        data["VIP Parking"][x] = 1
        #if found, subtract from total auction price to find how much the ticket is
        data["ticketPrice"][x] = data["ticketPrice"][x] - 150  
        

       
    if(data["junk"][x].lower().find("camp")>=0):
        data["CAR CAMPING PASS"][x] = 1
        data["ticketPrice"][x] = data["ticketPrice"][x] - 113  

       #Search for key word, if found calculate of many days till the event when the auction ended. 
    if(data["Title"][x].lower().find("weekend one")>=0):      
        data["weekend"][x] = 1                                             
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week1, date_format)    
        data["days till"][x] = int((b - a).days)
            
        y=0
        while (y< len(ranks1)):
        
            if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
            y=y+1        
        
        
        
    if(data["Title"][x].lower().find("weekend 1")>=0):      
        data["weekend"][x] = 1
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week1, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks1)):
        
            if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
            y=y+1
            
    if(data["Title"][x].lower().find("weekend two")>=0):      
        data["weekend"][x] = 2
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks2)):          
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]
    
            y=y+1
        
        
    if(data["Title"][x].lower().find("weekend 2")>=0):      
        data["weekend"][x] = 2
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks2)):    
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]
    
            y=y+1
        
        
    if(data["Title"][x].lower().find("Wknd 2")>=0):      
        data["weekend"][x] = 2
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days) 
        
        y=0
        while (y< len(ranks2)):          
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]
    
            y=y+1
        
    if(data["Title"][x].lower().find("Wknd 1")>=0):      
        data["weekend"][x] = 2
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days) 
        
        y=0
        while (y< len(ranks1)):
        
            if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
            y=y+1
           
        
    
        
     #Did overall search and found most auctions had a max two. Searching for 3 and more caused some issues
    #So I just dropped them. Total around 5 auctions had 3+ tickets included. Need to implement better way for future.
    if(data["junk"][x].find("1")>=0):
        data["# of ticket"][x] = 1
    if(data["junk"][x].find("2")>=0):
        data["# of ticket"][x] = 2
        
        
    if(data["junk"][x].lower().find("one")>=0):
        data["# of ticket"][x] = 1
    if(data["junk"][x].lower().find("two")>=0):
        data["# of ticket"][x] = 2
        
        
        
        
        
    if(data["junk"][x].lower().find("shuttle")>=0):
        data["Shuttle Passes"][x] = 1
        data["ticketPrice"][x] = data["ticketPrice"][x] - (75 *data["# of ticket"][x])    
    
    
    if(data["junk"][x].lower().find("vip")>=0):
        data["VIP"][x] = 1
        data["ticketPrice"][x]= data["ticketPrice"][x]/(data["# of ticket"][x])
        data["profit"][x] = data["ticketPrice"][x]- 999
       #Calculate the profit made or lost on the ticket 
        if(data["profit"][x]>0):
            data["profitpercent"][x] = data["profit"][x]/999
            data["profitloss"][x] = 1
        elif(data["profit"][x]==0):
            data["profitpercent"][x] = 0
            data["profitloss"][x] = 0
        
        elif(data["profit"][x]<0):
            data["profitpercent"][x] =data["profit"][x]/999
            data["profitloss"][x] = -1
        
        
        data["PriceRange"][x] = int(data["ticketPrice"][x] / 50)

    else:  
        data["ticketPrice"][x]= data["ticketPrice"][x] / (data["# of ticket"][x])
        data["profit"][x] = data["ticketPrice"][x] - 429
        data["PriceRange"][x] = int(data["ticketPrice"][x] / 50)
        if(data["profit"][x]>0):
            data["profitpercent"][x] = data["profit"][x]/429
            data["profitloss"][x] = 1
        elif(data["profit"][x]==0):
            data["profitpercent"][x] = 0
            data["profitloss"][x] = 0
        
        elif(data["profit"][x]<0):
            data["profitpercent"][x] =data["profit"][x]/429
            data["profitloss"][x] = -1




    x = x+1

#After all the cleaning if a entry in a dataframe had weekend value as 0,that means that its junk data or the listing didnt
#mention the date. Didnts see a auction of a ticket without a date.
tic = data.ix[data.weekend != 0]
#Group by to find average daily price
groups = tickets
group1 = tic.ix[tic.sold != 0]
group2 = tic.ix[tic.sold != 1]

groups1 = group1.groupby(['days till','VIP','weekend']).ticketPrice.mean().reset_index(name='avg sold price')
groups2 = group2.groupby(['days till','VIP','weekend']).ticketPrice.mean().reset_index(name='avg unsold price')


tickes = pd.merge(groups1, tic, on=['days till','VIP','weekend'], how='outer')
ticket = pd.merge(groups2, tickes, on=['days till','VIP','weekend'], how='outer')
ticket=ticket.fillna(0)


# In[ ]:



