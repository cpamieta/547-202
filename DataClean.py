
# coding: utf-8

# In[1]:


#from pyspark import SparkContext
#sc =SparkContext()
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


# In[2]:


#Search Rank obtained from google
ranks1 = pd.read_csv('weekend1.csv', sep=",", header=1)
ranks2 = pd.read_csv('weekend2.csv', sep=",", header=1)
                                                          
#Raw data from ebay api
data = pd.read_csv('ticketdata.txt', sep="\t", header=None, encoding='latin1')
data.columns = ["Title", "Price", "startTime", "endTime","link","subtitle","sold"]


# In[3]:


data.head()


# # Feature Engineering
# The main goal here is to extract useful information from the auction title.
# 
# First thing that was done was to  research the ticket options provided on the coachella website. Next was to see if some type of grouping of the tickets could be established. 
# 
# 
# 

# In[4]:


#New features added from my research, dummy values added. 
data["# of ticket"] = 1
data["Shuttle Passes"] = 0
data["CAR CAMPING PASS"] = 0
data["VIP"] = 0
data["profit"] = -1000000.00
data["days till"] = -100
data["ticketPrice"] = data["Price"]
data["VIP Parking"] = 0
data["weekend1"] = 0
data["weekend2"] = 0
data["PriceRange"] = 0
data["profitpercent"] = 0.00
data["profitgain"] = 0
data["profitloss"] = 0
data["auctionLength"] = 0
data['Title'] = data['Title'].astype(str)
data['subtitle'] = data['subtitle'].astype(str)
data["trend"] = 0

#Used to search for key words like VIP. This was done making a copy of the title data 
#and removing some data to make searching easier. 
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
    
    #total auction length
    a = datetime.strptime(data["startTime"][x][0:10], date_format)
    b = datetime.strptime(data["endTime"][x][0:10], date_format)    
    data["auctionLength"][x] = int((b - a).days)    
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

        
       #If found calculate how many days till the event when the auction ended. 
    if(data["Title"][x].lower().find("weekend one")>=0):      
        data["weekend1"][x] = 1                                             
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week1, date_format)    
        data["days till"][x] = int((b - a).days)       
        y=0
        while (y< len(ranks1)):        
            if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
            y=y+1        
          
    if(data["Title"][x].lower().find("weekend 1")>=0):      
        data["weekend1"][x] = 1
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week1, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks1)):        
            if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
            y=y+1
            
    if(data["Title"][x].lower().find("weekend two")>=0):      
        data["weekend2"][x] = 1
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks2)):          
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]    
            y=y+1
        
        
    if(data["Title"][x].lower().find("weekend 2")>=0):      
        data["weekend2"][x] = 1
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days)
        y=0
        while (y< len(ranks2)):    
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]   
            y=y+1
        
        
    if(data["Title"][x].lower().find("Wknd 2")>=0):      
        data["weekend2"][x] = 1
        a = datetime.strptime(data["endTime"][x][0:10], date_format)
        b = datetime.strptime(week2, date_format)    
        data["days till"][x] = int((b - a).days) 
        y=0
        while (y< len(ranks2)):          
            if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
                data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]
                y=y+1
        
    if(data["Title"][x].lower().find("Wknd 1")>=0):      
        data["weekend2"][x] = 1
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
            data["profitgain"][x] = 1
        elif(data["profit"][x]==0):
            data["profitpercent"][x] = 0       
        elif(data["profit"][x]<0):
            data["profitpercent"][x] =data["profit"][x]/999
            data["profitloss"][x] = 1                
        data["PriceRange"][x] = int(data["ticketPrice"][x] / 50)

    else:  
        data["ticketPrice"][x]= data["ticketPrice"][x] / (data["# of ticket"][x])
        data["profit"][x] = data["ticketPrice"][x] - 429
        data["PriceRange"][x] = int(data["ticketPrice"][x] / 50)
        if(data["profit"][x]>0):
            data["profitpercent"][x] = data["profit"][x]/429
            data["profitgain"][x] = 1
        elif(data["profit"][x]==0):
            data["profitpercent"][x] = 0        
        elif(data["profit"][x]<0):
            data["profitpercent"][x] =data["profit"][x]/429
            data["profitloss"][x] = 1
    x = x+1

#After all the cleaning if a entry in a dataframe had weekend value as 0,that means that its junk data or the 
#listing didnt mention the date. Didnts see a auction of a ticket without a date.
tic = data[data['weekend1'] != 0 | (data['weekend2'] != 0)]


#Group by to find average price
#groups = tickets
group1 = tic.ix[tic.sold != 0]
group2 = tic.ix[tic.sold != 1]

groups1 = group1.groupby(['days till','VIP','weekend1','weekend2']).ticketPrice.mean().reset_index(name='avg sold price')
groups2 = group2.groupby(['days till','VIP','weekend1','weekend2']).ticketPrice.mean().reset_index(name='avg unsold price')


tickes = pd.merge(groups1, tic, on=['days till','VIP','weekend1','weekend2'], how='outer')
ticket = pd.merge(groups2, tickes, on=['days till','VIP','weekend1','weekend2'], how='outer')
ticket=ticket.fillna(0)


# In[5]:


#check for nulls
print('columns with null values:\n', ticket.isnull().sum())


# In[7]:


ticket


# # Data Visualization
# 
# The goal here is to see if a particular feature plays a bigger role in the price or not.
# I also suspect the the ticket price would be going up until they reach a certain point than decline. I suspect this from what I witness when I was looking to buy or sell tickets. 
# 
# It would also be interesting to see how week one compares to week two. Both weekends have the exact same acts playing, so I would suspect the prices to be similar.

# # Average price vs Time
# See how Average sold ticket price changes over time. Looking at the graphs, one can see that the first weekend is more populare. People are willing to spend more money for weekend one. I found this surpising since both weekends have the exact same artist playing. Looks like people are willing to pay more so they could be first.

# In[17]:


#daily avg price for tickets
group1 = ticket.ix[ticket.sold != 0]
wk1 = group1.ix[group1.VIP != 1]
wk1 = wk1.ix[wk1.weekend1 == 1]
wk1 = wk1.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


wk2 = group1.ix[group1.VIP != 1]
wk2 = wk2.ix[wk2.weekend2 == 1]
wk2 = wk2.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


plt.title('avg daily sold price WK 1 vs wk2')
plt.ylabel('Average Price')
plt.xlabel('Days till the Event')

plt.plot(wk1['days till'],wk1['avg sold price'],label="Weekend 1")

plt.plot(wk2['days till'],wk2['avg sold price'] ,label="Weekend 2")
plt.legend(loc='best')
plt.show()


# In[18]:


#daily avg price for VIP tickets

wk2v = group1.ix[group1.VIP != 0]
wk2v = wk2v.ix[wk2v.weekend2 == 1]
wk2v = wk2v.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

wk1v = group1.ix[group1.VIP != 0]
wk1v = wk1v.ix[wk1v.weekend1 == 1]
wk1v = wk1v.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

plt.title('avg daily sold VIP price WK 1 vs WK2')
plt.ylabel('Average Price')
plt.xlabel('Days till the Event')
plt.plot(wk2v['days till'],wk2v['avg sold price'],label="Weekend 2")
plt.plot(wk1v['days till'],wk1v['avg sold price'],label="Weekend 1")
plt.legend(loc='best')
plt.show()


# # Google trend
# Data optained from google search trend. I used Coachella weekend one and Coachella weekend two. 
# The data shows hope popular the words are in regards to google searches. The hypothesis I had was that if more people are searching about coachella, than the prices of the tickets would go up.(supply vs demand).
# 

# In[27]:


from datetime import datetime

date_format = "%Y-%m-%d"
week1 = '2018-04-13'
week2 = '2018-04-20'
y=0
while (y< len(ranks1)):
    
        
    a = datetime.strptime(ranks2["Day"][y], date_format)
    b = datetime.strptime(week2, date_format)    
    ranks2["Day"][y] = int((b - a).days)
    y=y+1
ranks2=ranks2.rename(index=str, columns={"Day": "days till"})
priceTrend2 = pd.merge(ranks2, wk2, on="days till")
   


# In[30]:


#Google trend week one Vs week two
group1 = ticket.ix[ticket.sold != 0]
wk1 = group1.ix[group1.VIP != 1]
wk1 = wk1.ix[wk1.weekend1 == 1]
wk1 = wk1.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


wk2 = group1.ix[group1.VIP != 1]
wk2 = wk2.ix[wk2.weekend2 == 1]
wk2 = wk2.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


plt.title('Google Search trend')
plt.ylabel('Trend ranking out of 100')
plt.xlabel('Days till the Event')

#plt.plot(priceTrend['days till'],priceTrend['coachella weekend 1: (Worldwide)'],label="Weekend 1")
plt.plot(priceTrend2['days till'],priceTrend2['coachella weekend 2: (Worldwide)'],label="Weekend 2")

plt.legend(loc='best')
plt.show()


# # Fixing skew in data.

# In[21]:


# check the skew of all the values
from scipy.stats import norm, skew #for some statistics
numeric_feats = ticket.dtypes[ticket.dtypes != "object"].index


skewed_feats = ticket[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(300)



# In[5]:


ticket.dtypes


# In[22]:


# Statistics
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
from scipy.special import boxcox1p
import numpy as np

skewed_features = skewness.index
for feat in skewed_features:
    #only fix these since the rest are pretty much just one hot encoding
    if(feat == "days till" or feat == "avg unsold price" or feat == "avg sold price" or feat == "profitpercent" or feat == "profit" or feat == "ticketPrice" or feat == "Price" or feat == "auctionLength" or feat == "PriceRange" or feat == "profitpercent"): 
        
        
        #need to do +100 becasue of 0 values.
        ticket[feat] = np.log(ticket[feat]+100)



#skewed_feats = cleaner[numeric_feats].apply(lambda x: np.log(x.dropna())).sort_values(ascending=False)
skewed_feats = ticket[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)

print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(300)








# # Heatmap 
# to see what features play the biggest role.

# In[26]:


# Create a heatmap correlation to find relevant variables
import seaborn as sns

corr = ticket.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()


# In[25]:


# Select columns with a correlation > 0.2 that impact profit
rel_vars = corr.profit[(corr.profit > 0.2)]
rel_cols = list(rel_vars.index.values)

corr2 = ticket[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()

corr2


# # Saving the cleaned data
# two versions of the cleaned data was saved, One with the log transformed and another without it.

# In[10]:


ticket.to_pickle('ticketwithlog.pkl')    #to save the dataframe, df to 123.pkl
#ticket = pd.read_pickle('ticketwithoutlog.pkl')

