import pickle
import os
import sys
import pandas as pd
import numpy as np
import sys

#"# of ticket", "CAR CAMPING PASS", "VIP", "VIP Parking", "weekend","days till","PriceRange",
 #                     "ticketPrice","trend","Price","Shuttle Passes","avg unsold price","avg sold price"  

ticket = pd.read_csv('C:/xampp/htdocs/ticketpredictCopy/ticketDatafinal.csv', sep="\t", header=0)

ticketPrice = float(sys.argv[5])

print(sys.argv[0])
print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])
print(sys.argv[4])
print(sys.argv[5])
group1 = ticket.ix[ticket['days till'] == int(sys.argv[4])]
group1 = group1.ix[group1['VIP'] == int(sys.argv[2])]
group1 = group1.ix[group1['weekend'] == int(sys.argv[3])]
group11 =group1.reset_index(drop=True)

avgSold = group11["avg sold price"][0]
avgUnsold = group11["avg unsold price"][0]
trend = group11["trend"][0]

if(int(sys.argv[6]) == 1):
    ticketPrice=ticketPrice-75
if(int(sys.argv[1]) == 1):
    ticketPrice=ticketPrice-113
    
PriceRange = ticketPrice/25

zz= np.array([[float(1),float(sys.argv[1]),float(sys.argv[2]),float(0),float(sys.argv[3]),float(sys.argv[4])
               ,float(PriceRange),float(ticketPrice),float(trend),float(sys.argv[5]),float(sys.argv[6]),float(avgUnsold),float(avgSold)]])


z= np.array([[1,0,0,0,2,2,12,300,41,375,1,532.299,488.427]])





# load the model from disk
f = open("C:/xampp/htdocs/ticketpredictCopy/RandomForestModel", 'rb')
#RandomForestModel = pickle.load(open("C:/xampp/htdocs/ticketpredictCopy/RandomForestModel", 'rb'))
predicted = f.predict(zz)


print(predicted)




