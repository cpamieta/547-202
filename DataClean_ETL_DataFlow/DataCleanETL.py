#!/usr/bin/python
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
import pandas as pd
import csv
from datetime import datetime
import matplotlib.pyplot as plt
# Statistics
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
import numpy as np

def dataClean(data,ranks1,ranks2):
    print("shit")
    print("shit")
    #Search Rank obtained from google
  #  ranks1 = pd.read_csv('weekend1.csv', sep=",", header=1)
   # ranks2 = pd.read_csv('weekend2.csv', sep=",", header=1)
    ranks1=pd.DataFrame(ranks1)
    print(ranks1)
    #Raw data from ebay api
   # data = pd.read_csv('coachellaticks.csv',header=None, encoding='latin1')
    #dat = data.split(',')
    data = pd.DataFrame([data]) 
    data.columns =["Title", "Price", "startTime", "endTime","link","subtitle","shippingCost","sold"]
    data['Price'] = data['Price'].astype(float)
    data['shippingCost'] = data['shippingCost'].astype(float)
    # # Feature Engineering
    # The main goal here is to extract useful information from the auction title.
    # 
    # First thing that was done was to  research the ticket options provided on the coachella website. Next was to see if some type of grouping of the tickets could be established. 
    # 
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
          #  while (y< len(ranks1)):        
          #      if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
          #          data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
          #      y=y+1        

        if(data["Title"][x].lower().find("weekend 1")>=0):      
            data["weekend1"][x] = 1
            a = datetime.strptime(data["endTime"][x][0:10], date_format)
            b = datetime.strptime(week1, date_format)    
            data["days till"][x] = int((b - a).days)
            y=0
         #   while (y< len(ranks1)):        
         #       if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
         #           data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
         #       y=y+1

        if(data["Title"][x].lower().find("weekend two")>=0):      
            data["weekend2"][x] = 1
            a = datetime.strptime(data["endTime"][x][0:10], date_format)
            b = datetime.strptime(week2, date_format)    
            data["days till"][x] = int((b - a).days)
            y=0
          #  while (y< len(ranks2)):          
           #     if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
            #        data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]    
             #   y=y+1

        if(data["Title"][x].lower().find("weekend 2")>=0):      
            data["weekend2"][x] = 1
            a = datetime.strptime(data["endTime"][x][0:10], date_format)
            b = datetime.strptime(week2, date_format)    
            data["days till"][x] = int((b - a).days)
            y=0
    #        while (y< len(ranks2)):    
     #           if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
      #              data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]   
       #         y=y+1

        if(data["Title"][x].lower().find("Wknd 2")>=0):      
            data["weekend2"][x] = 1
            a = datetime.strptime(data["endTime"][x][0:10], date_format)
            b = datetime.strptime(week2, date_format)    
            data["days till"][x] = int((b - a).days) 
            y=0
      #      while (y< len(ranks2)):          
       #         if(ranks2['Day'][y] ==data["endTime"][x][0:10]):
        #            data["trend"][x] = ranks2['coachella weekend 2: (Worldwide)'][y]
        #            y=y+1

        if(data["Title"][x].lower().find("Wknd 1")>=0):      
            data["weekend2"][x] = 1
            a = datetime.strptime(data["endTime"][x][0:10], date_format)
            b = datetime.strptime(week2, date_format)    
            data["days till"][x] = int((b - a).days)         
            y=0
     #       while (y< len(ranks1)):       
      #          if(ranks1['Day'][y] ==data["endTime"][x][0:10]):
       #             data["trend"][x] = ranks1['coachella weekend 1: (Worldwide)'][y]
        #        y=y+1

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
            data["ticketPrice"][x]= data["ticketPrice"][x].astype('float')/(data["# of ticket"][x].astype('float'))
            data["profit"][x] = data["ticketPrice"][x]- 999
           #Calculate the profit made or lost on the ticket 
            if(data["profit"][x]>0):
                data["profitpercent"][x] = data["profit"][x].astype('float')/999
                data["profitgain"][x] = 1
            elif(data["profit"][x]==0):
                data["profitpercent"][x] = 0       
            elif(data["profit"][x]<0):
                data["profitpercent"][x] =data["profit"][x].astype('float')/999
                data["profitloss"][x] = 1                
            data["PriceRange"][x] = int(data["ticketPrice"][x].astype('float') / 50)

        else:  
            data["ticketPrice"][x]= data["ticketPrice"][x].astype('float') / (data["# of ticket"][x])
            data["profit"][x] = data["ticketPrice"][x] - 429
            data["PriceRange"][x] = int(data["ticketPrice"][x].astype('float') / 50)
            if(data["profit"][x]>0):
                data["profitpercent"][x] = data["profit"][x].astype('float')/429
                data["profitgain"][x] = 1
            elif(data["profit"][x]==0):
                data["profitpercent"][x] = 0        
            elif(data["profit"][x]<0):
                data["profitpercent"][x] =data["profit"][x].astype('float')/429
                data["profitloss"][x] = 1
        x = x+1

    #After all the cleaning if a entry in a dataframe had weekend value as 0,that means that its junk data or the 
    #listing didnt mention the date. Didnts see a auction of a ticket without a date.
    tic = data[data['weekend1'] != 0 | (data['weekend2'] != 0)]

    #Group by to find average price
    #groups = tickets
    #group1 = tic.iloc[tic.sold != 0]
    #group2 = tic.iloc[tic.sold != 1]

    #groups1 = group1.groupby(['days till','VIP','weekend1','weekend2']).ticketPrice.mean().reset_index(name='avg sold price')
    #groups2 = group2.groupby(['days till','VIP','weekend1','weekend2']).ticketPrice.mean().reset_index(name='avg unsold price')


    #tickes = pd.merge(groups1, tic, on=['days till','VIP','weekend1','weekend2'], how='outer')
    #ticket = pd.merge(groups2, tickes, on=['days till','VIP','weekend1','weekend2'], how='outer')
    #ticket=ticket.fillna(0)
    tic=tic.fillna(0)




    date_format = "%Y-%m-%d"
    week1 = '2018-04-13'
    week2 = '2018-04-20'
    y=0
    #while (y< len(ranks1)):


 #       a = datetime.strptime(ranks2["Day"][y], date_format)
  #      b = datetime.strptime(week2, date_format)    
   #     ranks2["Day"][y] = int((b - a).days)
    #    y=y+1
#    ranks2=ranks2.rename(index=str, columns={"Day": "days till"})
 #   priceTrend2 = pd.merge(ranks2, wk2, on="days till")




    # # Fixing skew in data.
    # check the skew of all the values
    #numeric_feats = ticket.dtypes[ticket.dtypes != "object"].index
    #skewed_feats = ticket[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
    #skewness = pd.DataFrame({'Skew' :skewed_feats})

    #skewed_features = skewness.index
    #for feat in skewed_features:
        #only fix these since the rest are pretty much just one hot encoding
   #     if(feat == "days till" or feat == "avg unsold price" or feat == "avg sold price" or feat == "profitpercent" or feat == "profit" or feat == "ticketPrice" or feat == "Price" or feat == "auctionLength" or feat == "PriceRange" or feat == "profitpercent"): 


            #need to do +100 becasue of 0 values.
    #        ticket[feat] = np.log(ticket[feat]+100)



    #skewed_feats = cleaner[numeric_feats].apply(lambda x: np.log(x.dropna())).sort_values(ascending=False)
#    skewed_feats = ticket[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)

    #skewness = pd.DataFrame({'Skew' :skewed_feats})




    # # Saving the cleaned data
   # ticket.to_pickle('ticketwithlog.pkl')    #to save the dataframe, df to 123.pkl
    #ticket = pd.read_pickle('ticketwithoutlog.pkl')

    yield ''.join(data.to_csv( index=False))


def run(project, bucket, dataset):
    
    

    with beam.Pipeline('DirectRunner') as pipeline:
        weekend1 = (pipeline
#The ReadFromText transform returns a PCollection, which contains all lines from the file.
         | 'weekend1:read' >> beam.io.ReadFromText(file_pattern="gs://ticket-prediction.appspot.com/RawTicketData/weekend1.csv")
         | 'weekend1:fields' >> beam.Map(lambda line: next(csv.reader([line])))
      )
        weekend2 = (pipeline
#The ReadFromText transform returns a PCollection, which contains all lines from the file.
         | 'weekend2:read' >> beam.io.ReadFromText(file_pattern="gs://ticket-prediction.appspot.com/RawTicketData/weekend2.csv")
         | 'weekend2:fields' >> beam.Map(lambda line: next(csv.reader([line])))
      )

        tickets = (pipeline
#The ReadFromText transform returns a PCollection, which contains all lines from the file.
         | 'tickets:read' >> beam.io.ReadFromText(file_pattern="gs://ticket-prediction.appspot.com/RawTicketData/test.csv")
         | 'tickets:fields' >> beam.Map(lambda line: next(csv.reader([line])))
         | 'tickets:clean' >> beam.FlatMap(dataClean,beam.pvalue.AsList(weekend1),beam.pvalue.AsList(weekend2))

      )
# use group by to group data than a map to fin avg or something
    #  flights = (pipeline
    #     | 'flights:read' >> beam.io.ReadFromText('201501_part.csv')
    #     | 'flights:tzcorr' >> beam.FlatMap(dataClean, beam.pvalue.AsDict(tickets))
    #  )
        print("test")
        tickets | beam.io.textio.WriteToText("gs://ticket-prediction.appspot.com/RawTicketData/output_file.csv",file_name_suffix='.csv')
        pipeline.run()
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Run pipeline on the cloud')
    parser.add_argument('-p','--project', help='Unique project ID', required=True)
    parser.add_argument('-b','--bucket', help='Bucket where your data were ingested in Chapter 2', required=True)
    parser.add_argument('-d','--dataset', help='BigQuery dataset', default='flights')
    args = vars(parser.parse_args())

    print ("Correcting timestamps and writing to BigQuery dataset {}".format(args['dataset']))
    run(project=args['project'], bucket=args['bucket'], dataset=args['dataset'])
    #run()











