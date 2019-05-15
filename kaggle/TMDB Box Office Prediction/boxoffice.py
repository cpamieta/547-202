
# coding: utf-8

# # TMDB Box Office Prediction
# Can you predict a movie's worldwide box office revenue?

# In[1]:


#note: probably not all of these imports are used
import wptools

import time
from omdb import OMDBClient
import re
import tmdbsimple as tmdb
import ast
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
from scipy.special import boxcox1p,inv_boxcox1p,boxcox
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.metrics import mean_squared_log_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import itertools


# In[2]:


import math


get_ipython().magic('matplotlib inline')
from sklearn.metrics import r2_score, make_scorer, mean_squared_error 

from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, Ridge, RidgeCV, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from catboost import Pool, CatBoostRegressor, cv
from sklearn import cross_validation,preprocessing,tree
from sklearn.metrics import r2_score, make_scorer, mean_squared_error 


# In[3]:



#load the data, and create a new column to be used to combine to one df and perform clean 
train = pd.read_csv('train.csv', header = 0, )
train['train'] =1


test  = pd.read_csv('test.csv' , header = 0, )
test['train'] =0

#combin train and test to do EDA
cleaner = pd.concat([train,test]).reset_index()

moreData= pd.read_csv('additionalTrainData.csv' , header = 0,encoding = "ISO-8859-1" )
#dropped all rows without imdb since its needed for the apis below. This is just the extra training data.
moreData=moreData.dropna(subset = ['imdb_id'])
moreData=moreData.reset_index()
moreData['genres'] = 'none'
moreData['spoken_languages'] = 'none'
moreData['production_companies'] = 'none'
moreData['production_countries'] = 'none'

  


# In[113]:


#Loading the final form of the dataset after gathering more data from the three apis listed below. 
#cleaner  = pd.read_csv('data4ths.csv' , header = 0,encoding = "ISO-8859-1" )

moreData = pd.read_pickle('combineAPILookUpExtraThird.pkl')
cleaner = pd.read_pickle('OMDBLookUpFirst.pkl')



# In[8]:


print('columns with null values:\n', cleaner.isnull().sum())


# In[114]:


#check for null
print('columns with null values:\n', moreData.isnull().sum())



#cleaner.loc[cleaner['budget'] == 0]['imdb_id']


# # Replacing null values

# In[ ]:



cleaner['belongs_to_collection'].fillna('none', inplace = True)
cleaner['cast'].fillna('none', inplace = True)
cleaner['crew'].fillna('none', inplace = True)
cleaner['Keywords'].fillna('none', inplace = True)

cleaner['genres'].fillna('none', inplace = True)
cleaner['homepage'].fillna('none', inplace = True)
cleaner['overview'].fillna('none', inplace = True)
cleaner['poster_path'].fillna('none', inplace = True)
cleaner['production_companies'].fillna('none', inplace = True)
cleaner['production_countries'].fillna('none', inplace = True)
cleaner['release_date'].fillna('0', inplace = True)
cleaner['runtime'].fillna(0.0, inplace=True)
cleaner['spoken_languages'].fillna('none', inplace = True)
cleaner['status'].fillna('none', inplace = True)
cleaner['tagline'].fillna('none', inplace = True)
#Just use homemade url as title
cleaner['title'].fillna(cleaner['homepage'], inplace = True)



# In[20]:


#extra data
moreData['runtime'].fillna(0.0, inplace=True)


# # Search for missing data and also looking for new features.
# For this, I used three apis.
# 
# 1)https://github.com/celiao/tmdbsimple/ which is a python wrapper for TMDb
# 
# 2)https://github.com/dgilland/omdb.py is a python wrapper for OMDb, which access imdb website.
# 
# 3) https://pypi.org/project/wikipedia/  https://github.com/siznax/wptools     is a python wrapper for wikipedia
# 
# Each website provides data that the others might be missing. For example, alot of wikipedia pages lists a budget for the movie while the other two dont list one.
# 
# Note: Running these api took me hours to complete. Took well over 24 hours, so I would dump the results using pickle which I could just load back the data instead of gathering the data again. 

# In[18]:


#gets data from imdb website.
#To find missing values for runtime,genres,release_date,spoken_languages,production_companies, and production_countries
#Get two new fields, avg rating and vote count

def OMDBLookUp(cleaner):
    
    


    client = OMDBClient(apikey='7046848d')
    xx=0
    cleaner['rating'] = 0
    cleaner['votes'] = 0


    while(len(cleaner)>xx):
        print(cleaner['imdb_id'][xx])
        
        #1000 call limit a day so need to cycle api keys
        if(xx==997):
            
            client = OMDBClient(apikey='587318e6')
        elif(xx==1900):
            client = OMDBClient(apikey='a56b78cb')
        elif(xx==2900):
            client = OMDBClient(apikey='234fa744')
        elif(xx==3900):
            client = OMDBClient(apikey='60b74f3c')            
        elif(xx==4900):
            client = OMDBClient(apikey='8ba8a4d5') 
        
        
        
        res = client.imdbid(cleaner['imdb_id'][xx])
        xml_content = res
        avg = 0.0
        #sum up the rating for all provided sources
        for x in res['ratings']:
            
            
            if(x['source'] == "Internet Movie Database"):
        
                avg=avg+ (float(x['value'][0:int(x['value'].find('/'))]))*10

        
            elif(x['source'] == "Rotten Tomatoes"):
                avg=avg + float(re.sub("[^0-9]", "", x['value']))
            elif(x['source'] == "Metacritic"):
                avg=avg + float(x['value'][0:int(x['value'].find('/'))])
        if(avg != 0.0):
            
            cleaner['rating'][xx] = avg/ len(res['ratings'])
        votes = re.sub("[^0-9]", "", res['imdb_votes'])
        if(len(votes)>0):
            cleaner['votes'][xx] = int(votes)

        
        
        
        
        if(cleaner['runtime'][xx] == 0.0):

        
            cleaner['runtime'][xx] = re.sub("[^0-9]", "", xml_content['runtime'])
        
        
        
        if(cleaner['genres'][xx] == 'none'):

            #split sting into list of genres
            genre = xml_content['genre'].split(",")
        
            for y in genre:
                y = y.lstrip(' ')
            
                if y in cleaner.columns:         
    
                    cleaner[y][xx] = 1    
                else:
            
                    cleaner[y] = 0
                    cleaner[y][xx] = 1
    
        

        if(cleaner['release_date'][xx] == '0'):


            cleaner['release_date'][xx] =xml_content['released']

        if(cleaner['spoken_languages'][xx] == 'none'):


            #added a try block since some movies would not have a language object from the website.
            try:
                language = xml_content['language'].split(",")

                for y in language:
                    y = y.lstrip(' ')

                    if y in cleaner.columns:         

                        cleaner[y][xx] = 1    
                    else:

                        cleaner[y] = 0
                        cleaner[y][xx] = 1
            except:
                pass
        if(cleaner['production_companies'][xx] == 'none'):


            try:
                production = xml_content['production'].split(",")

                for y in production:
                    y = y.lstrip(' ')

                    if y in cleaner.columns:         

                        cleaner[y][xx] = 1    
                    else:

                        cleaner[y] = 0
                        cleaner[y][xx] = 1

            except:
                pass

        if(cleaner['production_countries'][xx] == 'none'):

            try:
                country = xml_content['country'].split(",")

                for y in country:
                    y = y.lstrip(' ')

                    if y in cleaner.columns:         

                        cleaner[y][xx] = 1    
                    else:

                        cleaner[y] = 0
                        cleaner[y][xx] = 1

            except:
                pass


        #added a sleep timer since if I hit the site to frequently, it would lock me out.
        time.sleep(1)         
        xx=xx+1  
    return cleaner


# In[12]:


# for tmdb, the goal was to create two new features, 
#movie rating and total number of countries that the movie was released in.
def tmbdFeatures(cleaner):
    tmdb.API_KEY = 'a995d49573e4ba48c6a6ea762a6cfd75'
    xx=0
    cleaner['NumReleaseCountry']=0
    
    cleaner['Rated']=0

    while(len(cleaner)>xx):
        print(cleaner['title'])


        for x in cleaner['title']:
            print(x)
            #added a 7 second wait time since to many calls within a certain time would block your call to the api
            time.sleep(7) 
            search = tmdb.Search()



            response = search.movie(query=x)


            totalCountry = 0
            for s in search.results:
                movie = tmdb.Movies(s['id'])
                try:
                    response = movie.info()

                    if(response['imdb_id'] != None and response['imdb_id'].lower().rstrip() == cleaner['imdb_id'][xx].lower().rstrip()):

                        movie = tmdb.Movies(s['id'])
                        response = movie.releases()
                        cleaner['NumReleaseCountry'][xx] = len(movie.countries)

                        for c in movie.countries:
                                if c['iso_3166_1'] == 'US':

                                    cleaner['Rating'][xx]=c['certification']

                except:
                    print(s['id'])

                    pass 

            xx=xx+1  
    
    return cleaner


# In[7]:


#From the two apis, I ran into some issues that created not found conditions when looking up for the movie.
#I notice some of the movie names in the provided data has some invalid char, so I combined both api in here. 
# I also notice that omdb only allows for 1000 calls a day which I hit and so would skip some of the movies. 
#I later got more keys from the api so I would avoid that daily limit.
#Note 'Unnamed: 0' is a index column that was generated when I read in the dumped pickle files, so I used that 
#to represent the row index in the method below. 
def combineAPI(cleaner):
    cleaner['Rating'].fillna('0', inplace = True)
    cleaner['NumReleaseCountry'].fillna(0, inplace = True)




    
    client = OMDBClient(apikey='a56b78cb')
    tmdb.API_KEY = 'a995d49573e4ba48c6a6ea762a6cfd75'

    cleaner['Rating'].fillna('0', inplace = True)
    cleaner['NumReleaseCountry'].fillna(0, inplace = True)
    count = 0
    for index, row in cleaner.loc[(cleaner['NumReleaseCountry'] == 0) | (cleaner['Rating'] == '0') | (cleaner['budget'] == 0)].iterrows():
        time.sleep(3)  
        if(count==997):
            
            
            client = OMDBClient(apikey='a56b78cb')
        count = count +1
        try:
            res = client.imdbid(row['imdb_id'])    
            xml_content = res

            if((row['Rating'] == '0') &  (xml_content['rated'].lower().rstrip() != 'n/a') & (xml_content['rated'].lower().rstrip() != 'not rated') & (xml_content['rated'].rstrip() != '') & (xml_content['rated'].lower().rstrip() != 'unrated') & (xml_content['rated'].lower().rstrip() != 'nr')):
                print(xml_content['rated'])
                cleaner['Rating'][row['Unnamed: 0']] = xml_content['rated']
                search = tmdb.Search()

                response = search.movie(query=xml_content['title'])
                for s in search.results:



                    movie = tmdb.Movies(s['id'])
                    try:

                        response = movie.info()

                        if(response['imdb_id'] != None and response['imdb_id'].lower().rstrip() == row['imdb_id'].rstrip()):
                            movie = tmdb.Movies(s['id'])
                            response = movie.releases()
                            if(len(movie.countries)> 0):

                                cleaner['NumReleaseCountry'][row['Unnamed: 0']] = len(movie.countries)
                    except:

                        pass 


            else:

                search = tmdb.Search()
                response = search.movie(query=xml_content['title'])



                for s in search.results:
                    movie = tmdb.Movies(s['id'])
                    try:
                        response = movie.info()





                        if(response['imdb_id'] != None and response['imdb_id'].lower().rstrip() == row['imdb_id'].rstrip()):

                            movie = tmdb.Movies(s['id'])
                            response = movie.releases()
                            if(len(movie.countries)> 0):
                                cleaner['NumReleaseCountry'][row['Unnamed: 0']] = len(movie.countries)




                        #print(len(movie.countries))
                            for c in movie.countries:
                                    if c['iso_3166_1'] == 'US':
                                        cleaner['Rating'][row['Unnamed: 0']]=c['certification']


                    except:

                        pass 




        except:
            print(row['imdb_id'])


            pass           
    return cleaner


# # Getting missing budget

# In[3]:


#find missing budget
def wikiBudget(movieName):     
    #Some movies in wiki have (film) appended to it while others dont. Created a try/catch to search both ways.
    try:
        so = wptools.page(movieName.rstrip() + ' (film)').get_parse()
        budget = so.data['infobox']['budget']
        #budget = re.sub("[^0-9]", "", budget)
        

        return budget
     
            
    except:
        try:
            
            
            so = wptools.page(movieName.rstrip()).get_parse()
            budget = so.data['infobox']['budget']
            return budget
        
        except:
                
            pass 
 


# In[4]:


#formating budget. Goal of this function is to format all the newly found budget to usd. 
#not taking into account inflation.
def budgetformats(budget):

    formatBudget = 0.0
    if(budget.lower().find("million")>=0):
        
        if(budget.lower().find("¥")>=0):            
        #convert to usd
            start = budget.find('¥') + 1
            end = budget.find('million', start)
            bud = budget[start:end]
            if(bud.lower().find("-")>=0):  
                
                
                
                x= (len(bud)-1)/2
                
                   
                end = bud.find('-') + 1
                start = bud.find('-') - int(x)
                try:
                    avg = (float(bud[start:int(x)]) + float(bud[end:]))/2
                    mill = avg * 1000000.0
                #convert to usd
                    usd = float(mill)* 0.0089
            #    print(usd)
                    formatBudget = usd                
                except:
                    
                    pass
                
                
            else:
                #convert to millions
                mill = int(re.sub("[^0-9]", "", bud)) * 1000000
                #convert to usd
                usd =  float(mill) * 0.0089
             #   print(usd)

                formatBudget = usd         
        
        
        if(budget.lower().find("HK$")>=0):
        #convert to usd
            start = budget.find('HK$') + 1
            end = budget.find('million', start)
            bud = budget[start:end]
            if(bud.lower().find("-")>=0):               
                   
                x= (len(bud)-1)/2
                
                   
                end = bud.find('-') + 1
                start = bud.find('-') - int(x)
                try:
                    avg = (float(bud[start:int(x)]) + float(bud[end:]))/2
                    mill = avg * 1000000.0
                #convert to usd
                    usd = float(mill) * 0.13
              #  print(usd)

                    formatBudget = usd
                except:
                    
                    pass                
                
            else:
                #convert to millions
                mill = int(re.sub("[^0-9]", "", bud)) * 1000000
                #convert to usd
                usd =  float(mill) * 0.13
             #   print(usd)
                
                formatBudget = usd                                
        if(budget.lower().find("£")>=0):
        #convert to usd
            start = budget.find('£') + 1
            end = budget.find('million', start)
            bud = budget[start:end]
            if(bud.lower().find("-")>=0):               
                   
                x= (len(bud)-1)/2
                
                   
                end = bud.find('-') + 1
                start = bud.find('-') - int(x)
                try:
                    avg = (float(bud[start:int(x)]) + float(bud[end:]))/2
                    mill = avg * 1000000.0
                #convert to usd
                    usd = float(mill) * 1.3
              #  print(usd)

                    formatBudget = usd                
                except:
                    
                    pass
                
                
            else:
                #convert to millions
                mill = int(re.sub("[^0-9]", "", bud)) * 1000000
                #convert to usd
                usd = float(mill) * 1.3
               # print(usd)
                
                formatBudget = usd                                
                
        if(budget.lower().find("€")>=0):
        #convert to usd
            start = budget.find('€') + 1
            end = budget.find('million', start)
            bud = budget[start:end]
            if(bud.lower().find("-")>=0):               
                   
                x= (len(bud)-1)/2
                
                   
                end = bud.find('-') + 1
                start = bud.find('-') - int(x)
                try:
                    avg = (float(bud[start:int(x)]) + float(bud[end:]))/2
                    mill = avg * 1000000.0
                #convert to usd
                    usd = float(mill) * 1.13
              #  print(usd)

                    formatBudget = usd                
                except:
                    
                    pass
                
                
            else:
                #convert to millions
                mill = int(re.sub("[^0-9]", "", bud)) * 1000000
                #convert to usd
                usd = float(mill) * 1.13
            #    print(usd)

                formatBudget = usd                                
                
                
                
        if(budget.lower().find("$")>=0):
        #convert to usd
            start = budget.find('$') + 1
            end = budget.find('million', start)
            bud = budget[start:end]
            if(bud.lower().find("-")>=0):               
                   
                x= (len(bud)-1)/2
                
                   
                end = bud.find('-') + 1
                start = bud.find('-') - int(x)
                print(bud)
                try:
                    avg = (float(bud[start:int(x)]) + float(bud[end:]))/2
                    print(avg)
                    mill = avg * 1000000
                    print(mill)

                    #convert to usd
                    formatBudget = mill
                except:
                    
                    pass
                
                
            else:
                #convert to millions
                mill = int(re.sub("[^0-9]", "", bud)) * 1000000
             #   print(mill)

                formatBudget = mill
                                

      
    elif(budget.lower().find("¥")>=0):
        #convert to usd
        usd = float(re.sub("[^0-9]", "", budget)) * 0.0089
     #   print(usd)

        formatBudget = usd        
    elif(budget.lower().find("HK$")>=0):
        
                #convert to usd
        usd = float(re.sub("[^0-9]", "", budget)) * 0.13
     #   print(usd)

        budget['cd']['index'] = usd        


    elif(budget.lower().find("£")>=0):
        #convert to usd
        usd = float(re.sub("[^0-9]", "", budget)) * 1.3
       # print(usd)

        formatBudget = usd        
    elif(budget.lower().find("€")>=0):
        #convert to usd
        usd = float(re.sub("[^0-9]", "", budget)) * 1.13
    #    print(usd)

        formatBudget = usd    
    elif(budget.lower().find("$")>=0):
        usd = re.sub("[^0-9]", "", budget) 
     #   print(usd)

        formatBudget = usd        

    return formatBudget




# # Feature engineering

# In[ ]:


OMDBLookUp(moreData)
moreData.to_pickle('OMDBLookUpExtraFirst.pkl')    #to save the dataframe, df to 123.pkl


# In[ ]:


tmbdFeatures(moreData)
moreData.to_pickle('tmbdLookUpExtraSecond.pkl')    #to save the dataframe, df to 123.pkl


# In[ ]:


combineAPI(moreData)
moreData.to_pickle('combineAPILookUpExtraThird.pkl')    #to save the dataframe, df to 123.pkl


# In[7]:


#Second dataset
OMDBLookUp(cleaner)
cleaner.to_pickle('OMDBLookUpFirst.pkl')    #to save the dataframe, df to 123.pkl


# In[ ]:


tmbdFeatures(cleaner)
cleaner.to_pickle('tmbdLookUpSecond.pkl')    #to save the dataframe, df to 123.pkl


# In[ ]:


combineAPI(cleaner)
cleaner.to_pickle('combineAPILookUpThird.pkl')    #to save the dataframe, df to 123.pkl


# In[44]:


#Finding missing budget.
for index, row in cleaner.loc[(cleaner['budget'] == 0)].iterrows():
    bud = wikiBudget(row['title'])
    if(bud is not None):
        print(bud)
        budgetFormat = budgetformats(bud)
        print(budgetFormat)
        cleaner['budget'][row['Unnamed: 0']] = budgetFormat
        #la.loc[x]= bud
        

cleaner.to_pickle('budget.pkl')    #to save the dataframe, df to 123.pkl


# # Parse the json within six fields in the data.
# For this, pretty much one hot encoding the values. So after the json is converted, if the dictionary that is generated has more then 50 rows that has that value, it creates a new field for that value. 
# 

# In[16]:



#This was used to parse the json

from tqdm import tqdm
#json_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
json_cols = ['genres']
#json_cols = ['cast', 'crew']


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

#for col in tqdm(json_cols) :
for col in tqdm(json_cols + ['belongs_to_collection']) :
    cleaner[col] = cleaner[col].apply(lambda x : get_dictionary(x))
    #test[col] = test[col].apply(lambda x : get_dictionary(x))
    #train[col] = train[col].apply(lambda x : get_dictionary(x))

print(cleaner.shape)
cleaner.head()




# In[17]:




# parse json data and build category dictionary
def get_json_dict(df) :
    global json_cols
    result = dict()
    for e_col in json_cols :
        d = dict()
        rows = df[e_col].values
        for row in rows :
            if row is None : continue
            for i in row :
                if i['name'] not in d :
                    d[i['name']] = 0
                d[i['name']] += 1
        result[e_col] = d
    return result

train_dict = get_json_dict(cleaner)


    



# In[10]:



for p_id, p_info in train_dict.items():
    print("\n ID:", p_id)
    
    for key in p_info:
        print(key + ':', p_info[key])
        
        if(p_info[key]>50):
            x=0

            cleaner[key]=0
            while(x<len(cleaner)):
            
                if(str(cleaner[p_id][x]).find(key)>=0):
                    cleaner[key][x]=1
                        
            
                x=x+1






# In[27]:


#drop columns that are not in the main training set
#add columns that are not in the extra data set. Reason I added was that when I merged the two df, would 
#fill missing data with nan and also convert some of the decimals into floats.

for col in moreData:
    dropCol = True
    for colls in cleaner:
        if(colls.lower().rstrip() == col.lower().rstrip()):
            dropCol = False
            

        
    if(dropCol):
        
        moreData=moreData.drop([col], axis=1)

        
for col in cleaner:
    addCol = True
    for colls in moreData:
        if(colls.lower().rstrip() == col.lower().rstrip()):
            dropCol = False        
        
    if(addCol):
        moreData[col] = 0


# In[28]:


moreData['train'] =1


# In[122]:


#split the date  to year month day. Maybe add feature of holiday?

#Two data sets use different date formats
moreData[['day','month','year']] = moreData['release_date'].str.split('-', expand=True)
moreData=moreData.drop(['release_date'], axis=1)

moreData["month"] = moreData["month"].astype('int')
moreData["day"] = moreData["day"].astype('int')
moreData["year"] = moreData["year"].astype('int')


cleaner[['month','day','year']] = cleaner['release_date'].str.split('/', expand=True)
cleaner=cleaner.drop(['release_date'], axis=1)


cleaner["month"] = cleaner["month"].astype('int')
cleaner["day"] = cleaner["day"].astype('int')
cleaner["year"] = cleaner["year"].astype('int')
           


# In[29]:


#merge main training data with the extra data
x=pd.concat([moreData,cleaner])


# In[30]:


# one hot encode rating 
#
cleaner['Rating'].fillna('0', inplace = True)
for index, row in  cleaner.loc[(cleaner['Rating'] == '0')  | (cleaner['Rating'] == '') | (cleaner['Rating'] == 'Not Rated') |  (cleaner['Rating'] == 'NR')].iterrows():
    cleaner['Rating'][row['Unnamed: 0']] = 'Unrated'
    
    
for index, row in  cleaner.loc[(cleaner['Rating'] == 'PASSED')].iterrows():
    cleaner['Rating'][row['Unnamed: 0']] = 'Passed'
    
for index, row in  cleaner.loc[(cleaner['Rating'] == 'APPROVED')].iterrows():
    cleaner['Rating'][row['Unnamed: 0']] = 'Approved'
    
    
print(cleaner['Rating'].unique())
cleaner = pd.concat([cleaner, pd.get_dummies(cleaner["Rating"])], axis=1)  



# In[31]:


#Filling missing runtime. Since only four missing values, I will just manually search for it since the api found nothing.
cleaner.loc[(cleaner['runtime'].isnull())]['imdb_id']
#94 min   98  , 86  87


               
               
for index, row in cleaner.loc[(cleaner['runtime'].isnull())].iterrows():
    
    
    if(row['imdb_id'] =="tt0116485"):
        cleaner['runtime'][row['Unnamed: 0']] = 94.0
    elif(row['imdb_id'] =="tt3956312"):
        cleaner['runtime'][row['Unnamed: 0']] = 98.0
    elif(row['imdb_id'] =="tt1620464"):
        cleaner['runtime'][row['Unnamed: 0']] = 86.0
    elif(row['imdb_id'] =="tt1190905"):
        cleaner['runtime'][row['Unnamed: 0']] = 87.0
            
               
cleaner.loc[(cleaner['runtime'].isnull())]['imdb_id']   


# # Data Cleaning

# In[32]:


# remove some rows that I dont think we need.

cleaner=cleaner.drop(['index'], axis=1)
cleaner=cleaner.drop(['Keywords'], axis=1)
cleaner=cleaner.drop(['belongs_to_collection'], axis=1)
cleaner=cleaner.drop(['genres'], axis=1)

cleaner=cleaner.drop(['Unnamed: 0'], axis=1)

cleaner=cleaner.drop(['imdb_id'], axis=1)
cleaner=cleaner.drop(['homepage'], axis=1)
cleaner=cleaner.drop(['cast'], axis=1)
cleaner=cleaner.drop(['crew'], axis=1)
#maybe one hot encode this
cleaner=cleaner.drop(['original_language'], axis=1)
cleaner=cleaner.drop(['original_title'], axis=1)
cleaner=cleaner.drop(['poster_path'], axis=1)
cleaner=cleaner.drop(['title'], axis=1)

cleaner=cleaner.drop(['production_companies'], axis=1)
cleaner=cleaner.drop(['production_countries'], axis=1)
cleaner=cleaner.drop(['tagline'], axis=1)

                            
cleaner=cleaner.drop(['spoken_languages'], axis=1)
    

cleaner=cleaner.drop(['overview'], axis=1)

cleaner=cleaner.drop(['Unnamed: 0.1'], axis=1)
cleaner=cleaner.drop(['Unnamed: 0.1.1'], axis=1)
cleaner=cleaner.drop(['Rating'], axis=1)


# In[33]:


#convert to int
#cleaner['runtime']= cleaner['runtime'].replace('none', 0)
cleaner["runtime"] = cleaner["runtime"].astype('int')


# In[34]:


cleaner = pd.concat([cleaner, pd.get_dummies(cleaner["status"],prefix=['status'])], axis=1)
cleaner=cleaner.drop(['status'], axis=1)


# In[35]:


#XGBoost does not like having these chars
cleaner.columns=cleaner.columns.str.replace('[','')
cleaner.columns=cleaner.columns.str.replace(']','')
cleaner.columns=cleaner.columns.str.replace('<','')


# # New feature, average revenue.
# The goal with this is create a few features with this. Example, average revenus for the production company at the time of the movie release. 
# 

# In[37]:



    
Action=cleaner.groupby("Action")["revenue"].aggregate('mean')[1]
Adventure=cleaner.groupby("Adventure")["revenue"].aggregate('mean')[1]
Animation=cleaner.groupby("Animation")["revenue"].aggregate('mean')[1]
Crime=cleaner.groupby("Crime")["revenue"].aggregate('mean')[1]
Documentary=cleaner.groupby("Documentary")["revenue"].aggregate('mean')[1]
Drama=cleaner.groupby("Drama")["revenue"].aggregate('mean')[1]
Family=cleaner.groupby("Family")["revenue"].aggregate('mean')[1]
Fantasy=cleaner.groupby("Fantasy")["revenue"].aggregate('mean')[1]
Foreign=cleaner.groupby("Foreign")["revenue"].aggregate('mean')[1]
History=cleaner.groupby("History")["revenue"].aggregate('mean')[1]
Horror=cleaner.groupby("Horror")["revenue"].aggregate('mean')[1]
Music=cleaner.groupby("Music")["revenue"].aggregate('mean')[1]
Mystery=cleaner.groupby("Mystery")["revenue"].aggregate('mean')[1]
Romance=cleaner.groupby("Romance")["revenue"].aggregate('mean')[1]
ScienceFiction=cleaner.groupby("Science Fiction")["revenue"].aggregate('mean')[1]
Thriller=cleaner.groupby("Thriller")["revenue"].aggregate('mean')[1]
War=cleaner.groupby("War")["revenue"].aggregate('mean')[1]
Western= cleaner.groupby("Western")["revenue"].aggregate('mean')[1]

x= 0
cleaner["AverageRevenue"] = 0
while(len(cleaner)>x):
    avg = 0
    count = 0
    if(cleaner['Action'][x] == 1):
        count = count + 1
        avg = Action
    if(cleaner['Adventure'][x] == 1):
        count = count + 1
        avg = avg + Adventure
    if(cleaner['Animation'][x] == 1):
        count = count + 1
        avg = avg + Animation
    if(cleaner['Crime'][x] == 1):
        count = count + 1
        avg = avg + Crime
    if(cleaner['Documentary'][x] == 1):
        count = count + 1
        avg = avg + Documentary
    if(cleaner['Drama'][x] == 1):
        count = count + 1
        avg = avg + Drama
    if(cleaner['Family'][x] == 1):
        count = count + 1
        avg = avg + Family
    if(cleaner['Fantasy'][x] == 1):
        count = count + 1
        avg = avg + Fantasy
    if(cleaner['Foreign'][x] == 1):
        count = count + 1
        avg = avg + Foreign
    if(cleaner['History'][x] == 1):
        count = count + 1
        avg = avg + History
    if(cleaner['Horror'][x] == 1):
        count = count + 1
        avg = avg + Horror
    if(cleaner['Music'][x] == 1):
        count = count + 1
        avg = avg + Music
    if(cleaner['Mystery'][x] == 1):
        count = count + 1
        avg = avg + Mystery
    if(cleaner['Romance'][x] == 1):
        count = count + 1
        avg = avg + Romance
    if(cleaner['Science Fiction'][x] == 1):
        count = count + 1
        avg = avg + ScienceFiction
    if(cleaner['Thriller'][x] == 1):
        count = count + 1
        avg = avg + Thriller
            
    if(cleaner['War'][x] == 1):
        count = count + 1
        avg = avg + War
            
    if(cleaner['Western'][x] == 1):
        count = count + 1
        avg = avg + Western
    if(count>0):
        
        cleaner["AverageRevenue"][x] = avg / count
    x=x+1
            
                

    


# In[29]:




cleaner.hist(column='budget', bins=10,normed=True)


# In[38]:


#convert all object to cat type needed to check skew

cleaner = pd.concat([
        cleaner.select_dtypes([], ['object']),
        cleaner.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ], axis=1).reindex_axis(cleaner.columns, axis=1)


# In[39]:


# check the skew of all the values
from scipy.stats import norm, skew #for some statistics
numeric_feats = cleaner.dtypes[cleaner.dtypes != "category"].index


skewed_feats = cleaner[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(300)



# In[40]:


# Statistics
from scipy import stats
from scipy.stats import norm, skew
from statistics import mode
from scipy.special import boxcox1p

skewed_features = skewness.index
for feat in skewed_features:
    #only fix these since the rest are one hot encoding
    if(feat == "popularity" or feat == "budget" or feat == "runtime" or feat == "AverageRevenue" or feat == "NumReleaseCountry" or feat == "rating" or feat == "votes"): 
        
        
        #need to do +1 becasue of 0 values.
        #cleaner[feat] = np.log(cleaner[feat]+1)
        #cant use boxcox becasue neg number 
        #cleaner[feat] = stats.boxcox(cleaner[feat])
        cleaner[feat] = np.log(cleaner[feat]+1)



#skewed_feats = cleaner[numeric_feats].apply(lambda x: np.log(x.dropna())).sort_values(ascending=False)
skewed_feats = cleaner[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=True)

print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(300)








# In[25]:


#split the data
train1 = cleaner[cleaner["train"]==1]
test = cleaner[cleaner["train"]==0]
test=test.drop(['train'], axis=1)
train1=train1.drop(['train'], axis=1)
#finalTrain= pd.concat([train1, train['revenue']], axis=1)

#finalTrain[revenue] = np.log(finalTrain[revenue])


# In[24]:


#Fix scew for y
#y = np.log(cleaner["revenue"])
 
#test=test.drop(['revenue'], axis=1)
#test=test.drop(['train'], axis=1)
#train1=train1.drop(['train'], axis=1)
#train1=train1.drop(['revenue'], axis=1)

#test1=test.drop(['id'], axis=1)


# In[25]:


#skewed_feats = y.apply(lambda x: skew(x.dropna())).sort_values(ascending=True)
#skewness = pd.DataFrame({'Skew' :skewed_feats})
#y = np.log(cleaner["revenue"])

#train1=train1.drop(['revenue'], axis=1)


# In[41]:


# Create a heatmap correlation to find relevant variables
corr = train1.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()


# In[44]:


# Select columns with a correlation > 0.5
rel_vars = corr.revenue[(corr.revenue > 0.2)]
rel_cols = list(rel_vars.index.values)

corr2 = train1[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()

corr2


# In[55]:


corr2.columns.tolist()


# In[55]:


finalTrain.hist(column='revenue', bins=50,normed=True)


# # Data models
# Created a few different models. For a baseline value, I just created two different random forest models. The main model was an Ensemble learning the combined a few weak learners.  I also tried a neural network, but that didnt perform well and didnt spend to much time since the best kernals didnt use it.

# In[14]:


#y = np.array(train['revenue'])
#train=train.drop(['revenue'], axis=1)
x = np.array(train)
#x = np.array(train[['OverallQual',"['KitchenQual']_Ex","['BsmtQual']_Ex", "GarageArea","GarageCars","TotRmsAbvGrd","FullBath","GrLivArea","1stFlrSF","TotalBsmtSF","YearRemodAdd","YearBuilt","OverallQual"]])   
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state = 0)
#x = np.array(train[['budget']])   



#Split data 80% training 10% test.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=2)

#Transform the scale of the x data sets.
stdscaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = stdscaler.transform(X_train)
X_test_scaled  = stdscaler.transform(X_test)



# In[56]:


def root_mean_squared_error(y_true, y_pred):
        n = len(y_true)

        return np.sqrt( 1/n*np.sum((y_pred-y_true)**2) )
    


# In[57]:



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y))**2))


# In[25]:


#Base Model: random forest-.




scorer = make_scorer(r2_score)

n_estimators_list = [10,100,200,400,500]
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
rfc = RandomForestRegressor(random_state=1,oob_score = True,n_jobs = 1)
grid = GridSearchCV(estimator=rfc, param_grid=dict(n_estimators=n_estimators_list,min_samples_leaf = [1,2,3,5,10] ),scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train_scaled, y_train)
rf_opt = grid_fit.best_estimator_

print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.max_depth)

#pickle.dump(grid, open("RandomForestModel", 'wb'))

predicted = rf_opt.predict(X_test_scaled)
#print(predicted)
#predicted = grid.predict(z)
#print("Decision Tree Results\nConfusion Matrix:")
#print (confusion_matrix(y_test,predicted))
#print ("\n Classifcation Report")
#print (classification_report(y_test,predicted))
importances = grid.best_estimator_.feature_importances_

indices = np.argsort(importances)[::-1]
print("Feature Ranking", indices)
#print "Accuracy is ", accuracy_score(y_test,predicted)*100
#predicted
# show the inputs and predicted outputs
#predictedValue = [round(x[0]) for x in predicted]
#print(predictedValue)

#for i in range(len(X_train)):
#	print("X=%s, Predicted=%s" % (X_train[i], predicted[i]))

#ids = test['PassengerId']
#predicted1 = grid.predict(xt)
# Use the forest's predict method on the test data
rf_r2 = r2_score(y_test, predicted)
rf_mse = root_mean_squared_error(y_test, predicted)
print('rf_mse')
print(rf_mse)
grid.score(X_train, y_train)
rf_opt_preds = rf_opt.predict(X_test) # RF predictions


# In[15]:






# In[27]:


#The prediction using first model
xx = np.array(test1)   


rf_opt_preds = rf_opt.predict(xx) # RF predictions
y_pred_final = np.exp(rf_opt_preds)

print(rf_opt_preds.shape)
# Final submission
my_submission1 = pd.DataFrame({'id': test.id, 'revenue': y_pred_final})
my_submission1.to_csv('submissionOne.csv', index=False)


# In[28]:


#Random forerst version 2 using shufflesplit instead of GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV


rf_regressor = RandomForestRegressor(random_state=42)
cv_sets = ShuffleSplit(random_state = 4) # shuffling our data for cross-validation
parameters = {'n_estimators':[10,100,200,500,1000], 
              'min_samples_leaf':[1, 2, 3,5,10], 
              'max_depth':[5,10,15,20,50]}
scorer = make_scorer(r2_score)
n_iter_search = 10
grid_obj = RandomizedSearchCV(rf_regressor, 
                              parameters, 
                              n_iter = n_iter_search, 
                              scoring = scorer, 
                              cv = cv_sets,
                              random_state= 99)
grid_fit = grid_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_


print(grid_obj)
# summarize the results of the grid search
print(grid_obj.best_score_)
print(grid_obj.best_estimator_.max_depth)

importances = grid_obj.best_estimator_.feature_importances_

indices = np.argsort(importances)[::-1]
print("Feature Ranking", indices)

rf_r2 = r2_score(y_test, predicted)
rf_mse = root_mean_squared_error(y_test, predicted)
print('rf_mse')
print(rf_mse)

rf_opt_preds = rf_opt.predict(X_test) # RF predictions


# In[29]:


#The prediction using second model
xx = np.array(test1)   


rf_opt_preds = rf_opt.predict(xx) # RF predictions
y_pred_final = np.exp(rf_opt_preds)

print(rf_opt_preds.shape)
# Final submission
my_submission1 = pd.DataFrame({'id': test.id, 'revenue': y_pred_final})


# In[97]:


#stacking models



# Set up variables
X_train = train1
X_test = test.drop(['id'], axis=1)
X_train = X_train.drop(['id'], axis=1)

y_train = train1['revenue']

y_train = np.log(y_train)

X_train=X_train.drop(['revenue'], axis=1)

#This will ensure that all rmse scores produced have 
#been smoothed out across the entire dataset and are 
#not a result of any irregularities, which otherwise
#would provide a misleading representation of model
#performance. 

#Define a evaluation matrix 
from sklearn.metrics.scorer import make_scorer

RMSLE = make_scorer(rmsle)



# Defining two rmse_cv functions
#CV=10 is 10 folds
def rmse_cv(model):
    rmse = np.sqrt(cross_val_score(model, X_train, y_train, scoring=RMSLE, cv = 10))

   

    #rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=rmsle, cv = 10))
    return(rmse)


# In[105]:


corr3 =corr2.drop(['revenue'], axis=1)[corr2.columns.tolist()]


# In[109]:


#1. Ridge Regression (L2 Regularisation)

#Ridge regression shrinks the regression coefficients, so that variables, 
#with minor contribution to the outcome, have their coefficients close to zero.


#For this, we are trying to find the best alpha value to use.
#Alpha is a regularization parameter that measures how flexible our model is.
#The higher the regularization the less prone our model will be to overfit. 

#the optimal value will have the lowest RMSE on the graph.

# Setting up list of alpha's

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 28,29,30,31,32,33]

# Iterate over alpha's
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
# Plot findings
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# In[110]:


#2. Lasso Regression (L1 regularisation)
#It shrinks the regression coefficients toward zero by penalizing the regression model
#with a penalty term called L1-norm, which is the sum of the absolute coefficients.

#In the case of lasso regression, the penalty has the effect of forcing some of the 
#coefficient estimates, with a minor contribution to the model, to be exactly equal to zero.
#This can be also seen as an alternative to the subset selection methods for performing
#variable selection in order to reduce the complexity of the model.

#find the best alpha

# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# Iterate over alpha's
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# In[111]:


#3. ElasticNet Regression
#Elastic Net produces a regression model that is penalized with both the L1-norm and L2-norm. 
#The consequence of this is to effectively shrink coefficients (like in ridge regression) and
#to set some coefficients to zero (as in LASSO).

# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# Iterate over alpha's
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_elastic = pd.Series(cv_elastic, index = alphas)
cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# In[28]:


# 4. Kernel ridge regression

#Kernel ridge regression (KRR) combines Ridge Regression 
#(linear least squares with l2-norm regularization) with the 'kernel trick'.

# Setting up list of alpha's
alphas = [30,25,20,15,10]

# Iterate over alpha's
cv_krr = [rmse_cv(KernelRidge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_krr = pd.Series(cv_krr, index = alphas)
cv_krr.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# # Model initazing

# In[78]:


#Differnet models that are initiazing 
#1. Ridge Regression
model_ridge = Ridge(alpha = 29)
#lasso  Pipeline to scale features
model_lasso  = make_pipeline(RobustScaler(), Lasso(alpha =0.005, random_state=1))
#Elastic net 
model_elastic  = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
#Kernel Ridge Regression :

#Kernel: Polynomial-his means that the algorithm will not just consider 
#similarity between features, but also similarity between combinations of features.
#Degree & Coef0: These are used to define the precise structure of the Polynomial kernel.  
    
model_krr  =make_pipeline(RobustScaler(), KernelRidge(alpha=10, kernel='polynomial', degree=2.65, coef0=6.9))


# In[74]:


#Ensemble methods-
#Boosting is an ensemble technique in which the predictors are
#not made independently, but sequentially.
#It is used to for reducing bias and variance in supervised learning.
#It combines multiple weak predictors to a build strong predictor.
# But we have to choose the stopping criteria carefully or it could lead to overfitting.

'''Random forest is a bagging technique and not a boosting technique.
In boosting as the name suggests, one is learning from other which in turn boosts the learning.
The trees in random forests are run in parallel. There is no interaction between these trees
while building the trees.
'''

model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=21,
                              loss_function='RMSE',
                              verbose=50)

# Initiating parameters ready for CatBoost's CV function, which I will use below
params = {'iterations':1350,
          'learning_rate':0.05,
          'depth':3,
          'l2_leaf_reg':4,
          'border_count':21,
          'loss_function':'RMSE',
          'verbose':200}





model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,
                              learning_rate=0.01, n_estimators=1475,
                              max_bin = 65, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.3, 
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=900,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.9, silent=1,
                             random_state =7)
model_gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.025,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=2,subsample= 0.8,  min_samples_split=7, 
                                   loss='huber', random_state =5)



# # Model tuning 

# In[19]:




#CatBoost is a new gradient boosting algorithm able to work with categorical
#features without any prior processing needed.

'''
iterations-num of trees
l2_leaf_reg-Coefficient at the L2 regularization term of the cost function.
border_count-The number of splits for numerical features.
loss_function-For 2-class classification use 'LogLoss' or 'CrossEntropy'. For multiclass use 'MultiClass'.
ctr_border_count-The number of splits for categorical features.

model_cat = CatBoostRegressor(iterations=2000,
                              learning_rate=0.10,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=15,
                              loss_function='RMSE',
                              verbose=200)

'''

model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=21,
                              loss_function='RMSE',
                              verbose=50)


# In[92]:



#find optimal estimator and learning rate using cv
#W


scorer = make_scorer(r2_score)


model_cat = CatBoostRegressor(iterations=2000,
                              learning_rate=0.10,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=15,
                              loss_function='RMSE',
                              verbose=200)

p_test3 = {'learning_rate':[0.1,0.05], 'iterations':[2000,1350]}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[ ]:



#find optimal depth and border_count using cv
#W


scorer = make_scorer(r2_score)


model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=15,
                              loss_function='RMSE',
                              verbose=200)

p_test3 = {'border_count':range(5, 25, 2), 'depth':range(1, 10, 1)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[26]:



#find optimal depth and border_count using cv
#W


scorer = make_scorer(r2_score)


model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=21,
                              loss_function='RMSE',
                              verbose=200)

p_test3 = {'verbose':range(50, 400, 25)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[27]:



#find optimal l2_leaf_regusing cv
#W


scorer = make_scorer(r2_score)


model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              
                              border_count=21,
                              loss_function='RMSE',
                              verbose=50)

p_test3 = {'l2_leaf_reg':range(0, 10, 1)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[16]:



#LightGBM :
'''
feature_fraction-% of features used in a tree

bagging_fraction-% of data to be used in a tree
bagging_freq- similar to fraction,enables bagging (subsampling) of the training data. 
both values need to be set for bagging to be used. The frequency controls how often (iteration) bagging is used. 
Smaller fractions and frequencies reduce overfitting.

num_leaves - like max depth

early_stopping_round-Model will stop training if one metric of one validation data doesn’t improve in last 

min_gain_to_split-minimum gain to make a split, finds the split points on the group boundaries, default:64


max_cat_group-for large num of category-

boosting_type-defines the type of algorithm you want to run, default=gdbt

max_bin- it denotes the maximum number of bin that feature value will bucket in.

min_sum_hessian_in_leaf- is the minimum sum of hessian in a leaf to keep splitting
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
'''

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,
                              learning_rate=0.01, n_estimators=1475,
                              max_bin = 65, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)


# In[17]:



#find optimal estimator and learning rate using cv
#W


scorer = make_scorer(r2_score)


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

p_test3 = {'learning_rate':[0.5,0.1,0.05,0.01,0.005,0.001], 'n_estimators':range(100, 1500, 25)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[19]:



#max_bin ,num_leaves tune
#W


scorer = make_scorer(r2_score)


model_lgb  = lgb.LGBMRegressor(objective='regression',
                              learning_rate=0.01, n_estimators=1475,
                               bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

p_test3 = {'max_bin':range(10, 70, 5), 'num_leaves':range(2, 10, 2)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[19]:



#bagging_freq ,bagging_fraction tune
#W


scorer = make_scorer(r2_score)


model_lgb =  lgb.LGBMRegressor(objective='regression',num_leaves=8,
                              learning_rate=0.01, n_estimators=1475,
                              max_bin = 65, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

p_test3 = {'bagging_freq':range(1, 8, 1), 'bagging_fraction':[0.1,0.2,0.3,.4,.5,.6,.7,.8,.9]}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[21]:



#min_data_in_leaf ,feature_fraction tune
#W


scorer = make_scorer(r2_score)


model_lgb =   lgb.LGBMRegressor(objective='regression',num_leaves=8,
                              learning_rate=0.01, n_estimators=1475,
                              max_bin = 65, bagging_fraction = 0.9,
                              bagging_freq = 4, 
                              feature_fraction_seed=9, bagging_seed=9,
                               min_sum_hessian_in_leaf = 11)

p_test3 = {'min_data_in_leaf':range(1, 5, 1), 'feature_fraction':[0.25,0.2,0.3,.2319]}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[22]:



#min_data_in_leaf ,feature_fraction tune
#W


scorer = make_scorer(r2_score)


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=8,
                              learning_rate=0.01, n_estimators=1475,
                              max_bin = 65, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, )

p_test3 = {'min_sum_hessian_in_leaf':range(1, 15, 1)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[15]:



#XGBoost 
'''
silent-no message displayed whenb running

learning_rate-learning rate, typical value of 0.01-.2 default 0.3

min_child_weight-sum of weights similar to min chidl leaf in gbm- controls overfitting, lower the 
value more chance of overfitting

max_depth - depth of tree

max_leaf_nodes- man number of  terminal nodes or leaves in a tree. If this is defined, GBM will ignore max_depth.

gamma -A node is split only when the resulting split gives a positive reduction in the loss function. 
Gamma specifies the minimum loss reduction required to make a split.

max_delta_step-In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0,
it means there is no constraint. If it is set to a positive value,
it can help making the update step more conservative.
Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.

colsample_bytree-Denotes the fraction of columns to be randomly samples for each tree.

reg_lambda -L2 regularization term on weights 

reg_alpha - L1 regularization term on weight good for high dimensionality.

scale_pos_weight -A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.

objective-This defines the loss function to be minimized.default linear

eval_metric -The metric to be used for validation data.

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7)
'''

model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.3, 
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=900,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.9, silent=1,
                             random_state =7)


scorer = make_scorer(r2_score)


# In[16]:



#find optimal estimator and learning rate using cv
#W


scorer = make_scorer(r2_score)


model_xgb = xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.3, 
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=926,
                            
                             subsample=0.9, silent=1,
                             random_state =7)

p_test3 = {'learning_rate':[0.025,0.01,0.0075,.005,.0025,.0005], 'n_estimators':range(900, 2000, 25)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[19]:


#Max depth min child weight tune
model_xgb = xgb.XGBRegressor(colsample_bytree=0.8, gamma=0, 
                             learning_rate=0.01, 
                             n_estimators=926,
                             
                             subsample=0.8, silent=1,
                             random_state =7)

p_test3 = {'max_depth':range(3,10,1), 'min_child_weight':range(1,6,1)}


#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[21]:


#gamma

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,  
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=926,
                           
                             subsample=0.5213, silent=1,
                             random_state =7)

p_test3 = {'gamma':[i/10.0 for i in range(0,5)]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[23]:


#Step 4: Tune subsample and colsample_bytree
    
    

model_xgb = xgb.XGBRegressor( gamma=0.3, 
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=926,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                              silent=1,
                             random_state =7)

p_test3 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[25]:


#Tuning Regularization Parameters
    
    
xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.3, 
                             learning_rate=0.01, max_depth=6, 
                             min_child_weight=5, n_estimators=926,
                        
                             subsample=0.9, silent=1,
                             random_state =7)

p_test3 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[29]:


'''
Gradient Boosting Regression :


Tree-Specific Parameters: These affect each individual tree in the model.

#XGBoost 
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
min_samples_split-minimum number of samples (or observations) which are required in a node to be
considered for splitting. Tuned using CV

min_samples_leaf-Defines the minimum samples (or observations) required in a terminal node or leaf.Pick low value for 
imbalanced class problems about 0.5% total sample

max_depth-The maximum depth of a tree. Tuned using CV

max_leaf_nodes -The maximum number of terminal nodes or leaves in a tree. If this is defined, GBM will ignore max_depth.

max_features-The number of features to consider while searching for a best split. These will be randomly selected.
As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the
total number of features.
Higher values can lead to over-fitting but depends on case to case.


managing boosting:

learning_rate:This determines the impact of each tree on the final outcome.
starting with an initial estimate which is updated using the output of each tree
The learning parameter controls the magnitude of this change in the estimates.
The lower the value the better the results but at a cost of performance.

n_estimators:Number of trees to be made. Tune using CV, to high of a value casuses overfit

subsample- % observation used for each tree, random sampling.
Typical values ~0.8 generally work fine but can be fine-tuned further.

loss-loss function to be minimized in each split.
huber-loss function as this is robust to outliers.

init-affects initialization of the output.This can be used if we have made another model whose outcome is to be used as the initial estimates for GBM.

random_state-The random number seed so that same random numbers are generated every time.


After each pass decrease learning rate and increase estimator and run all the cv again

start at 0.5 and 51 now 0.25 102

GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

'''

#
model_gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.025,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=2,subsample= 0.8,  min_samples_split=7, 
                                   loss='huber', random_state =5)


scorer = make_scorer(r2_score)



# In[50]:



#find optimal estimator using cv
#We found a value of 51 with 0.5 lr

#if n_estimators'> 100 try higher rate since tuning others will take longer, if less then 20, lower the rate.

#loss function 
#R2- tells you the x% of the variablity in rev can be explained by the variablity of the variables. So
#so the model can only explain X% of the diff  of what the movie makes the certain amount.
scorer = make_scorer(r2_score)


model_gbr = GradientBoostingRegressor(
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=15,subsample= 0.8,  min_samples_split=52, 
                                   loss='huber', random_state =5)

p_test3 = {'learning_rate':[0.5, 0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
n_estimators_list =  range(1, 1000, 25)
#n_estimators_list = [10,100,200,400,500]

#n_estimators_list = {'n_estimators':range(10,1000,50)}
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
#n_jobs = -1 use all cpu
grid = GridSearchCV(estimator=model_gbr, param_grid=p_test3,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[22]:


#Tune the max_depth
#resulted  depth of 9 with 52
model_gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.025,
                                   max_features='sqrt',
                                   subsample= 0.8, 
                                   loss='huber', random_state =5)
param_test2 = {'max_depth':range(3,16,1), 'min_samples_split':range(2,100,5)}


grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[23]:


#Tune the min_samples_leaf
#

model_gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.025,
                                   max_depth=5, max_features='sqrt',
                                   subsample= 0.8, 
                                   loss='huber', random_state =5)
param_test2 = {'min_samples_leaf':range(2,70,5), 'min_samples_split':range(2,100,5)}


grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[53]:


#max_features 
model_gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=7,subsample= 0.8,  min_samples_split=22, 
                                   loss='huber', random_state =5)
param_test2 = {'max_features':range(5,30,2)}


grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[54]:


#Tune the subsample
#

model_gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=7,  min_samples_split=22, 
                                   loss='huber', random_state =5)
param_test2 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# # Model initializing

# In[79]:


model_gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)



#XGBoost 
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


#LightGBM :
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
#catboost
#CatBoost is a new gradient boosting algorithm able to work with categorical
#features without any prior processing needed.
model_cat = CatBoostRegressor(iterations=2000,
                              learning_rate=0.10,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=15,
                              loss_function=RMSLE,
                              #loss_function='RMSE'
                              verbose=200)

# Initiating parameters ready for CatBoost's CV function, which I will use below
params = {'iterations':2000,
          'learning_rate':0.10,
          'depth':3,
          'l2_leaf_reg':4,
          'border_count':15,
          'loss_function':'RMSE',
          'verbose':200}


# # Model training

# In[98]:


#run the custom rmse_cv function on each algorithm to understand each model's performance.


# Fitting all models with rmse_cv function, apart from CatBoost
cv_ridge = rmse_cv(model_ridge).mean()
cv_lasso = rmse_cv(model_lasso).mean()
cv_elastic = rmse_cv(model_elastic).mean()
cv_krr = rmse_cv(model_krr).mean()
cv_gbr = rmse_cv(model_gbr).mean()
cv_xgb = rmse_cv(model_xgb).mean()
cv_lgb = rmse_cv(model_lgb).mean()

#cant use above for catboost so,

# Define pool
pool = Pool(X_train, y_train)

# CV Catboost algorithm
cv_cat = cv(pool=pool, params=params, fold_count=10, shuffle=True)

# Select best model
cv_cat = cv_cat.at[1999, 'train-RMSE-mean']


# In[99]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Ridge',
              'Lasso',
              'ElasticNet',
              'Kernel Ridge',
              'Gradient Boosting Regressor',
              'XGBoost Regressor',
              'Light Gradient Boosting Regressor',
              'CatBoost'],
    'Score': [cv_ridge,
              cv_lasso,
              cv_elastic,
              cv_krr,
              cv_gbr,
              cv_xgb,
              cv_lgb,
              cv_cat]})

# Build dataframe of values
result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
result_df.head(8)


# In[ ]:


0 	XGBoost Regressor 	0.455752
1 	Light Gradient Boosting Regressor 	0.457496
2 	Gradient Boosting Regressor 	0.462365
3 	Lasso 	0.462964
4 	ElasticNet 	0.462970
5 	Ridge 	0.463148
6 	Kernel Ridge 	0.463732
7 	CatBoost 	1.576397


# # Prediction of the test data

# In[106]:


# Fit and predict all models
model_lasso.fit(X_train, y_train)
lasso_pred = np.expm1(model_lasso.predict(X_test))

model_elastic.fit(X_train, y_train)
elastic_pred = np.expm1(model_elastic.predict(X_test))

model_ridge.fit(X_train, y_train)
ridge_pred = np.expm1(model_ridge.predict(X_test))

model_xgb.fit(X_train, y_train)
xgb_pred = np.expm1(model_xgb.predict(X_test))

model_gbr.fit(X_train, y_train)
gbr_pred = np.expm1(model_gbr.predict(X_test))

model_lgb.fit(X_train, y_train)
lgb_pred = np.expm1(model_lgb.predict(X_test))

model_krr.fit(X_train, y_train)
krr_pred = np.expm1(model_krr.predict(X_test))

model_cat.fit(X_train, y_train)
cat_pred = np.expm1(model_cat.predict(X_test))


# In[107]:


#I'm going to keep this very simple by equally weighting every model. This is done by
#summing together the models and then dividing by the total count. Weighted averages
#could be a means of gaining a slightly better final predictions, whereby the best performing
#models take a bigger cut of the stacked model.



# Create stacked model
#stacked = (cat_pred) 

stacked = (lasso_pred + elastic_pred + ridge_pred + xgb_pred + lgb_pred + krr_pred + gbr_pred) / 7
#since revenue was transformed with log need to reverse it
#stacked = np.exp(stacked) 


# In[108]:


my_submission1 = pd.DataFrame({'id': test.id, 'revenue': stacked})
my_submission1.to_csv('submission-070418.csv', index=False)


# # Neural Network
# Didnt spend much time on this model since it performed poorly

# In[ ]:


#3rd model nn
#Implementing a neural network.
model = Sequential()

#Best results I got was with two hidden laywers with first one being relu and second being sigmoid.
model.add(Dense(output_dim=100, input_shape=[X_train_scaled.shape[1]], 
                activation='relu',W_regularizer=l2(.01)))
model.add(Dense(output_dim=1, activation='sigmoid',W_regularizer=l2(.01)))
model.compile(optimizer="rmsprop",
              loss='binary_crossentropy',     
              metrics=['accuracy'])


#optimizer=keras.optimizers.Adadelta()
#model.compile(optimizer=keras.optimizers.Adadelta(),
#              loss=root_mean_squared_error,     
#              metrics=[root_mean_squared_error])




# save the model to disk
pickle.dump(model, open("NNModel", 'wb'))

history = model.fit(X_train_scaled, y_train, batch_size = 64,
          nb_epoch =250, verbose=1, validation_data=(X_test_scaled,y_test))
score = model.evaluate(X_test_scaled, y_test, verbose=0) 




#Plot
fig = plt.figure(figsize=(6,4))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], 'g--')
plt.title('Neural Network Model Loss')
plt.ylabel('Binary Crossentropy')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Testing Loss'], loc='upper right')
print ("Loss and accuracy after final iteration: ", model.evaluate(X_test, y_test, batch_size=64))
plt.show()



print('Test score:', score[0]) 
print('Test accuracy:', score[1])


# In[ ]:


# Predict new result
y_pred = grid.predict(X_test)

# Plot y_test vs y_pred
plt.figure(figsize=(12,8))
plt.plot(y_test, color='red')
plt.plot(y_pred, color='blue')
plt.show()


# In[ ]:


#Junk code
xx = np.array(test[['1stFlrSF',"FullBath","GarageArea", "GarageCars","GrLivArea","OverallQual","TotRmsAbvGrd","TotalBsmtSF","1stFlrSF","TotalBsmtSF","YearBuilt","YearRemodAdd","['ExterQual']_Gd","['Foundation']_PConc"]])   


rf_opt_preds = rf_opt.predict(xx) # RF predictions
y_pred_final = np.exp(rf_opt_preds)

print(rf_opt_preds.shape)
# Final submission
my_submission1 = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred_final})
my_submission1.to_csv('submission-070418.csv', index=False)

