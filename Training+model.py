
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras.regularizers import l2, l1
from keras.optimizers import SGD
from sklearn import cross_validation,preprocessing,tree
from scipy.io import loadmat
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pydotplus 
from sklearn.externals.six import StringIO
from IPython.display import Image
from keras import optimizers
from keras.layers import Dropout
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.layers.recurrent import LSTM
import matplotlib.ticker as mtick

ticket = pd.read_csv('ticketDatafinal.csv', sep="\t", header=0)



#New dataset with just avg price and days
ranks1 = pd.read_csv('weekend1.csv', sep=",", header=1)
ranks2 = pd.read_csv('weekend2.csv', sep=",", header=1)
#tic = data.ix[data.weekend != 0]
group1 = ticket.ix[ticket.sold != 0]
#group2 = ticket.ix[ticket.sold != 1]
#group1 = group1.ix[group1.VIP != 2]
group2 = group1.ix[group1.VIP != 1]

group1s = group2.ix[group2.weekend != 2]
#group11 = group2.ix[group2.weekend != 2]


groups1sss = group1s.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')
#groups1ss = group1s.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

#groups1ss = group11.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

#groups2 = group2.groupby(['days till','weekend']).ticketPrice.mean().reset_index(name='avg unsold price')






# In[2]:

#First model, neural network to predict if the ticket will sell.
y = np.array(ticket['sold'])
x = np.array(ticket[["# of ticket", "CAR CAMPING PASS", "VIP", "VIP Parking", "weekend","days till","PriceRange",
                      "ticketPrice","trend","Price","Shuttle Passes","avg unsold price","avg sold price"  ]])
#Split data 80% training 10% test.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=2)

#Transform the scale of the x data sets.
stdscaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = stdscaler.transform(X_train)
X_test_scaled  = stdscaler.transform(X_test)

#Implementing a neural network.
model = Sequential()

#Best results I got was with two hidden laywers with first one being relu and second being sigmoid.
model.add(Dense(output_dim=100, input_shape=[X_train_scaled.shape[1]], 
                activation='relu',W_regularizer=l2(.01)))
model.add(Dense(output_dim=1, activation='sigmoid',W_regularizer=l2(.01)))
model.compile(optimizer="rmsprop",
              loss='binary_crossentropy',     
              metrics=['accuracy'])


# save the model to disk
pickle.dump(model, open("NNModel", 'wb'))

history = model.fit(X_train_scaled, y_train, batch_size = 64,
          nb_epoch =250, verbose=1, validation_data=(X_test_scaled,y_test))


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

predicted = model.predict(X_test_scaled)
#predicted = model.predict(z)
#print ('confusion matrix\n', confusion_matrix(predicted.astype(int),y_test.astype(int)))
print (classification_report(predicted.astype(int),y_test.astype(int)))

predictedValue = [round(x[0]) for x in predicted]
print(predictedValue)




# In[22]:

#Model two: random forest- predict if a ticket will sell or not.
n_estimators_list = [10,100,500,1000]
#max_features none so use all the features n_estimators    min_samples_leaf : int, float, optional (default=1)  
z= np.array([[1,0,0,0,2,2,40,1000,41,1075,1,532.299,488.427]])
#n_jobs = -1 use all cpu
rfc = RandomForestClassifier(random_state=1,oob_score = True,n_jobs = 1)
grid = GridSearchCV(estimator=rfc, param_grid=dict(n_estimators=n_estimators_list,min_samples_leaf = [1,2,3,5,10] ))
#max_depths = range(1,30,1)

#dtc = tree.DecisionTreeClassifier(criterion='entropy')
#test each depth from 1 to 100 to see which depth performance the best for the data.rfc
#grid = GridSearchCV(estimator=dtc, param_grid=dict(max_depth=max_depths))
grid.fit(X_train, y_train)

print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.max_depth)

pickle.dump(grid, open("RandomForestModel", 'wb'))


predicted = grid.predict(X_test)
#predicted = grid.predict(z)
print("Decision Tree Results\nConfusion Matrix:")
#print (confusion_matrix(y_test,predicted))
print ("\n Classifcation Report")
#print (classification_report(y_test,predicted))
importances = grid.best_estimator_.feature_importances_

indices = np.argsort(importances)[::-1]
print("Feature Ranking", indices)
#print "Accuracy is ", accuracy_score(y_test,predicted)*100
predicted
# show the inputs and predicted outputs
#predictedValue = [round(x[0]) for x in predicted]
#print(predictedValue)

#for i in range(len(X_train)):
#	print("X=%s, Predicted=%s" % (X_train[i], predicted[i]))


# In[8]:

#Third model, LSTM - predict future price of the ticket.
#Create a new column with shifted price range, this will be what we are trying to predict
data = pd.concat ([groups1sss, groups1sss["avg sold price"].shift (-1)], axis =1)
data.columns = ['Days till', 'avg sold price','avg sold price-1']
     
data = data.dropna()
y = data['avg sold price-1']     
x = data[["avg sold price" , "Days till"]]
     
    
scaler_x = preprocessing.MinMaxScaler ( feature_range =( -1, 1))
x = np. array (x).reshape ((len( x) ,2))
x = scaler_x.fit_transform (x)
   
scaler_y = preprocessing. MinMaxScaler ( feature_range =( -1, 1))
y = np.array (y).reshape ((len( y), 1))
y = scaler_y.fit_transform (y)

x_train = x [0: 61,]
x_test = x[ 62:len(x),]    
y_train = y [0: 61] 
y_test = y[ 62:len(y)] 

x_train = x_train.reshape (x_train.shape + (1,)) 
x_test = x_test.reshape (x_test.shape + (1,))
 
     
model = Sequential ()
#LSTM model with 1000 neurons
model.add (LSTM (1000 , activation = 'tanh',  input_shape =(2, 1),return_sequences=False ))
model.add(Dropout(0.2))
#model.add (LSTM (1000 , activation = 'tanh' ,return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(500,activation='relu'))
#model.add(Dropout(0.2))
model.add (Dense (output_dim =1, activation = 'linear'))

model.compile (loss ="mean_squared_error" , optimizer = "adam")   


history = model.fit(x_train, y_train, epochs=50, batch_size =32, validation_data=(x_test, y_test), shuffle=False)
# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

 
#pickle.dump(model, open("LSTMModel", 'wb'))

score_train = model.evaluate (x_train, y_train, batch_size =1)
score_test = model.evaluate (x_test, y_test, batch_size =1)
print ("train MSE = ", round( score_train ,4)) 
print ("test MSE = ", score_test )
 
    
pred1 = model.predict (x_test) 
pred1 = scaler_y.inverse_transform (np. array (pred1). reshape ((len( pred1), 1)))
      
prediction_data = pred1[-1]     
    
model.summary()

   
print ("prediction data:")
print (prediction_data)
 
print ("actual data")
x_test = scaler_x.inverse_transform (np. array (x_test). reshape ((len( x_test),2)))
print(x_test) 
plt.plot(pred1, label="predictions") 
y_test = scaler_y.inverse_transform (np. array (y_test). reshape ((len( y_test), 1)))
plt.plot( [row[0] for row in y_test], label="actual")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2)
 
fmt = '$%.0f'
tick = mtick.FormatStrFormatter(fmt)
 
ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
 
plt.show()



# In[22]:

#daily avg price for tickets
group1 = ticket.ix[ticket.sold != 0]
wk1 = group1.ix[group1.VIP != 1]
wk1 = wk1.ix[wk1.weekend != 2]
wk1 = wk1.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


wk2 = group1.ix[group1.VIP != 1]
wk2 = wk2.ix[wk2.weekend != 1]
wk2 = wk2.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


plt.title('avg daily sold price WK 1 vs wk2')
plt.ylabel('Average Price')
plt.xlabel('Days till the Event')

plt.plot(priceTrend['days till'],priceTrend['avg sold price'],label="Weekend 1")

plt.plot(wk2['days till'],wk2['avg sold price'] ,label="Weekend 2")
plt.legend(loc='best')


# In[4]:

#daily avg price for VIP tickets

wk2v = group1.ix[group1.VIP != 0]
wk2v = wk2v.ix[wk2v.weekend != 1]
wk2v = wk2v.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

wk1v = group1.ix[group1.VIP != 0]
wk1v = wk1v.ix[wk1v.weekend != 2]
wk1v = wk1v.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

plt.title('avg daily sold VIP price WK 1 vs WK2')
plt.ylabel('Average Price')
plt.xlabel('Days till the Event')
plt.plot(wk2v['days till'],wk2v['avg sold price'],label="Weekend 2")
plt.plot(wk1v['days till'],wk1v['avg sold price'],label="Weekend 1")
plt.legend(loc='best')


# In[20]:

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
   


# In[24]:

#Google trend week one Vs week two
group1 = ticket.ix[ticket.sold != 0]
wk1 = group1.ix[group1.VIP != 1]
wk1 = wk1.ix[wk1.weekend != 2]
wk1 = wk1.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


wk2 = group1.ix[group1.VIP != 1]
wk2 = wk2.ix[wk2.weekend != 1]
wk2 = wk2.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')


plt.title('Google Search trend')
plt.ylabel('Trend ranking out of 100')
plt.xlabel('Days till the Event')

plt.plot(priceTrend['days till'],priceTrend['coachella weekend 1: (Worldwide)'],label="Weekend 1")
plt.plot(priceTrend2['days till'],priceTrend2['coachella weekend 2: (Worldwide)'],label="Weekend 2")

plt.legend(loc='best')


# In[ ]:



