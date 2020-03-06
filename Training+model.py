
# coding: utf-8

# In[1]:


get_ipython().magic('matplotlib inline')
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

ticket = pd.read_csv('ticketDatafinal.csv', sep="\t", encoding='latin-1',header=0)
ticket = pd.read_pickle('ticketwithoutlog.pkl')


#New dataset with just avg price and days used for predicting price trend with the LSTM model

#tic = data.ix[data.weekend != 0]
groupSold = ticket.ix[ticket.sold != 0]
#group2 = ticket.ix[ticket.sold != 1]
#group1 = group1.ix[group1.VIP != 2]
groupNonVIP = groupSold.ix[groupSold.VIP != 1]

groupWK1 = groupNonVIP.ix[groupNonVIP.weekend1 == 1]
groupWK2 = groupNonVIP.ix[groupNonVIP.weekend2 == 1]


dailyPriceWK1= groupWK1.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')
dailyPriceWK2 = groupWK2.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

#groups1ss = group11.groupby(['days till']).ticketPrice.mean().reset_index(name='avg sold price')

#groups2 = group2.groupby(['days till','weekend']).ticketPrice.mean().reset_index(name='avg unsold price')






# In[2]:


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

import matplotlib.image as mpimg
import seaborn as sns
import itertools



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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.kernel_ridge import KernelRidge
from catboost import Pool, CatBoostRegressor, cv, CatBoostClassifier
from sklearn import cross_validation,preprocessing,tree
from sklearn.metrics import r2_score, make_scorer, mean_squared_error 
from sklearn.metrics import log_loss 


# # Cross Validation

# In[3]:


y = np.array(ticket['sold'])
x = np.array(ticket[["# of ticket", "CAR CAMPING PASS", "VIP", "VIP Parking", "weekend1","weekend2","days till","PriceRange",
                      "ticketPrice","trend","Price","Shuttle Passes","avg unsold price","avg sold price"  ]])
#Split data 80% training 20% test.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=2)

#Transform the scale of the x data sets.
stdscaler = preprocessing.StandardScaler().fit(X_train)
#Perform standardization by centering and scaling
X_train_scaled = stdscaler.transform(X_train)
X_test_scaled  = stdscaler.transform(X_test)


# # Four main models where trained with the data set.
# 1)Random forest
# 
# 2)Neural network
# 
# 3)LSTM RNN
# 
# 4)Ensemble of weak leaners

# # Random forest
# Since random forest is easy to setup, will use this as the base model. This model will be used to see if an action would sell or not.
# This model had an accuracy of 88 percent

# In[11]:


n_estimators_list = [10,100,500,1000]
rfc = RandomForestClassifier(random_state=1,oob_score = True,n_jobs = 1)
grid = GridSearchCV(estimator=rfc, param_grid=dict(n_estimators=n_estimators_list,min_samples_leaf = [1,2,3,5,10] ))

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
print ("Accuracy is ", accuracy_score(y_test,predicted)*100)
predicted
# show the inputs and predicted outputs
#predictedValue = [round(x[0]) for x in predicted]
#print(predictedValue)

#for i in range(len(X_train)):
#	print("X=%s, Predicted=%s" % (X_train[i], predicted[i]))


# # Neural network
# After a the modeling trainging, the best results was a two lawer network. I didnt expect a neural network to perform that well with the limited amout of training data I have.
# This model had an accuracy of 71 percent. 
# 

# In[12]:


#First model, neural network to predict if the ticket will sell.


#Implementing a neural network.
model = Sequential()

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




# # LSTM Neural Network
# This model is used to to find trend and predict what the price will be in the furture.
# 

# In[16]:



#Create a new column with shifted price range, this will be what we are trying to predict
data = pd.concat ([dailyPriceWK1, dailyPriceWK1["avg sold price"].shift (-1)], axis =1)
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
plt.title('Predicted Price Trend')

ax = plt.axes()
ax.yaxis.set_major_formatter(tick)
 
plt.show()



# # Ensemble of weak leaners
# A total of eight models.

# In[4]:



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y, y_pred):
    
    assert len(y) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y))**2))

def root_mean_squared_error(y_true, y_pred):
        n = len(y_true)

        return np.sqrt( 1/n*np.sum((y_pred-y_true)**2) )


# In[5]:


#stacking models
y = np.array(ticket['sold'])

#somereason it does not like weekend or avg sold price
x = np.array(ticket[["# of ticket","ticketPrice", "CAR CAMPING PASS", "VIP", "VIP Parking", "days till","PriceRange"
                    ,"trend","Price","Shuttle Passes","avg unsold price","weekend1","weekend2","avg sold price"
                    ]])
#Split data 80% training 20% test.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2, random_state=2)

# Set up variables
#X_train = ticket
#X_test = test.drop(['id'], axis=1)
#X_train = X_train.drop(['id'], axis=1)

#y_train = train1['revenue']

#y_train = np.log(y_train)

#X_train=X_train.drop(['revenue'], axis=1)

#This will ensure that all rmse scores produced have 
#been smoothed out across the entire dataset and are 
#not a result of any irregularities, which otherwise
#would provide a misleading representation of model
#performance. 

#Define a evaluation matrix 
from sklearn.metrics.scorer import make_scorer

RMSLE = make_scorer(log_loss)



# Defining two rmse_cv functions
#CV=10 is 10 folds
def rmse_cv(model):
    rmse = np.sqrt(cross_val_score(model, X_train, y_train, scoring=RMSLE, cv = 10))

   

    #rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring=rmsle, cv = 10))
    return(rmse)


# In[6]:


np.isfinite(X_train).all()


# # Ridge Regression (L2 Regularisation)
# Ridge regression shrinks the regression coefficients, so that variables, with minor contribution to the outcome, have their coefficients close to zero.
# 
# Alpha is a regularization parameter that measures how flexible our model is.
# The higher the regularization the less prone our model will be to overfit. 
# The optimal value will have the lowest RMSE on the graph.

# In[23]:






# Setting up list of alpha's

alphas = [0.0001,0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 28,29,30,31,32,33,50,75,85,100,200,300,400,500]

# Iterate over alpha's
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
# Plot findings
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# # Lasso Regression (L1 regularisation)
# It shrinks the regression coefficients toward zero by penalizing the regression model
# with a penalty term called L1-norm, which is the sum of the absolute coefficients.
# 
# In the case of lasso regression, the penalty has the effect of forcing some of the coefficient estimates, with a minor contribution to the model, to be exactly equal to zero.
# This can be also seen as an alternative to the subset selection methods for performing variable selection in order to reduce the complexity of the model.
# 
# 
# 

# In[24]:




# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# Iterate over alpha's
cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_lasso = pd.Series(cv_lasso, index = alphas)
cv_lasso.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# # ElasticNet Regression
# Elastic Net produces a regression model that is penalized with both the L1-norm and L2-norm. 
# The consequence of this is to effectively shrink coefficients (like in ridge regression) and to set some coefficients to zero (as in LASSO).

# In[6]:



# Setting up list of alpha's
alphas = [0.01, 0.005, 0.001, 0.0005, 0.0001]

# Iterate over alpha's
cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_elastic = pd.Series(cv_elastic, index = alphas)
cv_elastic.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# # 4. Kernel ridge regression
# Kernel ridge regression (KRR) combines Ridge Regression (linear least squares with l2-norm regularization) with the 'kernel trick'

# In[8]:


# Setting up list of alpha's
alphas = [30,25,20,15,10,5,1,0.1,0.01,0.001]

# Iterate over alpha's
cv_krr = [rmse_cv(KernelRidge(alpha = alpha)).mean() for alpha in alphas]

# Plot findings
cv_krr = pd.Series(cv_krr, index = alphas)
cv_krr.plot(title = "Validation")
plt.xlabel("Alpha")
plt.ylabel("Rmse")


# # Model initazing

# In[6]:


#Differnet models that are initiazing 
#1. Ridge Regression
model_ridge = Ridge(alpha = 100)
#lasso  Pipeline to scale features
model_lasso  = make_pipeline(RobustScaler(), Lasso(alpha =0.005, random_state=1))
#Elastic net 
model_elastic  = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=.9, random_state=3))
#Kernel Ridge Regression :

#Kernel: Polynomial-his means that the algorithm will not just consider 
#similarity between features, but also similarity between combinations of features.
#Degree & Coef0: These are used to define the precise structure of the Polynomial kernel.  
    
model_krr  =make_pipeline(RobustScaler(), KernelRidge(alpha=0.001, kernel='polynomial', degree=2.65, coef0=6.9))


# # Boosting 
# Ensemble technique in which the predictors are not made independently, but sequentially.
# It is used to for reducing bias and variance in supervised learning.
# It combines multiple weak predictors to a build strong predictor, But we have to choose the stopping criteria carefully or it could lead to overfitting.
# 
# Notes:
# Random forest is a bagging technique and not a boosting technique.
# In boosting as the name suggests, one is learning from other which in turn boosts the learning.
# The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.

# # Initalze boosting models

# In[6]:




model_cat = CatBoostRegressor(iterations=1350,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=21,
                              loss_function='Logloss',
                              verbose=50)

# Initiating parameters ready for CatBoost's CV function, which I will use below
params = {'iterations':1350,
          'learning_rate':0.05,
          'depth':3,
          'l2_leaf_reg':4,
          'border_count':21,
          'loss_function':'log_loss',
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
# Using grid search and passing in a range of values to find which value performs the best for a parmeter.

# # CatBoost
# A gradient boosting algorithm able to work with categorical features without any prior processing needed.
# 

# In[6]:


'''
iterations-num of trees
l2_leaf_reg-Coefficient at the L2 regularization term of the cost function.
border_count-The number of splits for numerical features.
loss_function-For 2-class classification use 'LogLoss' or 'CrossEntropy'. For multiclass use 'MultiClass'.
ctr_border_count-The number of splits for categorical features.
'''

#best performing model
model_cat = CatBoostClassifier(iterations=2250,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=9,
                              border_count=15,
                              loss_function='Logloss',
                              verbose=50)


# In[13]:


#find optimal estimator and learning rate using cv

scorer = make_scorer(r2_score)


model_cat = CatBoostClassifier(
                              depth=3,
                              l2_leaf_reg=4,
                              border_count=21,
                              loss_function='Logloss',
                              verbose=50)

p_test3 = {'learning_rate':[0.5,0.1,0.05,0.01,0.005,0.001], 'iterations':range(500, 5000, 50)}


grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)



grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[7]:


#find optimal depth and border_count using cv



scorer = make_scorer(r2_score)


model_cat = CatBoostClassifier(iterations=2250,
                              learning_rate=0.05,
                              
                              l2_leaf_reg=9,
                              
                              loss_function='Logloss',
                              verbose=50)

p_test3 = {'border_count':range(5, 25, 2), 'depth':range(1, 10, 1)}



grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)
grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[8]:


#find optimal verbose using cv



scorer = make_scorer(r2_score)


model_cat = CatBoostClassifier(iterations=2250,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=9,
                              border_count=15,
                              loss_function='Logloss')

p_test3 = {'verbose':range(50, 400, 25)}

grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[9]:


#find optimal l2_leaf_regusing cv


scorer = make_scorer(r2_score)


model_cat = CatBoostClassifier(iterations=2250,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=9,
                              border_count=15,
                              loss_function='Logloss',
                              verbose=50)



p_test3 = {'l2_leaf_reg':range(0, 10, 1)}


grid = GridSearchCV(estimator=model_cat, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# # LightGBM

# In[ ]:


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

'''

#Tuned model
model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              learning_rate=0.01, n_estimators=1200,
                              max_bin = 40, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)





# In[14]:


#find optimal estimator and learning rate using cv


scorer = make_scorer(r2_score)


model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              
                              max_bin = 55, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)

p_test3 = {'learning_rate':[0.1,0.05,0.01,0.001], 'n_estimators':range(500, 3000, 50)}

grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[11]:


#max_bin ,num_leaves tune



scorer = make_scorer(r2_score)


model_lgb = lgb.LGBMClassifier(objective='binary',
                              learning_rate=0.01, n_estimators=1200,
                               bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)

p_test3 = {'max_bin':range(10, 70, 5), 'num_leaves':range(2, 10, 2)}

grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[12]:


#bagging_freq ,bagging_fraction tune


scorer = make_scorer(r2_score)
model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              learning_rate=0.01, n_estimators=1000,
                              max_bin = 55, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)


p_test3 = {'bagging_freq':range(1, 8, 1), 'bagging_fraction':[0.1,0.2,0.3,.4,.5,.6,.7,.8,.9]}


grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[13]:


#min_data_in_leaf ,feature_fraction tune



scorer = make_scorer(r2_score)

model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              learning_rate=0.01, n_estimators=1000,
                              max_bin = 55, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)

p_test3 = {'min_data_in_leaf':range(1, 5, 1), 'feature_fraction':[0.25,0.2,0.3,.2319]}


grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[12]:


#min_sum_hessian_in_leaf



scorer = make_scorer(r2_score)


model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              learning_rate=0.01, n_estimators=1200,
                              max_bin = 40, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)



p_test3 = {'min_sum_hessian_in_leaf':range(1, 15, 1)}



grid = GridSearchCV(estimator=model_lgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# # XGBoost

# In[11]:



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
    
model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.4,
                             learning_rate=0.5, max_depth=9, 
                             min_child_weight=2, n_estimators=1575,
                             reg_alpha=1,
                             subsample=0.9, silent=1,
                             random_state =7)




scorer = make_scorer(r2_score)


# In[15]:


#find optimal estimator and learning rate using cv



scorer = make_scorer(r2_score)


model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.4,
                              max_depth=8, 
                             min_child_weight=3,
                             reg_alpha=0.1,
                             subsample=0.9, silent=1,
                             random_state =7)

p_test3 = {'learning_rate':[0.5,0.05,0.025,0.01], 'n_estimators':range(900, 3000, 50)}

grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[13]:


#Max depth min child weight tune
scorer = make_scorer(r2_score)
log_loss

model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.4,
                             learning_rate=0.5 
                             , n_estimators=1575,
                             reg_alpha=0.1,
                             subsample=0.9, silent=1,
                             random_state =7)


p_test3 = {'max_depth':range(3,10,1), 'min_child_weight':range(1,6,1)}



grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[14]:


#gamma

model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, 
                             learning_rate=0.5, max_depth=9, 
                             min_child_weight=2, n_estimators=1575,
                             reg_alpha=0.1,
                             subsample=0.9, silent=1,
                             random_state =7)






p_test3 = {'gamma':[i/10.0 for i in range(0,5)]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[15]:


# Tune subsample and colsample_bytree
scorer = make_scorer(r2_score)

model_xgb = xgb.XGBClassifier( gamma=0.4,
                             learning_rate=0.5, max_depth=9, 
                             min_child_weight=2, n_estimators=1575,
                             reg_alpha=0.1,
                              silent=1,
                             random_state =7)



p_test3 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[16]:


#Tuning Regularization Parameters
    
    
model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.4,
                             learning_rate=0.5, max_depth=9, 
                             min_child_weight=2, n_estimators=1575,
                            
                             subsample=0.9, silent=1,
                             random_state =7)



p_test3 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}


grid = GridSearchCV(estimator=model_xgb, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# # Gradient Boosting Regression

# In[6]:


'''
Tree-Specific Parameters: These affect each individual tree in the model.

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

#if n_estimators'> 100 try higher rate since tuning others will take longer, if less then 20, lower the rate.

#loss function 
#R2- tells you the x% of the variablity in rev can be explained by the variablity of the variables. So
#so the model can only explain X% of the diff  of what the movie makes the certain amount.

'''

#
model_gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=42,subsample= 0.7,  min_samples_split=2, 
                                   loss='deviance', random_state =5)




scorer = make_scorer(r2_score)



# In[16]:


#find optimal estimator using cv


scorer = make_scorer(r2_score)


model_gbr = GradientBoostingClassifier(
                                   max_depth=6, max_features='sqrt',
                                   min_samples_leaf=42,subsample= 0.8,  min_samples_split=2, 
                                   loss='deviance', random_state =5)

p_test3 = {'learning_rate':[0.5, 0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750,2000,2500]}


grid = GridSearchCV(estimator=model_gbr, param_grid=p_test3,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[7]:


#Tune the max_depth,min sample split

model_gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005,
                                    max_features='sqrt',
                                   min_samples_leaf=42,subsample= 0.8, 
                                   loss='deviance', random_state =5)
param_test2 = {'max_depth':range(3,16,1), 'min_samples_split':range(2,100,5)}


grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[8]:


#Tune the min_samples_leaf
#

model_gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005,
                                   max_depth=6, max_features='sqrt',min_samples_split=2,
                                   subsample= 0.8, 
                                   loss='deviance', random_state =5)




param_test2 = {'min_samples_leaf':range(2,70,5)}


grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# In[9]:


#Tune the subsample
#

model_gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005,
                                   max_depth=6, max_features='sqrt',
                                   min_samples_leaf=42,  min_samples_split=2, subsample= 0.8, 
                                   loss='deviance', random_state =5)


param_test2 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

grid = GridSearchCV(estimator=model_gbr, param_grid=param_test2 ,scoring = scorer)

grid_fit = grid.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_
grid_fit.grid_scores_, grid_fit.best_params_, grid_fit.best_score_ ,grid_fit.best_estimator_


# # Tuned Model initializing

# In[7]:


model_gbr = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005,
                                   max_depth=5, max_features='sqrt',
                                   min_samples_leaf=42,subsample= 0.7,  min_samples_split=2, 
                                   loss='deviance', random_state =5)






#XGBoost 
model_xgb = xgb.XGBClassifier(colsample_bytree=0.6, gamma=0.4,
                             learning_rate=0.5, max_depth=9, 
                             min_child_weight=2, n_estimators=1575,
                             reg_alpha=1,
                             subsample=0.9, silent=1,
                             random_state =7)



#LightGBM :
model_lgb = lgb.LGBMClassifier(objective='binary',num_leaves=8,
                              learning_rate=0.01, n_estimators=1200,
                              max_bin = 40, bagging_fraction = 0.9,
                              bagging_freq = 4, feature_fraction = 0.25,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =1, min_sum_hessian_in_leaf = 7)
#catboost
model_cat = CatBoostClassifier(iterations=2250,
                              learning_rate=0.05,
                              depth=3,
                              l2_leaf_reg=9,
                              border_count=15,
                              loss_function='Logloss',
                              verbose=50)


# Initiating parameters ready for CatBoost's CV function, which I will use below
params = {'iterations':2000,
          'learning_rate':0.10,
          'depth':3,
          'l2_leaf_reg':4,
          'border_count':15,
          'loss_function':'RMSE',
          'verbose':200}


# # Model training

# In[8]:


#run the custom rmse_cv function on each algorithm to understand each model's performance.


# Fitting all models with rmse_cv function, apart from CatBoost
cv_ridge = rmse_cv(model_ridge).mean()
cv_lasso = rmse_cv(model_lasso).mean()
cv_elastic = rmse_cv(model_elastic).mean()
#failing need to check why
#cv_krr = rmse_cv(model_krr).mean()
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


# In[9]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Gradient Boosting Regressor',
              'XGBoost Regressor',
              'Light Gradient Boosting Regressor',
              'CatBoost','Ridge Regression', 'Lasso Regression', 'ElasticNet'],
    'Score': [cv_gbr,
              cv_xgb,
              cv_lgb,
              cv_cat,
             cv_ridge,
             cv_lasso,
             cv_elastic]})

# Build dataframe of values
result_df = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
result_df.head(8)


# # Prediction of the test data

# In[10]:


# Fit and predict all models
#only take exponent if we applied log transform to data 
model_lasso.fit(X_train, y_train)
lasso_pred = np.expm1(model_lasso.predict(X_test))

model_elastic.fit(X_train, y_train)
elastic_pred = np.expm1(model_elastic.predict(X_test))

model_ridge.fit(X_train, y_train)
ridge_pred = np.expm1(model_ridge.predict(X_test))

model_xgb.fit(X_train, y_train)
xgb_pred = (model_xgb.predict(X_test))

model_gbr.fit(X_train, y_train)
gbr_pred = (model_gbr.predict(X_test))

model_lgb.fit(X_train, y_train)
lgb_pred = (model_lgb.predict(X_test))

#model_krr.fit(X_train, y_train)
#krr_pred = np.expm1(model_krr.predict(X_test))

model_cat.fit(X_train, y_train)
cat_pred = (model_cat.predict(X_test))


# In[11]:


# Create stacked model
#stacked = (cat_pred) 

stacked = ( xgb_pred + lgb_pred +cat_pred + gbr_pred  + lasso_pred + elastic_pred + ridge_pred) / 7
#since  data was transformed with log need to reverse it
#stacked = np.exp(stacked) 


# In[ ]:


my_submission1 = pd.DataFrame({'id': test.id, 'revenue': stacked})
my_submission1.to_csv('submission-070418.csv', index=False)


# In[18]:



print ("Accuracy is ", accuracy_score(y_test,xgb_pred)*100)
print ("Accuracy is ", accuracy_score(y_test,lgb_pred)*100)
print ("Accuracy is ", accuracy_score(y_test,cat_pred)*100)
print ("Accuracy is ", accuracy_score(y_test,gbr_pred)*100)

