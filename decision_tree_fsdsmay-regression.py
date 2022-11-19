
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('household_power_consumption.csv', sep=';',
                  parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'],index_col='dt')
data = df
data.ffill(axis=0,inplace=True)
data['consumption'] = (data['Global_active_power']*1000/60) - (data['Sub_metering_1'] + data['Sub_metering_2'] + data['Sub_metering_3'])

data['Date'] = data.index.date
data['time'] = data.index.time

data['Date'] = data['Date'].astype(str)
data['time'] = data['time'].astype(str)



# In[4]:


data['exact_time'] = data['Date']+";"+df['time']
data['exact_time_DT'] = pd.to_datetime(data['exact_time'],format="%Y-%m-%d;%H:%M:%S")
data = data.drop(['Date', 'time','exact_time'],axis = 1).sort_values(by=['exact_time_DT'])


# In[5]:


data1 = data.groupby(pd.Grouper(key='exact_time_DT',freq='M')).sum()


# In[6]:


data1.head()


# In[7]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor


# In[9]:


y = data1["consumption"]
X = data1.drop("consumption", axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=10)


# # Simple Decision Tree Regressor

# In[10]:


model = DecisionTreeRegressor()


# In[11]:


model.fit(X_train,y_train)


# In[12]:


model.score(X_train,y_train)


# In[17]:


model.score(X_test,y_test)


# In[13]:


y_predict=model.predict(X_test)


# In[16]:


from sklearn.metrics import r2_score


# In[18]:


#R^2 score of the model
r2_score(y_test,y_predict)


# # Decision Tree Regressor with best hyper parameters
# 

# In[73]:


grid_param = {
    'criterion' : ['friedman_mse','mse','mae'],
    'max_depth' : range(2,10,1),
    'min_samples_leaf' : range(1,8,1),
    'min_samples_split': range(2,8,1),
    'splitter' : ['best', 'random']
}


# In[77]:


from sklearn.model_selection import GridSearchCV
grid_searh=GridSearchCV(estimator=model,param_grid=grid_param,cv=4,verbose=1)


# In[78]:


grid_searh.fit(X_train,y_train)


# In[79]:


grid_searh.best_params_


# In[85]:


#Model with best parameters

#model_with_best_params=DecisionTreeRegressor(criterion= 'mae',max_depth= 8,min_samples_leaf= 1,min_samples_split= 3,splitter='random')

model_with_best_params=DecisionTreeRegressor(criterion= 'friedman_mse',max_depth= 5,min_samples_leaf= 1,min_samples_split= 4,splitter='random')


# In[86]:


model_with_best_params.fit(X_train,y_train)
model_with_best_params.score(X_train,y_train)
model_with_best_params.score(X_test,y_test)

y_predict_new=model_with_best_params.predict(X_test)

#R^2 score of the model
r2_score(y_test,y_predict_new)


# In[87]:


model_with_best_params.score(X_test,y_test)


# ### GETTING POOR PERFORMANCE AFTER FINDING BEST PARAMETERS?!!!
