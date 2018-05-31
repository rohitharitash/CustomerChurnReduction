
# coding: utf-8

# # Customer Churn Reduction  - Random forest
# 
# Finding wether a customer will churn out or not. Random forest was use.

# In[1]:


# importing requried libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import time
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from pandas_ml import ConfusionMatrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading train data
inputTrain = pd.read_csv("Train_data.csv")
# Reading test data
inputTest = pd.read_csv("Test_data.csv")


# In[3]:


# change column names for ease of use and display first 5 rows
inputTrain.columns = inputTrain.columns.str.lower().str.replace(' ', '_')
inputTest.columns = inputTest.columns.str.lower().str.replace(' ', '_')


# In[4]:


#dimensions 
inputTrain.shape
inputTest.shape


# In[5]:


# calculating quick summary statstic for continous predictors
inputTrain.describe()


# In[6]:


#removing variables which not helpfull in churn reduction. Using these variables dosn't make sense.
#for training data
inputTrain = inputTrain.drop(['state','area_code','phone_number'], axis = 1)
inputTest = inputTest.drop(['state','area_code','phone_number'], axis = 1)


# In[7]:


# sanity check
print(inputTrain.shape)
print(inputTest.shape)


# In[8]:


print("Train dataset",inputTrain.churn.value_counts())
print("Test data", inputTest.churn.value_counts())


# In[9]:


# encoding categorical and target variable to binary
# converting international_plan,voice_mail_plan and churn

le = LabelEncoder()

inputTrain.international_plan = le.fit_transform(inputTrain.international_plan)
inputTrain.voice_mail_plan = le.fit_transform(inputTrain.voice_mail_plan)
inputTrain.churn = le.fit_transform(inputTrain.churn)

inputTest.international_plan = le.fit_transform(inputTest.international_plan)
inputTest.voice_mail_plan = le.fit_transform(inputTest.voice_mail_plan)
inputTest.churn = le.fit_transform(inputTest.churn)


# In[10]:


# selecting predictors
train_feature_space = inputTrain.iloc[:,inputTrain.columns != 'churn']
# selecting target class
target_class = inputTrain.iloc[:,inputTrain.columns == 'churn']


# In[11]:


# creating training and validation set
training_set, validation_set, train_taget, validation_target = train_test_split(train_feature_space,
                                                                    target_class,
                                                                    test_size = 0.30, 
                                                                    random_state = 12345)

# Cleaning test sets to avoid future warning messages
train_taget = train_taget.values.ravel() 
validation_target = validation_target.values.ravel() 


# ## Random forest Implementation 

# In[12]:


# using random forest classifier. setting a random state
fit_randomForest = RandomForestClassifier(random_state=12345)


# ### Hyperparameters Optimization¶
# 
# Utilizing the RandomizedSearchCV functionality, we create a dictionary with parameters we are looking to optimize to create the best model for our data.

# In[13]:


np.random.seed(12)
start = time.time()

# selecting best max_depth, maximum features, split criterion and number of trees
param_dist = {'max_depth': [2,4,6,8,10],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2',None],
              "criterion": ["gini", "entropy"],
              "n_estimators" : [100 ,200 ,300 ,400 ,500]
             }
cv_randomForest = RandomizedSearchCV(fit_randomForest, cv = 10,
                     param_distributions = param_dist, 
                     n_iter = 10)

cv_randomForest.fit(training_set, train_taget)
print('Best Parameters using random search: \n', 
      cv_randomForest.best_params_)
end = time.time()
print('Time taken in random search: {0: .2f}'.format(end - start))


# In[39]:


# Set best parameters given by random search 
fit_randomForest.set_params(criterion = 'gini',
                  max_features = 'auto', 
                  max_depth = 10,
                  n_estimators = 100
                )


# In[40]:


fit_randomForest.fit(training_set, train_taget)


# In[16]:


importances_rf = fit_randomForest.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]


# In[17]:


def variable_importance_plot(importance, indices,training_set):
    
    index = np.arange(len(training_set.columns))
   
    
    importance_desc = sorted(importance) 
    feature_space = []
    for i in range(16, -1, -1):
        feature_space.append(training_set.columns[indices[i]])

    fig, ax = plt.subplots(figsize=(14, 14))

    ax.set_facecolor('#fafafa')
    plt.title('Feature importances for Random Forest Model for Customer churn')
    plt.barh(index, 
             importance_desc,
             align="center", 
             color = '#875FDB')
    plt.yticks(index, 
               feature_space)
    plt.xlim(0, max(importance_desc))
    plt.xlabel('Mean Decrease in gini')
    plt.ylabel('Feature')
    plt.savefig('VarImp.png')
    #savefig('VarImp.pdf')
    plt.show()
   
    
    plt.close()


# In[18]:


variable_importance_plot(importances_rf, indices_rf,training_set)


# ## perfoming Cross validation

# In[19]:


# function to perform cross validation
def cross_val_metrics(fit, training_set, train_taget,print_results = True):
   
    n = KFold(n_splits=10)
    scores = cross_val_score(fit, 
                         training_set, 
                         train_taget, 
                         cv = n)
    if print_results:
        print("Accuracy: {0: 0.3f} (+/- {1: 0.3f})"              .format(scores.mean(), scores.std() / 2))
    else:
        return scores.mean(), scores.std() / 2


# In[20]:


cross_val_metrics(fit_randomForest, training_set, 
                  train_taget, 
                  print_results = True)


# ## prediction and performance measure on validation set

# In[41]:


predictions_randomForest_validation = fit_randomForest.predict(validation_set)

validation_crosstb = pd.crosstab(index = validation_target,
                           columns = predictions_randomForest_validation)
validation_crosstb = validation_crosstb.rename(columns= {0: 'False', 1: 'True'})
validation_crosstb.index = ['False', 'True']
validation_crosstb.columns.name = 'n = 1000'


# In[42]:


# confusion matrix of validation set
validation_crosstb


# In[23]:


# mean accuracy on validation set
accuracy_randomForest_val = fit_randomForest.score(validation_set, validation_target)
print(' Mean accuracy on validation set',accuracy_randomForest_val)


# In[24]:


# calculating test error rate on validation set
test_error_rate = 1 - accuracy_randomForest_val
print(' Test error rate on validation set',test_error_rate)


# In[25]:


#classification report on validation set
target_names =[0,1]

validation_report = classification_report(predictions_randomForest_validation, validation_target,  target_names )
print(validation_report)


# ## prediction and performance measure on test set

# In[43]:


#selecting predictors
test_set = inputTest.iloc[:,inputTest.columns != 'churn']
# selecting target class
test_target = inputTest.iloc[:,inputTest.columns == 'churn']

test_prediction = fit_randomForest.predict(test_set)


# In[44]:


#performing prediction on test set
test_prediction = fit_randomForest.predict(test_set)


# In[45]:


# creating confusion matrix of test set
confusion_matrix(test_target,test_prediction)


# In[46]:


test_rf_crosstb = pd.crosstab(index = test_target.churn,
                           columns = test_prediction)
test_rf_crosstb = test_rf_crosstb.rename(columns= {0: 'False', 1: 'True'})
test_rf_crosstb.index = ['False', 'True']
test_rf_crosstb.columns.name = 'n = 1667'


# In[47]:


test_rf_crosstb


# In[31]:


# mean accuracy on test set
accuracy_randomForest_test = fit_randomForest.score(test_set, test_target.churn)
print(' Mean accuracy on test set',accuracy_randomForest_test)


# In[32]:


# calculating test error rate on test set
test_error_rate_testset = 1 - accuracy_randomForest_test
print(' Test error rate on test set',test_error_rate)


# In[33]:


#classification report on test set
target_names =[0,1]

test_report = classification_report(test_prediction, test_target.churn,  target_names )
print(test_report)


# In[35]:


cm = ConfusionMatrix(test_target.churn, test_prediction)


# In[38]:


cm


# In[48]:


cm.print_stats()


# In[69]:


# Model input and output
test_set.to_csv('inputRandomForestPython.csv', encoding = 'utf-8', index = False)
pd.DataFrame(train_taget).to_csv('targetInputRandomForestPython.csv', index = False)
pd.DataFrame(test_prediction, columns=['predictions']).to_csv('outputRandomForestPython.csv')

