
# coding: utf-8

# # Customer Churn Reduction - Logistic Regression
# In the notebook, we will use logistic regression to predict whether a customer will churn or not. 

# ### importing librarires 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotnine import *
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from pandas_ml import ConfusionMatrix


# In[2]:


# Read in train and test data
train_data = pd.read_csv("Train_data.csv")
test_data = pd.read_csv("Test_data.csv")


# In[3]:


# change column names for ease of use and display first 5 rows
train_data.columns = train_data.columns.str.lower().str.replace(' ','_')
test_data.columns = test_data.columns.str.lower().str.replace(' ','_')


# In[4]:


print(" number of rows and columns in train data ",train_data.shape)


# In[5]:


print(" number of rows and columns in test data ",test_data.shape)
    


# In[6]:


train_data.describe()
train_data.head()


# In[7]:


#removing variables which not helpfull in churn reduction. Using these variables dosn't make sense.
#for training data
train_data = train_data.drop(['state','area_code','phone_number'], axis = 1)
test_data = test_data.drop(['state','area_code','phone_number'], axis = 1)


# In[8]:


# checking for missing values in train
train_data.isnull().sum()


# In[9]:


# checking for missing values in test dataset
test_data.isnull().sum()


# ##### There are no missing values in train and test data.

# Now looking at target variable.
# churn: This is the target variable. Churn is defined as whether the customer leaves the services or not. churn = True means customer left ,churn = false means customer stays

# In[10]:


plt.figure(figsize=(8,6))
sns.set_style('ticks')
sns.countplot(train_data.churn,palette='summer')
plt.xlabel('Customer churn')


# In[11]:


# churn ratio of customers with and without internation plan
ggplot(train_data) +     aes('international_plan', fill = 'churn') +    geom_bar(position = "fill", color= 'blue') + labs(x = "International plan", y = "") +     ggtitle("Churn ratio of customers with and without international plan") +     theme(figure_size=(6, 4))


# #### Customers with international plan are churning out more as compare to domestic customers. 

# In[12]:


# churn ratio of customers with voice_mail_plan
ggplot(train_data) +     aes('voice_mail_plan', fill = 'churn') +    geom_bar(position = "fill", color = 'blue') +     labs(x =  "voice mail plan", y = "") +     ggtitle("Churn ratio with customers with and without voice mail plan") +     theme(figure_size=(6, 4))


# #### Customers without voice mail plan are churning out more as compare to customers with voice mail plan.

# In[13]:


# # churn ratio of customers with respect to service calls
ggplot(train_data) +     aes('number_customer_service_calls', fill = 'churn') +    geom_bar(position = "fill", color = 'blue') +     labs(x = "Customer service calls", y = "")  +     ggtitle(" Churn ratio with service call frequency") +     theme(figure_size=(6, 4))


# #### customers with higher service calls ie > 3 are churning out more.

# ### 3. Correlation matrix for continous predictors

# In[14]:


churn_corr = train_data.corr()
cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]

churn_corr.style.background_gradient(cmap, axis=1)    .set_properties(**{'max-width': '90px', 'font-size': '10pt'})    .set_caption("Correlation matrix")    .set_precision(2)    .set_table_styles(magnify())


# ### From corrplot we can see that dataset consist of multicollinearity
# 1. total.day.minutes and total.day.charge are highly collinear
# 2. total.eve.minutes and total.eve.charge are highly collinear
# 3. total.night.minutes and total.night.charge are highly collinear
# 4. total.intl.minutes and total.intl.charge are highly collinear
# 
# Multi-collinearity voilates the assumption of logistic regression. So we will be removing one of these predictors 
# from the model. 

# ### 4. Exploring continous predictors

# In[15]:


# function for exploring distributions by continuous predictors with there summary stats
def countPred_eda(train_data, variableName, targetVariable):
    print(train_data[variableName].describe())
    return ggplot(train_data) +     aes(targetVariable, variableName, fill = targetVariable) + geom_boxplot(alpha = .8, outlier_color = "green") +     labs(x =  targetVariable, y = variableName) +     ggtitle("Churn ratio with "+ variableName) +     theme(figure_size=(6, 4))
     
    


# In[16]:


# --- total_day_minutes --- #
countPred_eda(train_data,'total_day_minutes','churn')


# #### It is evident from above plot that churn rate is higher when count of total_day_minute is higher

# In[17]:


# --- total_day_calls ---- #
countPred_eda(train_data,'total_day_calls','churn')


# In[18]:


# --- total_day_charge --- #
countPred_eda(train_data,'total_day_charge','churn')


# In[19]:


# --- total_eve_minutes --- #
countPred_eda(train_data,'total_eve_minutes','churn')


# In[20]:


# --- total_eve_calls --- #
countPred_eda(train_data,'total_eve_calls','churn')


# In[21]:


# --- total_eve_charge --- #
countPred_eda(train_data,'total_eve_charge','churn')


# In[22]:


# --- total_night_minutes --- #
countPred_eda(train_data,'total_night_minutes','churn')


# In[23]:


# --- total_night_calls --- #
countPred_eda(train_data,'total_night_calls','churn')


# In[24]:


# --- total_night_charge --- #
countPred_eda(train_data,'total_night_charge','churn')


# In[25]:


# --- total_intl_minutes --- #
countPred_eda(train_data,'total_intl_minutes','churn')


# In[26]:


# --- total_intl_calls --- #
countPred_eda(train_data,'total_intl_calls','churn')


# In[27]:


# --- total_intl_charge --- #
countPred_eda(train_data,'total_intl_charge','churn')


# In[28]:


# --- number_customer_service_calls --- #
countPred_eda(train_data,'number_customer_service_calls','churn')


# In[29]:


# Removing highly collinear variables from train and test
train_data = train_data.drop(['total_day_charge','total_eve_charge','total_night_charge', 'total_intl_charge'], axis = 1)
test_data = test_data.drop(['total_day_charge','total_eve_charge','total_night_charge', 'total_intl_charge'], axis = 1)


# In[30]:


# encoding target variables 
le = LabelEncoder()
# for train data
train_data.churn = le.fit_transform(train_data.churn)
# for test data
test_data.churn = le.fit_transform(test_data.churn)


# In[31]:


# Encoding categorical variables
# for train data
train_data.international_plan = le.fit_transform(train_data.international_plan)
train_data.voice_mail_plan = le.fit_transform(train_data.voice_mail_plan)

# for test data
test_data.international_plan = le.fit_transform(test_data.international_plan)
test_data.voice_mail_plan = le.fit_transform(test_data.voice_mail_plan)


# In[32]:


test_data.head()


# In[33]:


train_data.churn.value_counts()


# In[34]:


# selecting predictors
train_feature_space = train_data.iloc[:,train_data.columns != 'churn']
# selecting target class
target_class = train_data.iloc[:,train_data.columns == 'churn']


# In[35]:


# creating training and validation set
training_set, validation_set, train_taget, validation_target = train_test_split(train_feature_space,
                                                                    target_class,
                                                                    test_size = 0.30, 
                                                                    random_state = 456)

# Cleaning test sets to avoid future warning messages
train_taget = train_taget.values.ravel() 
validation_target = validation_target.values.ravel() 


# In[36]:


# logistic regression classifier 
classifier_logit_default = LogisticRegression(random_state=456)


# In[37]:


classifier_logit_default.fit(training_set, train_taget)


# In[38]:


# Predicting the validation set results 
validation_prediction = classifier_logit_default.predict(validation_set)


# In[39]:


# confusion matrix for validation set
validation_logit_crosstb = pd.crosstab(index = validation_target,
                           columns = validation_prediction)
validation_logit_crosstb = validation_logit_crosstb.rename(columns= {0: 'False', 1: 'True'})
validation_logit_crosstb.index = ['False', 'True']
validation_logit_crosstb.columns.name = 'n = 1000'


# In[40]:


validation_logit_crosstb


# In[41]:


#classification report on validation set
target_names =[0,1]

validation_report = classification_report(validation_prediction, validation_target,  target_names )
print(validation_report)


# In[42]:


mean_accuracy_validation = classifier_logit_default.score(validation_set, validation_target)
print(' Mean accuracy on validation set',mean_accuracy_validation)


# In[43]:


# calculating test error rate on validation set
test_error_rate = 1 - mean_accuracy_validation
print(' Test error rate on validation set',test_error_rate)


# From the confusion matrix we can see that high accuracy of model is due to disproportionate number of non-churn 
# customers predicted correctly. The This model is working great for identifing non churning customer but performing poorly for 
# churning customers. We will tune the model to increase accuracy on churning customer. 

# In[44]:


# model 2
classifier_logit_2 = LogisticRegression(class_weight='balanced') 


param = {'C':[0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2,3]} 
#rs_cv = RandomizedSearchCV(estimator=classifier_logit_2, cv = 10,
                            #n_iter = 100, 
                           #param_distributions=param, random_state=1234) 
rs_cv = GridSearchCV(estimator=classifier_logit_2, cv = 10,param_grid=param) 

rs_cv.fit(training_set,train_taget) 

print('Best parameter :{} Best score :{}'.format(rs_cv.best_params_,rs_cv.best_score_))


# In[45]:


classifier_logit_2.set_params(C = 3)


# In[46]:


classifier_logit_2.fit(training_set, train_taget)


# In[47]:


validation_prediction_tuned = classifier_logit_2.predict(validation_set)


# In[48]:


# confusion matrix for validation set
validation_logit_crosstb1 = pd.crosstab(index = validation_target,
                           columns = validation_prediction_tuned)
validation_logit_crosstb1 = validation_logit_crosstb1.rename(columns= {0: 'False', 1: 'True'})
validation_logit_crosstb1.index = ['False', 'True']
validation_logit_crosstb1.columns.name = 'n = 1000'


# In[49]:


validation_logit_crosstb1


# In[50]:


#classification report on validation set with hypertuning
target_names =[0,1]

validation_report_tuning = classification_report(validation_prediction_tuned, validation_target,  target_names )
print(validation_report_tuning)


# ### Prediction and performance on test data
# 
# Model classifier_logit_2 was selected.

# In[51]:


# test set
test_set = test_data.iloc[:,test_data.columns != 'churn']
# selecting target class for test set
test_set_target = test_data.iloc[:,test_data.columns == 'churn']


# Predicting the test set results
test_prediction = classifier_logit_2.predict(test_set)


# In[52]:


# confusion matrix for validation set
test_logit_crosstb = pd.crosstab(index = test_data.churn,
                           columns = test_prediction)
test_logit_crosstb = test_logit_crosstb.rename(columns= {0: 'False', 1: 'True'})
test_logit_crosstb.index = ['False', 'True']
test_logit_crosstb.columns.name = 'n = onservation'


# In[53]:


cm = ConfusionMatrix(test_data.churn, test_prediction)


# In[61]:


cm


# In[60]:


cm.print_stats()


# In[55]:


#classification report on test set with hypertuning
target_names =[0,1]

test_report_tuning = classification_report(test_prediction, test_data.churn,  target_names )
print(test_report_tuning)


# In[56]:


mean_accuracy_test = classifier_logit_2.score(test_set, test_set_target)
print(' Mean accuracy on test set',mean_accuracy_test)


# In[57]:


# calculating test error rate on test set
test_error_rate_testset = 1 - mean_accuracy_test
print(' Test error rate on test set',test_error_rate)


# In[63]:


test_set_target.churn.value_counts()


# In[65]:


# Model input and output
test_set.to_csv('inputLogisticRegressionPython.csv', encoding = 'utf-8', index = False)
pd.DataFrame(train_taget).to_csv('targetLogisticRegressionPython.csv', index = False)
pd.DataFrame(test_prediction, columns=['predictions']).to_csv('outputLogisticRegressionPython.csv')

