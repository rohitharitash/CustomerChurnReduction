# Churn Reduction Logistic regression implementation

library(stringr)
library(corrplot)
library(caret)
library(dplyr)
library(tidyr)
library(broom)



# One of the assumptions of logistic regression is that there is no high intercorrelations
#among predictors. 
# From EDA we found some variables have multi-collinearity 



# Now we will build our model.
# For the first model we will select all the predictors

fullModel <- glm(Churn ~., data = train_data, family = binomial(link = 'logit'))
summary(fullModel)

# Predicting and checking model performance with full model
pred_fullmodel_val <- predict(fullModel, validation_data[,-18], type = "response")
pred_fullmodel_valT <- ifelse(pred_fullmodel_val < 0.5,1,0)
xtab=table(observed=validation_data[,18],predicted=pred_fullmodel_valT)
fullmodel_confmat_val <- confusionMatrix(xtab)
print(fullmodel_confmat_val)

# now in second model we will remove variables affected by multi-collinearity 
model2 <- glm(Churn ~. -total.day.minutes-total.eve.minutes-
                   total.night.minutes-total.intl.minutes,
                 data = train_data, family = binomial(link = 'logit'))
summary(model2)

# Predicting and checking model performance with model2
pred_model2_val <- predict(model2, validation_data[,-18], type='response')
pred_model2_val <- ifelse(pred_model2_val > 0.5,1,0)
xtab1 <- table(observed = validation_data[,18], predicted = pred_model2_val)
model2_confmat_val <- confusionMatrix(xtab1)

# if we compare fullModel and model2 summary, we can see that 
# In full model due to variables with multi-collineary only international.plan, voice.mail.plan,
# number.customer.service.call are considered statistically significant with AIC score 1540.

# After removing highly collinear variables in model2, variables nternational.plan, voice.mail.plan,
# number.customer.service.call along with total.day.charge, total.eve.charge, total.night.charge,
# and total.intl.charge are considered statistically significant with AIC score 1543.

# model with lower AIC score is considered better

# NOw we will use Stepwise Procedures

#1 backwards elimination
backward <- step(fullModel)
# from backwards elimination approach, we got a model with AIC 1529.02 which is better
#than model2
# training model using best result from 'backward'
model_backward <- glm(Churn ~ international.plan+voice.mail.plan+number.vmail.messages+
                        total.day.minutes+total.eve.minutes+total.night.charge+
                        total.intl.calls+total.intl.charge+number.customer.service.calls,
                      data = train_data, family = binomial(link = 'logit'))
summary(model_backward)

#predicting on validation set
pred_backmodel_val <- predict(model_backward, validation_data[,-18], type = 'response')
pred_backmodel_valT <- ifelse(pred_backmodel_val > 0.5,1,0)
xtab2 <- table(observed = validation_data[,18], predicted = pred_backmodel_valT)
backward_confmat_val <- confusionMatrix(xtab2)



# model_backward is slightly better than model2

# 2. forward elimination
nullModel <- glm(Churn ~ 1, data =train_data, family = binomial)
summary(nullModel)

forward <- step(nullModel, scope = list(lower =formula(nullModel), upper = formula(fullModel)),
                direction = 'forward')

formula(backward)
formula(forward)
# both the steps are giving the same formula with same AIC score ie. 1537.7

# We are rejecting fullModel due to multi-collinearity assumption of logistic regression.
# so selecting model_backward because its AIC score is lower than model2 and all the predictors
# in model_backward are stastically significant. 



#predicting using model_backward on test set
pred_backmodel <- predict(model_backward, test_dataset[,-18], type = 'response')
pred_backmodelT<- ifelse(pred_backmodel > 0.5 ,1,0)
xtab_backmodel <- table(observed = test_dataset[,18], predicted = pred_backmodelT)
backmodel_confmat <- confusionMatrix(xtab_backmodel)
backmodel_performance <- data.frame(backmodel_confmat$byClass)
backmodel_performance <- rbind(accuracy = backmodel_confmat$overall, backmodel_performance)
#comparing model_backward performance on validation and test set for overfitting 
z<- as.data.frame(rbind(backmodel_confmat$byClass,backward_confmat_val$byClass))
#model_backward is preforming with acceptable variation on both dataset, so no overfitting.


# ploting histogram of prediction
pred_hist <- data.frame(pred_backmodel)
pred_hist %>%
  ggplot(mapping = aes(x = pred_backmodel)) +
  geom_histogram(bins = 50, fill = 'grey40') +
  labs(title = " Prediction histograms")

# range of predictions
round(range(pred_backmodel),2) 
median(pred_backmodel)
#The prediction range from 0 to 0.97 with median 0.0872


# Selecting  probablity threshold value is business context decision and a 
# tradoff between  true positve and false positive classifications
# The threshold  here is set to .5 . This means that anyone with a probability of 
# more than .5 is predicted to churn. If we reduce the probability threshold, more people will 
# be predicted to churn, this gives us a higher number of “at risk customers” to target. 
# However, this increases the likelihood that customers who are not at risk will pass the 
# threshold and be predicted to churn.
# If we are concerened with marketing expenditure, then a higher threshold should be 
# targeted (above 0.8 or 0.9). Otherwise, lower thresholds can be targeted, 
# so the company can target larger amounts of customers who are at risk of churning.


#Logistic regression diagnostics - now we are going to check for logistic regression assumptions

#1. Linearity assumption
# Checking for a linear relationship between continous variables and logit of the outcome.This
#can be done by visually inspecting the scatter plot between each predictor and logit valuees.

# selecting continous predictors
continous_data <- test_dataset %>%
  select_if(is.numeric)
predictors <- colnames(continous_data)

# calculating logit and adding it in 'continous_data'
continous_data <- continous_data %>%
  mutate(logit = log(pred_backmodel/(1-pred_backmodel))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)
ggplot(data = continous_data, mapping = aes(x = logit, predictor.value)) +
  geom_point(size = 0.5, alpha = 0.5) + 
  geom_smooth(method = "loess") +
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

# The smoothed scatter plots show all the continous predictors are very near 
# linearly associated with the outcome in logit scale.

# 2. Multicollinearity 
# we have removed multi-collinear variables while models.
#performing double check using variation inflation factor (VIF)
car::vif(model_backward)
#As a rule of thumb, a VIF value that exceeds 5 or 10 indicates a problematic amount of 
#collinearity. In our model voice.mail.plan and number.vmail.messages is showing vif score of 
# < 16. SO, we will remove one of the variable and retrain and check the final model after 
#dignostics.

#3. Influential values
# top 3 largest values
plot(model_backward, which = 4, id.n = 3)
#278, 1862, 3292
# Data points with an absolute standardized residuals above 3 represent possible outliers 
#and may deserve closer attention
#augment - add columns to the original dataset such as predictions, residuals and cluster assignments

model.data <- augment(model_backward) %>%
  mutate(index = 1:n()) 
model.data %>% top_n(3, .cooksd)
# plotting standardized residuals
ggplot(model.data, aes(index, .std.resid)) + 
  geom_point(aes(color = Churn), alpha = .5) +
  theme_bw()
#
model.data %>% 
  filter(abs(.std.resid) > 3)
# one possible influential observations was found on our training set. 
# row 1890

#--------------------------------------------------------------------------------------
# creating a final model by removing one of the variables detected by VIF and removing 
# one influential observations

train_data <- train_data[-1890,]
formula(model_backward)

final_model <- glm(Churn ~ international.plan + voice.mail.plan +
                     total.day.minutes + total.eve.minutes + total.night.charge + 
                     total.intl.calls + total.intl.charge + number.customer.service.calls, 
                   data = train_data, family = binomial(link = 'logit'))

#checking VIF
car::vif(final_model)
# no multicollinear variable found in model

pred_final <- predict(final_model, test_dataset, type = 'response')
pred_finalT <- ifelse(pred_final>0.5, 1,0)
xtab_final <- table(observed = test_dataset[,18], predicted = pred_finalT)
final_confmat <- confusionMatrix(xtab_final)
finalModel_performance <- data.frame(final_confmat$byClass)
finalModel_performance <- rbind(accuracy = final_confmat$overall, finalModel_performance)



# After comparing confusion matrix of model_backward and final_modal, we found that 
# both models are performing very similar. 

#----------------------------------------------------------------------------------------

# for comparision with random forest performance with probabilty threshlod .75,.25

pred_compareRF <- predict(final_model, test_dataset[,-18], type = 'response')
pred_compareRFT <- ifelse(pred_compareRF>0.25, 1,0)

xtab_compare <- table(observed = test_dataset[,18], predicted = pred_compareRFT)
confMatLogit <- confusionMatrix(xtab_compare)
compare_performance <- data.frame(confMatLogit$byClass)
compare_performance <- rbind(accuracy = confMatLogit$overall, compare_performance)
print(confMatLogit)
print(compare_performance)


# Model input and output for logistic regression 
write.csv(test_dataset, file = "InputLogisticRegressionR.csv")
write.csv(pred_compareRFT, file="outputLogisticRegressionR.csv")
