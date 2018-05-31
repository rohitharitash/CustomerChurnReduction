# preparing train,validation and test dataset for random forest and logistic regression

library(caret)
library(stringr)

# reading train set
inputdata <- read.csv("Train_data.csv", sep = ',', header = TRUE, na.strings = c(" ","NA"))
inputTest <- read.csv("Test_data.csv", sep = ',', header = TRUE, na.strings = c(" ","NA"))

#removing variables which not helpfull in churn reduction. Using these variables dosn't make sense.
#for training data
inputdata$state <- NULL
inputdata$area.code <- NULL
inputdata$phone.number <- NULL

#for test data
inputTest$state <- NULL
inputTest$area.code <- NULL
inputTest$phone.number <- NULL

# We are going to implement random forest and logistic regresion and compare performance of 
# both models. For comparision we need same training and test data.
# random forest can use input both character and numeric data but logistic regression can only 
# work on numeric data. 
# Transforming and re-encoding the data so that both random forest and logistic regression can train and predict 
# on same dataset. 

#for training data
inputdata[,'international.plan'] <- ifelse(str_detect(inputdata[,'international.plan'],"yes"),1,0)
inputdata$international.plan <- as.factor(inputdata$international.plan)
inputdata[,'voice.mail.plan'] <- ifelse(str_detect(inputdata[,'voice.mail.plan'],"yes"),1,0)
inputdata$voice.mail.plan <- as.factor(inputdata$voice.mail.plan)
inputdata[,"Churn"] <- str_replace(inputdata[,"Churn"],"\\.","")
inputdata[,"Churn"] <- ifelse (str_detect(inputdata[,"Churn"],"True"),1,0) 
inputdata$Churn <- as.factor(inputdata$Churn)

#for test data
inputTest[,'international.plan'] <- ifelse(str_detect(inputTest[,'international.plan'],"yes"),1,0)
inputTest$international.plan <- as.factor(inputTest$international.plan)
inputTest[,'voice.mail.plan'] <- ifelse(str_detect(inputTest[,'voice.mail.plan'],"yes"),1,0)
inputTest$voice.mail.plan <- as.factor(inputTest$voice.mail.plan)
inputTest[,"Churn"] <- str_replace(inputTest[,"Churn"],"\\.","")
inputTest[,"Churn"] <- ifelse (str_detect(inputTest[,"Churn"],"True"),1,0) 
inputTest$Churn <- as.factor(inputTest$Churn)

set.seed(987)
# spliting input train data into train and validation set
churnIndex <- createDataPartition(inputdata$Churn, p = 0.7, times = 1, list = FALSE)
train_data <- inputdata[churnIndex,]
validation_data <- inputdata[churnIndex,]

test_dataset <- inputTest

