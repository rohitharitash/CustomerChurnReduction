# Customer churn reduction EDA

rm(list=ls())
#loading requried libraries

library(dplyr)
library(ggplot2)
library(stringr)
library(corrplot)


fillColor = "#FFA07A"
fillColorRed = "#56B4E9"

train_data <-
  read.csv("Train_data.csv",
           sep = ',',
           na.strings = c(" ", "NA"))

#looking at dimensions
dim(train_data)

#Train data set consist of 3333 observations and 21 varaiables

#checking structure of dataset
str(train_data)

# Visualizing target class freqquencies
train_data %>%
  count(Churn) %>%
  ggplot(aes(x = Churn,
             y = n)) +
  geom_bar(stat = 'identity',
           colour = "white",
           fill = fillColor) +
  labs(x = 'Churn rate', y = 'count ', title = 'Customer churn count') +
  theme_bw()

table(train_data$Churn)
#Looking at the frequencies of churn , it is not looking like highly imbalance problem.

# Now looking for any missing values
sapply(train_data, function(x) {
  sum(is.na(x))
}) # There are no missing values in dataset



#selecting numeric variables
numCols <- unlist(lapply(train_data,is.numeric))
numVarDataset <- train_data[,numCols]

# Visualizing correlation 
par(mfrow = c(1, 1))
corr <- cor(numVarDataset)
corrplot(
  corr,
  method = "color",
  outline = TRUE,
  cl.pos = 'n',
  rect.col = "black",
  tl.col = "indianred4",
  addCoef.col = "black",
  number.digits = 2,
  number.cex = 0.60,
  tl.cex = 0.70,
  cl.cex = 1,
  col = colorRampPalette(c("green4", "white", "red"))(100)
)

# From corrplot we can see that dataset consist of multicollinearity
# total.day.minutes and total.day.charge are highly collinear
# total.eve.minutes and total.eve.charge are highly collinear
# total.night.minutes and total.night.charge are highly collinear
# total.intl.minutes and total.intl are highly collinear
# we can exclude one of these predictors later during modeling

############## Generic EDA function for continous variables
plot_continous <- function(dataset, variable,targetVariable) {
  var_name = eval(substitute(variable), eval(dataset))
  target_var = eval(substitute(targetVariable), eval(dataset))
  par(mfrow = c(1, 2))
  print(summary(var_name))
  print(summary(target_var))
  possible_outliers <- (boxplot.stats(var_name)$out)
  print(possible_outliers)
  print(paste("Total possible outliers", length(possible_outliers)))
  table(possible_outliers)
  ggplot(train_data, aes(target_var, var_name, fill = target_var)) + 
    geom_boxplot(alpha = 0.8) + theme(legend.position = "null")
}


############################### looking at 'state' variable. It is a factor variable
train_data %>%
  count(state) %>%
  ggplot(mapping = aes(x = state, y = n)) +
  geom_bar(stat = 'identity',
           colour = 'white',
           fill = fillColor) +
  labs(x = "states", y = "count", "Customers per state") +
  coord_flip()
# Fom the plot we can that maximum customers are from west vergenia and lowest are from California




# looking at each variable
plot_continous(train_data, account.length,Churn)
# As we can see, that there are some possible outliers but they are not very extreme. Ignoring them


##################################### analysing international.plan #################################
str(train_data$international.plan) # it is a categorical variable
table(train_data$international.plan)
train_data %>%
  count(international.plan) %>%
  ggplot(mapping = aes(x = international.plan, y = n)) +
  geom_bar(stat = 'identity',
           colour = 'white',
           fill = fillColor)
# From the plot we can see that most customers dont have international plan.

# next examining for the churn rate percentage of customers with national and internation plan

national_cust_churnRate <- train_data %>%
  select(international.plan, Churn) %>%
  filter(str_detect(international.plan, "no")) %>%
  group_by(Churn) %>%
  summarise (n = n()) %>%
  mutate(percantage = (n / sum(n)) * 100)
#Only 11.49 % customer with national plan churn out.


international_cust_churnRate <- train_data %>%
  select(international.plan, Churn) %>%
  filter(str_detect(international.plan, "yes")) %>%
  group_by(Churn) %>%
  summarise (n = n()) %>%
  mutate(percantage = (n / sum(n)) * 100)
# 42.42 %  customers with international plan had churn out. It means that the telecom company
# is mainly loosing customers with internation plans.

#####################################Analysing voice.mail.plan ####################################
table(train_data$voice.mail.plan)

# customers with voice plan and their churn rate
voice_plan_churnRate <- train_data %>%
  select(voice.mail.plan, Churn) %>%
  filter(str_detect(voice.mail.plan, "yes")) %>%
  group_by(Churn) %>%
  summarise(n = n()) %>%
  mutate(churnRatePercentage = (n / sum(n)) * 100)

ggplot(data = voice_plan_churnRate,
       mapping = aes(x = Churn, y = churnRatePercentage)) +
  geom_bar(stat = 'identity',
           colour = 'white',
           fill = fillColorRed) +
  labs(title = 'Voice main plan customers churn rate')

# 922 customers have voice mail plan and 80 (8.68 %) customers out of 922  churn out.

#customers without voice plan and their churn rate
non_voice_plan_churnRate <- train_data %>%
  select(voice.mail.plan, Churn) %>%
  filter(str_detect(voice.mail.plan, "no")) %>%
  group_by(Churn) %>%
  summarise(n = n()) %>%
  mutate(churnRatePercentage = (n / sum(n)) * 100)

ggplot(data = non_voice_plan_churnRate,
       mapping = aes(x = Churn, y = churnRatePercentage)) +
  geom_bar(stat = 'identity',
           colour = 'white',
           fill = fillColor) +
  labs(title = 'Non voice plan Customer churn rate')

# 2411 customers dont have voice mail plan  and 403 (16.7 %) out of 2411 churn out

#So customers without voice plan have higher churn rate



# removing parameters that dosn't seem to be logical parameter for customer churn. 
#So removing state, area code and phone number
train_data$state <- NULL
train_data$area.code <- NULL
train_data$phone.number <- NULL


############################ Analysing number.vmail.messages ################################
str(train_data$number.vmail.messages)
plot_continous(train_data, number.vmail.messages,Churn)

# no extreme outliers detected.

############################ Analysing total.day.minutes ################################
str(train_data$total.day.minutes)
plot_continous(train_data, total.day.minutes,Churn)

# no extreme outliers detected

############################ Analysing total.day.calls ################################
str(train_data$total.day.calls)
plot_continous(train_data, total.day.calls,Churn)

# no extreme outliers detected

############################ Analysing total.day.charge ################################
str(train_data$total.day.charge)
plot_continous(train_data, total.day.charge, Churn)

# no extreme outliers detected

############################ Analysing total.eve.minutes ################################
str(train_data$total.eve.minutes)
plot_continous(train_data, total.eve.minutes, Churn)

# no extreme outliers detected

############################ Analysing total.eve.calls ################################
str(train_data$total.eve.calls)
plot_continous(train_data, total.eve.calls, Churn)

# no extreme outliers detected

############################ Analysing total.eve.charge ################################
str(train_data$total.eve.charge)
plot_continous(train_data, total.eve.charge, Churn)

# no extreme outliers detected

############################ Analysing total.night.minutes ################################
str(train_data$total.night.minutes)
plot_continous(train_data, total.night.minutes, Churn)

# no extreme outliers detected

############################ Analysing total.night.calls ################################
str(train_data$total.night.calls)
plot_continous(train_data, total.night.calls, Churn)

# no extreme outliers detected

############################ Analysing total.night.charge ################################
str(train_data$total.night.charge)
plot_continous(train_data, total.night.charge, Churn)

# no extreme outliers detected

############################ Analysing total.intl.minutes ################################
str(train_data$total.intl.minutes)
plot_continous(train_data, total.intl.minutes, Churn)

# no extreme outliers detected

############################ Analysing total.intl.calls ################################
str(train_data$total.intl.calls)
plot_continous(train_data, total.intl.calls, Churn)

# no extreme outliers detected

############################ Analysing total.intl.charge ################################
str(train_data$total.intl.charge)
plot_continous(train_data, total.intl.charge, Churn)

# no extreme outliers detected

######################## Analysing number.customer.service.calls #############################
str(train_data$number.customer.service.calls)

plot_continous(train_data , number.customer.service.calls, Churn)
table(train_data$number.customer.service.calls)



