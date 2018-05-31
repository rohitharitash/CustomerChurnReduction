# Randomforest and parameter tuning unis MLR package

library(mlr)
#library(caret)


trainTask <- makeClassifTask(data = train_data, target = "Churn")
validationTask <- makeClassifTask(data = validation_data, target = "Churn")

#creating randomoforest classifier/learner
randomFOrest.learner.baseline <- makeLearner("classif.randomForest")
randomFOrest.learner <- makeLearner("classif.randomForest")


# setting validaion staertgy using cv with 10 folds
cvstr <- makeResampleDesc("CV", iters = 10L)

# Our main aim is churn reduction. So we are interest in a model which can classify customers
# who are going to churn out. There for model should reduce false positive rate. As positve class
#for the model is 'false'


randomFOrest.learner$par.vals <-
  list(ntree = 500L,
       importance = TRUE,
       cutoff = c(0.70, 0.30))

r <- resample(
  learner = randomFOrest.learner,
  task = trainTask,
  resampling = cvstr,
  measures = list(tpr, fpr, fnr, fpr, acc),
  show.info = TRUE
)

# Probablity threshold or cutoff is a bussiness context decision. The threshold is usually set
#to .5 by default. This means that anyone with a probability of more than .5 is predicted to churn. 
#If we reduce the probability threshold, more people will be predicted to churn
# to reduce the false positve rate , we are tuning cut-off here.
# cut-off  tuning is perfomed by cross-validation
# Tuning cutoff value starting with default .50,.50
#.70 and .30 is selected as cutoff value
# Main purpose here is tuning cutoff.
cutoff1 <- calculateConfusionMatrix(r$pred)

# tuning mtry ,ntree, nodesize using hypertuning
params <- makeParamSet(
  makeIntegerParam("mtry", lower = 2, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50),
  makeIntegerParam("ntree", lower = 100, upper = 600)
)

#random search with 100 iterations
ctrl <- makeTuneControlRandom(maxit = 100L)

tuneRF <- tuneParams(
  learner = randomFOrest.learner,
  task = trainTask,
  resampling = cvstr,
  measures = list(acc),
  par.set = params,
  control = ctrl,
  show.info = TRUE
)

tunedRFmodel <- setHyperPars(learner = randomFOrest.learner,
                             par.vals = tuneRF$x)

# tuning with default random forest
modelRFBaseline <- mlr::train(randomFOrest.learner.baseline, trainTask)

# training random forest with tuned parameters
modelRF1 <- mlr::train(tunedRFmodel, trainTask)

#prediction on validation set using basline
predictBaseline <- predict(modelRFBaseline, validationTask)

#performance measures for baseline model on validation set
accuracy_bl <- performance(pred = predictBaseline, measures = acc)
f1Measure_bl <- performance(pred = predictBaseline, measures = f1)
truePositveRate_bl <-
  performance(pred = predictBaseline, measures = tpr)
trueNegativeRate_bl <-
  performance(pre = predictBaseline, measures = tnr)
falsePositiveRate_bl <-
  performance(pred = predictBaseline, measures = fpr)
falseNegativeRate_bl <-
  performance(pred = predictBaseline, measures = fnr)

baselinePerformanceMetric <-
  c(
    accuracy_bl,
    f1Measure_bl,
    truePositveRate_bl,
    trueNegativeRate_bl,
    falsePositiveRate_bl,
    falseNegativeRate_bl
  )



#prediction on validation set using tuned random forest
predictRF1 <- predict(modelRF1, validationTask)

#performance measures for tuned model on validation set
accuracy_val <- performance(pred = predictRF1, measures = acc)
f1Measure_val <- performance(pred = predictRF1, measures = f1)
truePositveRate_val <-
  performance(pred = predictRF1, measures = tpr)
trueNegativeRate_val <-
  performance(pre = predictRF1, measures = tnr)
falsePositiveRate_val <-
  performance(pred = predictRF1, measures = fpr)
falseNegativeRate_val <-
  performance(pred = predictRF1, measures = fnr)


validationPerformanceMetric <-
  c(
    accuracy_val,
    f1Measure_val,
    truePositveRate_val,
    trueNegativeRate_val,
    falsePositiveRate_val,
    falseNegativeRate_val
  )







#prediction on  main testset using tuned RF


mainTestTask <- makeClassifTask(data = test_dataset, target = "Churn")

predictTest <- predict(modelRF1, mainTestTask)
confusionMatrixTest <- calculateConfusionMatrix(predictTest)

#calculating performance measure for predictions
accuracy <- performance(pred = predictTest, measures = acc)
f1Measure <- performance(pred = predictTest, measures = f1)
truePositveRate <- performance(pred = predictTest, measures = tpr)
trueNegativeRate <- performance(pre = predictTest, measures = tnr)
falsePositiveRate <- performance(pred = predictTest, measures = fpr)
falseNegativeRate <- performance(pred = predictTest, measures = fnr)

testPerformanceMetric = c(
  accuracy,
  f1Measure,
  truePositveRate,
  trueNegativeRate,
  falsePositiveRate,
  falseNegativeRate
  )



xt <- table(observed = test_dataset[,18], predicted = predictTest$data$response)
final_confmatRF <- confusionMatrix(xt)
compare_performanceRF <- data.frame(final_confmatRF$byClass)
compare_performanceRF <- rbind(accuracy = final_confmatRF$overall, compare_performanceRF)
print(final_confmatRF)
print(compare_performanceRF)

# Model input and output for Random Forest
write.csv(test_dataset, file = "InputRandomForestR.csv")
write.csv(predictTest$data$response, file="outputRandomForestR.csv")
