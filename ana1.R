library(gbm)
library(caret)

library(lubridate)
library(plyr)
library(dplyr)
library(doParallel)
#library(data.table)

# load in the data sets and take a look at them

setClass("quoted.numeric")
setAs("character", "quoted.numeric", function(from) as.numeric(gsub('"','',from)))
df <- read.csv("pml-training.csv",colClasses=c(
  "character",
  "factor",
  rep("character", 4),
  "integer",
  rep("quoted.numeric", 159-8 + 1),
  "factor"
))

df_submit <- read.csv("pml-testing.csv")
dmy_hm(df$cvtd_timestamp) -> df$time
dmy_hm(df_submit$cvtd_timestamp) -> df_submit$time

# we found that the time of the observation and the window related parameters are not relevant.
select(df, user_name, time, num_window, classe) %>% group_by(classe) %>% 
  summarize(median(num_window), sd(num_window))


qplot(df$time, df$num_window, color=df$classe, 
      main="time appears to be an experiment artifact")
qplot(df$num_window, fill=df$classe, main="num_window appears distributed across classes")
qplot(df$num_window, fill=df$user_name, main="num_window appears to be an artifact associated with individual users")
qplot(num_window, classe, color=user_name, data=df, main="num_window appears related to the combination of user_name and activity classe")

select(df_submit, user_name, time, new_window, num_window) %>% arrange(user_name) 

# these columns have no data at all in the submit set
bad_cols <- which(apply(is.na(df_submit), 2, sum)/nrow(df_submit) == 1)

# we can see the same bad cols in the training data have very higher proportion of NA's, >0.97
# they have too little meaningful data
apply(is.na(df[,bad_cols]), 2, sum)/nrow(df)

# for curiosity, we look at the correlation of the excluded columns, seeing they're minimal or 
# unreliable
cor(df[,bad_cols], as.numeric(df[,"classe"]), use="pairwise.complete.obs")
qplot(df[,"stddev_roll_belt"], df[,"classe"])

# use 80% for training set, 20% for testing set. train() will cross-validate
# within the training set. Must set.seed() immediately before calling createDataPartition()
# in order to set aside the same testing set every time.
set.seed(1234)
inTrain <- createDataPartition(y=df$classe, p=0.80,list=FALSE)
training <- df[inTrain, -bad_cols]
# testing set will be used ONE TIME after model selection and parameter
# tuning are completed.
testing <- df[-inTrain, -bad_cols]

#the sensor columns
sensor_cols <- c(8:59)

# look at the top 2 principal components colored by activity class
typecolor <- rainbow(5)[as.integer(training$classe)]

M <- cor(training[,sensor_cols])
diag(M) <- 0
which(abs(M) > 0.9, arr.ind=TRUE)

plot(training[,sensor_cols[c(1,4,9,10)]], col=typecolor)

prepro_obj <- preProcess(training[,sensor_cols], method=c("center","scale"))
processed <- predict(prepro_obj, training[,sensor_cols])
plot(prcomp(processed)$sd)
prcomp(training[,sensor_cols[c(1,4)]]) -> pcomp
plot(pcomp$x[,c(1,2)], col=typecolor)

# now normalize the covariates

prepro_obj <- preProcess(training[,sensor_cols], method=c("pca"), thresh=0.95)
processed <- predict(prepro_obj, training[,sensor_cols])
summary(processed)

par(mfrow=c(1,2))
qqnorm(training[,10])
qqnorm(processed[,10-7])

restartClusters <- function() {
  if (exists("cl",mode="list")) try(stopCluster(cl))
  # leaving one core out improves system responsiveness
  cl <<- makeCluster(detectCores()-1)
  registerDoParallel(cl)
}

###### training with different options, and put the results in a matrix
models <- NULL

restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      method="rf"))
models <- list(preprocess=F, includes_user_name=F, method="rf", cv="bootstrap", 
                     model=mod, time=proc_time)

##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,c(2,sensor_cols)], training[,"classe"], 
                                      method="rf"))
models <- rbind(models, list(preprocess=F, includes_user_name=T, 
                             method="rf", cv="bootstrap", 
                             model=mod, time=proc_time))

##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,c(2,7,sensor_cols)], training[,"classe"], 
                                      method="rf"))
models <- rbind(models, list(preprocess=F, includes_user_name=T, 
                             method="rf", cv="bootstrap", 
                             model=mod, time=proc_time, includes_num_window=T))
##############################################################################
cvControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      method="rf", trControl = cvControl))
models <- rbind(models, list(preprocess=F, includes_user_name=F, 
                             method="rf", cv="10-fold", 
                             model=mod, time=proc_time))


##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      method="gbm"))
models <- rbind(models, list(preprocess=F, includes_user_name=F, 
                             method="gbm", cv="bootstrap", 
                             model=mod, time=proc_time))

##############################################################################
### this one contains a memory leak!
cvControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 1)
restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      method="gbm", trControl = cvControl))
models <- rbind(models, list(preprocess=F, includes_user_name=F, 
                             method="gbm", cv="10-fold", 
                             model=mod, time=proc_time))

##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,c(2,sensor_cols)], training[,"classe"], 
                                      method="gbm", trControl = cvControl))
models <- rbind(models, list(preprocess=F, includes_user_name=T, 
                             method="gbm", cv="10-fold", 
                             model=mod, time=proc_time))

##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      preProcess=c("center","scale"),
                                      method="gbm", trControl = cvControl))
models <- rbind(models, list(preprocess="center,scale", includes_user_name=F, 
                             method="gbm", cv="10-fold", 
                             model=mod, time=proc_time))

##############################################################################
restartClusters()
proc_time <- system.time(mod <- train(training[,sensor_cols], training[,"classe"], 
                                      preProcess=c("pca"),
                                      method="gbm", trControl = cvControl))
models <- rbind(models, list(preprocess="pca", includes_user_name=F, 
                             method="gbm", cv="10-fold", 
                             model=mod, time=proc_time))

##############################################################################
tail(models[,"model"]$results,1)[c("Accuracy","AccuracySD")]
sapply(models[,"model"], function(x) x[["results"]])

stopCluster(cl)

pred1 <- predict(model1, testing[,sensor_cols])
table(testing[,"classe"], pred1)
sum(diag(pred1))

# with preprocessing
system.time(model2 <- train(processed, training[,"classe"]))
pred2 <- predict(model2, predict(prepro_obj, testing[,sensor_cols]))
table(testing[,"classe"], pred2)
sum(diag(pred2))
df_submit[,-bad_cols][,c(2,7,sensor_cols)]

boxplot(list(
  c(1,2,3),
  c(2,2.5,3)
  ),
  names=c("random forrest", "gbm"),
  horizontal=TRUE)
