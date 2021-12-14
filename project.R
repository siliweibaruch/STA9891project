library(tidyverse)
library(modelr)
library(glmnet)
library(glmnetUtils)
library(readr)
library(ISLR)
library(randomForest)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(grid)
library(data.table)
library(wesanderson)
library(curl)
library(caret)
library(pROC)
library(imbalance)
# read raw data from github
data = read.csv("https://raw.githubusercontent.com/siliweibaruch/STA9891project/main/default_plus_chromatic_features_1059_tracks.txt",header=F)

# Data preparation
# Make column names lowercase for easy input, change the dependent variable name to 'latitude'
names(data) = tolower(names(data))
colnames(data)[colnames(data) == 'v117'] = 'latitude'

# filter our those rows with missing values
data = data %>% filter(!is.na(latitude))

# classify countries in northern hemisphere as 1, in southern hemisphere as 0
for (i in 1:length(data$latitude)){
  if (data$latitude[i] > 0){
      data$latitude[i] = 1
  }
  else{
    data$latitude[i] = 0
  }
}


# count the imbalance ratio n+/n-
table(data$latitude)
imbalanceRatio(data,classAttr="latitude")

# define the response and predictors 
y = data$latitude
X = data %>% select(-latitude & -v118)
X = data.matrix(X)

# count the sample size and features
n = nrow(data)
p = ncol(X)
n
p

# define n.train and n.test, the train set use 90% of the data
n.train = floor(0.9*n)
n.test = n - n.train

# Loop times set to 50, create empty vectors for loop use
M = 50
auc.train.lasso = rep(0,M)
auc.test.lasso = rep(0,M)
auc.train.en = rep(0,M)
auc.test.en = rep(0,M)
auc.train.ridge = rep(0,M)
auc.test.ridge = rep(0,M)
auc.train.rf = rep(0,M)
auc.test.rf = rep(0,M)

# Q2 lasso
for (m in c(1:M)) {
  
  # divide both response and predictor matrix into test and train sets randomly
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # fit the lasso logistic regression and get the coefficients 
  fit = cv.glmnet(X.train, y.train, family = "binomial", alpha = 1, type.measure = "auc")
  fit = glmnet(X.train, y.train, family = "binomial", alpha = 1, lambda = fit$lambda.min )
  beta0.hat = fit$a0
  beta.hat = as.vector(fit$beta)
  prob.train = exp(X.train %*% beta.hat + beta0.hat)/(1 + exp(X.train %*% beta.hat + beta0.hat))
  prob.test = exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  dt = 0.01
  thta = 1-seq(0,1, by=dt)
  thta.length = length(thta)
  FPR.train = matrix(0, thta.length)
  TPR.train = matrix(0, thta.length)
  FPR.test = matrix(0, thta.length)
  TPR.test = matrix(0, thta.length)
    
  for (i in c(1:thta.length)){
    
  # calculate the FPR and TPR for train data 
  y.hat.train = ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train = sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the train data that were predicted as positive
  TP.train = sum(y.hat.train[y.train==1] == 1) # true positives = positives in the train data that were predicted as positive
  P.train = sum(y.train==1) # total positives in the data
  N.train = sum(y.train==0) # total negatives in the data
  FPR.train[i] = FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i] = TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
      
  # calculate the FPR and TPR for test data 
  y.hat.test = ifelse(prob.test > thta[i], 1, 0)
  FP.test = sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the test data that were predicted as positive
  TP.test = sum(y.hat.test[y.test==1] == 1) # true positives = positives in the test data that were predicted as positive
  P.test = sum(y.test==1) # total positives in the data
  N.test = sum(y.test==0) # total negatives in the data
  FPR.test[i] = FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i] = TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    

  }
  
  # calculate the AUC of lasso logistic regression for 50 loops
  auc.train.lasso[m] = sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test.lasso[m] = sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  print(paste("train AUC =",sprintf("%.2f", auc.train.lasso[m])))
  print(paste("test AUC  =",sprintf("%.2f", auc.test.lasso[m])))
  
}


# Q2 en
for (m in c(1:M)) {
  
  # divide both response and predictor matrix into test and train sets randomly
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # fit the elastic-net logistic regression and get the coefficients
  fit = cv.glmnet(X.train, y.train, family = "binomial", alpha = 0.5, type.measure = "auc")
  fit = glmnet(X.train, y.train, family = "binomial", alpha = 0.5, lambda = fit$lambda.min )
  beta0.hat = fit$a0
  beta.hat  = as.vector(fit$beta)
  prob.train = exp(X.train %*% beta.hat + beta0.hat)/(1 + exp(X.train %*% beta.hat + beta0.hat))
  prob.test = exp(X.test %*% beta.hat + beta0.hat)/(1 + exp(X.test %*% beta.hat + beta0.hat))
  dt = 0.01
  thta = 1-seq(0,1, by=dt)
  thta.length = length(thta)
  FPR.train = matrix(0, thta.length)
  TPR.train = matrix(0, thta.length)
  FPR.test = matrix(0, thta.length)
  TPR.test = matrix(0, thta.length)
 
  for (i in c(1:thta.length)){
    
  # calculate the FPR and TPR for train data 
  y.hat.train = ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train = sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the train data that were predicted as positive
  TP.train = sum(y.hat.train[y.train==1] == 1) # true positives = positives in the train data that were predicted as positive
  P.train = sum(y.train==1) # total positives in the data
  N.train = sum(y.train==0) # total negatives in the data
  FPR.train[i] = FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i] = TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
  # calculate the FPR and TPR for test data 
  y.hat.test = ifelse(prob.test > thta[i], 1, 0)
  FP.test = sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the test data that were predicted as positive
  TP.test = sum(y.hat.test[y.test==1] == 1) # true positives = positives in the test data that were predicted as positive
  P.test = sum(y.test==1) # total positives in the data
  N.test = sum(y.test==0) # total negatives in the data
  FPR.test[i] = FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i] = TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    

  }

  # calculate the AUC of elastic-net logistic regression for 50 loops
  auc.train.en[m] = sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test.en[m] = sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  print(paste("train AUC =",sprintf("%.2f", auc.train.en[m])))
  print(paste("test AUC  =",sprintf("%.2f", auc.test.en[m])))
  
}

# Q2 ridge
for (m in c(1:M)) {
  
  # divide both response and predictor matrix into test and train sets randomly
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # fit the ridge logistic regression and get the coefficients
  fit = cv.glmnet(X.train, y.train, family = "binomial", alpha = 0, type.measure = "auc")
  fit = glmnet(X.train, y.train, family = "binomial", alpha = 0, lambda = fit$lambda.min )
  beta0.hat = fit$a0
  beta.hat = as.vector(fit$beta)
  prob.train = exp(X.train %*% beta.hat + beta0.hat)/(1 + exp(X.train %*% beta.hat + beta0.hat))
  prob.test = exp(X.test %*% beta.hat + beta0.hat)/(1 + exp(X.test %*% beta.hat + beta0.hat))
  dt = 0.01
  thta = 1-seq(0,1, by=dt)
  thta.length = length(thta)
  FPR.train = matrix(0, thta.length)
  TPR.train = matrix(0, thta.length)
  FPR.test =  matrix(0, thta.length)
  TPR.test =  matrix(0, thta.length)
  
  for (i in c(1:thta.length)){
    
  # calculate the FPR and TPR for train data 
  y.hat.train = ifelse(prob.train > thta[i], 1, 0) #table(y.hat.train, y.train)
  FP.train = sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the train data that were predicted as positive
  TP.train = sum(y.hat.train[y.train==1] == 1) # true positives = positives in the train data that were predicted as positive
  P.train = sum(y.train==1) # total positives in the data
  N.train = sum(y.train==0) # total negatives in the data
  FPR.train[i] = FP.train/N.train # false positive rate = type 1 error = 1 - specificity
  TPR.train[i] = TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
  
  # calculate the FPR and TPR for test data 
  y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
  FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the test data that were predicted as positive
  TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the test data that were predicted as positive
  P.test                  =        sum(y.test==1) # total positives in the data
  N.test                  =        sum(y.test==0) # total negatives in the data
  FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
  TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    

  }

  # calculate the AUC of ridge logistic regression for 50 loops
  auc.train.ridge[m] = sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test.ridge[m] = sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  print(paste("train AUC =",sprintf("%.2f", auc.train.ridge[m])))
  print(paste("test AUC  =",sprintf("%.2f", auc.test.ridge[m])))
  
}

# Q2 rf
for (m in c(1:M)) {
  
  # divide both response and predictor matrix into test and train sets randomly
  shuffled_indexes = sample(n)
  train = shuffled_indexes[1:n.train]
  test = shuffled_indexes[(1+n.train):n]
  X.train = X[train, ]
  y.train = y[train]
  X.test = X[test, ]
  y.test = y[test]
  
  # fit the random forest
  rf.fit = randomForest(X.train,y.train, mtry = sqrt(p),importance=T)
  y.test.hat = predict(rf.fit, X.test)
  y.train.hat = predict(rf.fit, X.train)
  dt = 0.01
  thta = 1-seq(0,1, by=dt)
  thta.length = length(thta)
  FPR.train = matrix(0, thta.length)
  TPR.train = matrix(0, thta.length)
  FPR.test = matrix(0, thta.length)
  TPR.test = matrix(0, thta.length)
  
  for (i in c(1:thta.length)){
    
    # calculate the FPR and TPR for train data 
    y.hat.train = ifelse(y.train.hat > thta[i], 1, 0) #table(y.hat.train, y.train)
    FP.train = sum(y.train[y.hat.train==1] == 0) # false positives = negatives in the train data that were predicted as positive
    TP.train = sum(y.hat.train[y.train==1] == 1) # true positives = positives in the train data that were predicted as positive
    P.train = sum(y.train==1) # total positives in the data
    N.train = sum(y.train==0) # total negatives in the data
    FPR.train[i] = FP.train/N.train # false positive rate = type 1 error = 1 - specificity
    TPR.train[i] = TP.train/P.train # true positive rate = 1 - type 2 error = sensitivity
    
    # calculate the FPR and TPR for test data 
    y.hat.test = ifelse(y.test.hat > thta[i], 1, 0)
    FP.test = sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test = sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test =  sum(y.test==1) # total positives in the data
    N.test =  sum(y.test==0) # total negatives in the data
    FPR.test[i] = FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i] = TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity    

  }

  # calculate the AUC of random forest for 50 loops
  auc.train.rf[m] = sum((TPR.train[1:(thta.length-1)] + 0.5 * diff(TPR.train)) * diff(FPR.train))
  auc.test.rf[m] = sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  print(paste("train AUC =",sprintf("%.2f", auc.train.rf[m])))
  print(paste("test AUC  =",sprintf("%.2f", auc.test.rf[m])))
}

# boxplot of 50 AUCs

par(mfrow = c(1,2))
boxplot(auc.train.lasso,auc.train.en,auc.train.ridge,auc.train.rf,
        main = "TRAIN SET AUC",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))

boxplot(auc.test.lasso,auc.test.en,auc.test.ridge,auc.test.rf,
        main = "TEST SET AUC",
        names = c("LASSO", "EN", "RIDGE", "RF"),
        col = c("red","blue", "yellow", "green"))


# 10-fold cv & time
# 10-fold cross-validation for lasso
par(mfrow = c(1,1))
ptm = proc.time()
lasso.cv = cv.glmnet(X.train, y.train, family = 'binomial', alpha = 1, nfolds = 10, type.measure = "class")
lasso = glmnet(X.train, y.train, lambda = lasso.cv$lambda.min, alpha = 1, family = "binomial")
ptm = proc.time() - ptm
time_lasso = ptm["elapsed"]
cat(sprintf("Run Time for Lasso: %0.3f(sec):",time_lasso))
plot(lasso.cv) + title("10-fold CV curve for Lasso", line = 2.5)
lasso$df

# 10-fold cross-validation for elastic-net
ptm = proc.time()
en.cv = cv.glmnet(X.train, y.train, family = 'binomial', alpha = 0.5, nfolds = 10, type.measure = "class")
en = glmnet(X.train, y.train, lambda = en.cv$lambda.min, alpha = 0.5, family = "binomial")
ptm = proc.time() - ptm
time_en = ptm["elapsed"]
cat(sprintf("Run Time for Elastic-net: %0.3f(sec):",time_en))
plot(en.cv) + title("10-fold CV curve for Elastic-net", line = 2.5)
en$df

# 10-fold cross-validation for ridge
ptm = proc.time()
ridge.cv = cv.glmnet(X.train, y.train, alpha = 0, family = 'binomial', nfolds = 10, type.measure = "class")
ridge = glmnet(X.train, y.train, lambda = ridge.cv$lambda.min, alpha = 0, family = "binomial")
ptm = proc.time() - ptm
time_ridge = ptm["elapsed"]
cat(sprintf("Run Time for ridge: %0.3f(sec):",time_ridge))
plot(ridge.cv) + title("10-fold CV curve for Ridge", line = 2.5)
ridge$df

# Q4
# record the time of logistic regression including CV parameter tuning 
# lasso
la_start = Sys.time()
cv.la = cv.glmnet(X, y, family = 'binomial',alpha = 1, nfolds = 10)
la = glmnet(X, y, alpha = 1, lambda = cv.la$lambda.min)
la_end = Sys.time()
la_time = la_end - la_start
median.lasso = median(auc.test.lasso)

# elastic-net
en_start = Sys.time()
cv.en = cv.glmnet(X, y, family = 'binomial',alpha = 0.5, nfolds = 10)
en = glmnet(X, y, alpha = 0.5, lambda = cv.en$lambda.min)
en_end = Sys.time()
en_time = en_end - en_start
median.en = median(auc.test.en)

# ridge
ri_start = Sys.time()
cv.ri = cv.glmnet(X, y, family = 'binomial',alpha = 0, nfolds = 10)
ri = glmnet(X, y, alpha = 0, lambda = cv.ri$lambda.min)
ri_end = Sys.time()
ri_time = ri_end - ri_start
median.ridge = median(auc.test.ridge)

# random forest
rf_start = Sys.time()
rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)
rf_end = Sys.time()
rf_time = rf_end - rf_start
median.rf = median(auc.test.rf)

# record the median of test AUCs
comparison = data.frame(c(la_time,en_time,ri_time,rf_time),c(median.lasso,median.en,median.ridge,median.rf))
colnames(comparison) = c("time","median AUC")
rownames(comparison) = c("lasso","en","ridge","rf")
comparison


# bar-plots of the estimated coefficients

# random forest
fit.rf = randomForest(X, y, mtry = sqrt(p), importance = TRUE)

betaS.en = data.frame(c(1:p), as.vector(en$beta))
colnames(betaS.en) = c( "feature", "value")

betaS.ls = data.frame(c(1:p), as.vector(lasso$beta))
colnames(betaS.ls) = c( "feature", "value")

betaS.ri = data.frame(c(1:p), as.vector(ridge$beta))
colnames(betaS.ri) = c( "feature", "value")

betaS.rf = data.frame(c(1:p), as.vector(fit.rf$importance[,1]))
colnames(betaS.rf) = c( "feature", "importance")

lsPlot = ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

enPlot = ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

riPlot = ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")

rfPlot = ggplot(betaS.rf, aes(x=feature, y=importance)) +
  geom_bar(stat = "identity", fill="white", colour="black")

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)

# we need to change the order of factor levels by specifying the order explicitly.
betaS.ls$feature = factor(betaS.ls$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.en$feature = factor(betaS.en$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.ri$feature = factor(betaS.ri$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.rf$feature = factor(betaS.rf$feature, levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
arrange(betaS.ls,by = value)
arrange(betaS.en,by=value)

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("lasso estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("elastic_net estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

riPlot =  ggplot(betaS.ri, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("ridge estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

rfPlot =  ggplot(betaS.rf, aes(x = feature, y = importance)) +
  geom_bar(stat = "identity", fill="white", colour="black") +
  ggtitle("random forest estimated coefficients") + theme(plot.title = element_text(hjust = .5,size = 60/.pt))

grid.arrange(lsPlot, enPlot, riPlot, rfPlot, nrow = 4)