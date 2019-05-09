rm(list=ls(all=TRUE))
library(data.table)

### models
library(gbm)
library(PRROC) #roc.curve
library(timeROC) #timeROC curve 
library(glmnet)
library(e1071) #svm and naiveBayes
library(C50)
library(ranger)
library(DMwR) #k-nearest neighbor
library(caret) #k-nearest neighbor
library(kernlab) #gaussian process
library(survival)

### for evaluation
library(PresenceAbsence)
library(PerfMeas)
library(survivalROC)

### for imputation
library(missForest)
library(mice) #
library(VIM) #hotdeck
library(dlsem)
library(bnstruct)

### for plotting
library(caTools)
library(pROC) #roc.test too
library(ggplot2)
library(survminer)

### for parallelization
library(foreach)
library(doParallel)



################################################################################
##################################### data #####################################
################################################################################
source("learnFunctionsProd.R") # functions for learning
load("example data path") ## load data in



################################################################################
################################## Build model #################################
################################################################################

############################ Tuned Hyperparameters #############################
vModel.Parameters <- vector("list", 9)
names(vModel.Parameters) <- c("lr","bn","nb","dt","rf","gbm","svm","gp","knn")
vModel.Parameters[["lr"]] <- list(model_family = "binomial", alpha=1)
vModel.Parameters[["bn"]] <- list()
vModel.Parameters[["nb"]] <- list(laplace=0)
vModel.Parameters[["dt"]] <- list()
vModel.Parameters[["gp"]] <- list(kernel="rbfdot",params=list(sigma=0.1))
vModel.Parameters[["knn"]] <- list(clusters=10)
vModel.Parameters[["svm"]] <- list(gamma=0.2, cost=8, svmkernel="radial")
vModel.Parameters[["rf"]] <- list(var.split=15, minobsinnode=50, ntrees=500)
vModel.Parameters[["gbm"]] <- list(ntrees=300, shrinkage=0.005, 
                                   depth=7, bag=0.5, minobsinnode=30)
vModel.Parameters[["rsf"]] <- list(var.split=15, minobsinnode=50, ntrees=500)
vModel.Parameters[["cox"]] <- list()

rm(vModel.Param.Tune,kModel)

################################################################################
################################ Evaluate model ################################
################################################################################

nFold <- 5;  #number of folds
vModelsEval <- c("lr","dt","svm","rf","gbm","cox","rsf") #models you want to use for evaluation

### creating datatable to save evaluation
vEvalMetrics <- c("auc")
dtOut <- data.table(matrix(nrow=nFold,ncol=length(vEvalMetrics)*length(vModelsEval)))
dtOut[,(names(dtOut)):= lapply(.SD, as.numeric), .SDcols = names(dtOut)]
for (kModel in vModelsEval) {
  if (!exists("vN")) {
    vN <- paste0(kModel,".",vEvalMetrics)  
  } else {
    vN <- c(vN,paste0(kModel,".",vEvalMetrics))
  }
}; rm(kModel)
names(dtOut) <- vN
rm(vN)


################### Evaluate models through cross validation ###################
### set cross validation indices
set.seed(100)
vTestInd <- split(sample(seq(nrow(dtData))),seq(nFold)) #samples for cross validation
varP <- vPredVars0 #predictors
varO <- "death" #event variable
varT <- "fu2" #survival time variable


for (kFold in seq(nFold)) {
  print(paste0("XXXXXXXXXX CrossVal Fold: ", kFold, " XXXXXXXXXX"))
  
  train.data <- dtData[-vTestInd[[kFold]]])
  test.data <- dtData[vTestInd[[kFold]]]
 
  
  ### loop across different models
  for (kModel in vModelsEval) {
    print(paste0("XXXXXXXXX MODEL: ",kModel," XXXXXXXXX"))
    
    ############################### Learn Models ###############################
    ### fit model using wrapper function
    cModel <- funcClassify(train = copy(train.data),
                           test = test.data,
                           outcomeVar = varO,
                           predictorVars = varP,
                           timeVar = varT, 
                           timeThresh = 5,
                           flagOversample=F,
                           methodClassify=kModel,
                           positiveOutcome=1,
                           negativeOutcome=0,
                           h.parameters=vModel.Parameters[[kModel]])
    
    ############################### evaluate fit ###############################
    cAUC <- survivalROC(Stime = dtData[vTestInd[[kFold]],get(varT)],
                        status = dtData[vTestInd[[kFold]],get(varO)],
                        marker = cModel$predicted.test,
                        method="KM",
                        predict.time = 5)$AUC

    print(cAUC)
    set(dtOut,kFold,paste0(kModel,".auc"),cAUC)

    ### reset some variables 
    rm(cModel,cAUC,vTmp)
  } #end of model loop
  
  print(paste0("OO    END Fold: ",kFold,"    OO"))
  rm(train.data,test.data)
} #end of crossval loop
rm(kFold,kModel)
print(c(colMeans(dtOut,na.rm=T),unlist(lapply(dtOut,sd,na.rm=T))))


#################################### Output ####################################
a0 <- colMeans(dtOut,na.rm=T)
b0 <- unlist(lapply(dtOut,sd,na.rm=T))


################################################################################
################################ Plotting model ################################
################################################################################
