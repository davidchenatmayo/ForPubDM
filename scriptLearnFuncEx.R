library(DMwR) #for SMOTE
#library(bnlearn) #bayesian network
library(glmnet) #regularized logistic regression
library(glmnetUtils) #utilities to allow formula interfact for glmnet
library(e1071) #svm
library(ranger) #random forest
library(gbm) #gbm
library(PRROC)

################################################################################
################################## Functions ###################################
################################################################################

funcClassify = function(train, test, outcomeVar, predictorVars,
                        flagOversample=F, methodOver="over",
                        timeVar=NULL, timeThresh=0,
                        methodClassify="gbm",
                        positiveOutcome=1,
                        negativeOutcome=0,
                        ### options for GBM
                        h.parameters=list()) {
  ###################### Wrapper for classification models #######################
  # Wrapper function for classification models. Allows for a consistent interface
  # across all classification models.
  # Also creates ensemble model by averaging predictions
  #
  # Inputs:
  # train: data.table of the training data. Can be a list of data.table if you want
  #        an ensemble model
  # test: data.table of the testing data
  # outcomeVar: name of dependent variable
  # predictorsVars: list of column names in dt that may explain dependent
  #                     variables. Can be a list of list for multiple levels.
  #                     List has to be in reverse temporal order. The first
  #                     element are potential parents of the dependent variables
  #                         predictorVars[[n]][1...k]
  #                                   ...
  #                         predictorVars[[2]][1...k]
  #                         predictorVars[[1]][1...k]
  #                               outcomeVar
  # flagOversample: boolean flag for whether to SMOTE training data
  # timeVar: name of time var (this is to threshold outcome if used, default not
  # used)
  # timeThresh: threshold of time
  # methodClassify: string which specifies what method to use
  #                 Deployed models include "lr","bn","nb","dt","rf","gbm","svm",
  #                 "gp", and "knn"
  # postiveOutcome: define what the positive outcome is
  # h.parameters: hyper parameters for the particular model
  #
  # Outputs:
  # predicted.train: probabilities for training set
  # predicted.test: probabilities for testing set
  # trueValues: true values for testing set
  ################################################################################
  
  ### methods check
  methodClassify <- tolower(methodClassify)
  if (!(methodClassify %in% c("lr","bn","nb","dt","rf","gbm","svm","gp","knn",
                              "rsf","cox","sgbm"))) {
    error("Undeployed model chosen")
  }
  
  ### formula for fitting model
  if (methodClassify %in% c("rsf","cox","sgbm")) {
    cForm <- as.formula(paste0("Surv(",timeVar,",",outcomeVar,"==", as.character(positiveOutcome),") ~ ",
                               paste0(predictorVars,collapse="+")))
  } else { # if classification method
    cForm <- as.formula(paste(outcomeVar," ~ ",paste(unique(unlist(predictorVars)), collapse = "+")))
  }
  
  ### set flag if ensemble
  flagEnsemble <- !is.data.table(train)
  
  ### find number of training datasets
  if (flagEnsemble) {
    nDatasets <- length(train)
    nObsTrain <- nrow(train[[1]])
  } else {
    nDatasets <- 1
    nObsTrain <- nrow(train)
  }
  nObsTest <- nrow(test)
  
  ### initialize output
  vTrainOut <- vector("list", nDatasets)
  dtTestOut <- data.table(matrix(nrow = nObsTest, ncol = nDatasets))
  
  ######################## doing some basic data checks ########################
  ### change the outcome var at the threshold for test set
  if (is.character(timeVar)) {
    set(test,which(test[,get(timeVar)]>timeThresh),(outcomeVar),negativeOutcome)
  }
  
  ### checking to see if an ensemble of models is used
  if (flagEnsemble) {
    ######### have to loop through number of datasets to do data integrity checks
    for (kData in seq(nDatasets)) {
      cDT <- train[[kData]][,c(outcomeVar,timeVar,predictorVars),with=F]
      
      ### change the outcome var at the threshold
      if (is.character(timeVar)) {
        set(cDT,which(cDT[,get(timeVar)]>timeThresh),(outcomeVar),negativeOutcome)
        ### remove censored data if using classification techniques
        if (!(methodClassify %in% c("rsf","cox"))) {
          cDT <- cDT[((timeVar)>timeThresh) | ((outcomeVar)==positiveOutcome)]
        }
      }
      
      ### make sure the outcome variables are correct (either factor or 0/1)
      if ((methodClassify %in% c("lr","bn","rf","gbm","knn")) &
          (is.factor(cDT[,get(outcomeVar)]))) {
        cDT[,(outcomeVar):=as.integer(get(outcomeVar))-1]
      }
      if ((methodClassify %in% c("dt","gp","svm")) & (!is.factor(cDT[,get(outcomeVar)]))) {
        cDT[,(outcomeVar):=as.factor(get(outcomeVar))]
      }
      
      ### oversample the training data if wanted
      if (flagOversample) {
        cDT <- funcSMOTE(cDT, outcomeVar=outcomeVar, method = methodOver)
      }
      
      train[[kData]] <- cDT
      rm(CDT)
    }; rm(kData) #end of dataset loop
    
  } else {
    
    ### keep only relevant data
    train <- train[,c(outcomeVar,timeVar,predictorVars),with=F]
    
    ### change the outcome var at the threshold
    if (is.character(timeVar)) {
      set(train,which(train[,get(timeVar)]>timeThresh),(outcomeVar),negativeOutcome)
      ### remove censored data if using classification techniques
      if (!(methodClassify %in% c("rsf","cox"))) {
        train <- train[((timeVar)>timeThresh) | ((outcomeVar)==positiveOutcome)]
      }
    }
    #print(train)
    ### if outcome variable is binary, it must be integer 0/1 for gbm package
    ### thus have to do this check
    if ((methodClassify %in% c("lr","bn","rf","gbm","svm","knn")) &
        (is.factor(train[,get(outcomeVar)]))) {
      train[,(outcomeVar):=as.integer(get(outcomeVar))-1]
    }
    if ((methodClassify %in% c("dt","gp")) & (!is.factor(train[,get(outcomeVar)]))) {
      train[,(outcomeVar):=as.factor(get(outcomeVar))]
    }
    
    ### oversample the training data if wanted
    if (flagOversample) {
      train <- funcSMOTE(train, outcomeVar=outcomeVar, method = methodOver)
    }
    
  } #end of data quality checks
  
  
  ##################### fit the model on the training data #####################
  for (kData in seq(nDatasets)) {
    ### get data for this bag
    if (flagEnsemble) {
      cDT <- train[[kData]]
    } else {
      cDT <- train
    }
    
    ### select classification algorithm to use
    if (methodClassify=="gbm") { #for gbm
      ### train gbm
      model <- gbm(cForm, data = cDT,
                   distribution='bernoulli',
                   n.trees=h.parameters[['ntrees']],
                   shrinkage=h.parameters[['shrinkage']],
                   interaction.depth=h.parameters[['depth']],
                   n.minobsinnode=h.parameters[['minobsinnode']],
                   bag.fraction=0.5,cv.folds=0,
                   keep.data=F,verbose=F,n.cores=1)
      
      ###predict each dependent variable, given all the parents
      ind.Best <- gbm.perf(model,method="OOB",plot.it=F)
      
      vTrainOut[[kData]] <- predict(model, cDT, ind.Best)
      dtTestOut[,kData] <- predict(model, test, ind.Best)
      
    } else if (methodClassify=="lr") { #for lr
      ### train lr
      possibleError <- tryCatch({
        ### check if any factors, if there are, just do regular glm
        if (any(sapply(cDT,class) %in% c("character","factor","ordinal"))) {
          model <- glm(formula=cForm, data=cDT,
                       family='binomial')
        } else {
          model <- glmnet(formula=cForm, data=cDT,
                          family='binomial',
                          alpha=h.parameters[['alpha']])
        }
      }, error = function(e) {
        print("Error glm")
      })
      
      ###predict each dependent variable, given all the parents
      if (inherits(possibleError, "error")) {
        vTrainOut[[kData]] <- numeric(nrow(cDT))
        vTestOut[,kData] <- numeric(nrow(test))
      } else {
        vTrainOut[[kData]] <- predict(model, newdata=cDT)
        dtTestOut[,kData] <- predict(model, newdata=test)
      }
      
    } else if (methodClassify=="bn") { #for bn
      ### TODO
      ### train bn
      
      ###predict each dependent variable, given all the parents
      vTrainOut[[kData]] <- predict(model, data=cDT)
      dtTestOut[,kData] <- predict(model, data=test)
      
    } else if (methodClassify=="nb") { #for nb
      ### train nb
      model <- naiveBayes(formula=cForm, data=cDT,
                          laplace=h.parameters[['laplace']])
      
      ###predict each dependent variable, given all the parents
      vTrainOut[[kData]] <- predict(model, newdata=cDT, type="raw")[,as.character(positiveOutcome)]
      dtTestOut[,kData] <- predict(model, newdata=test, type="raw")[,as.character(positiveOutcome)]
      
    } else if (methodClassify=="dt") { #for dt
      ### train dt
      model <- C5.0(formula=cForm, data=cDT)
      #print("DT did not converge, setting CF to 0.5")
      #if (model$size==1) {
      #  model <- C5.0(cForm, data = cDT,
      #                control=C5.0Control(CF = 0.5))
      #}
      print("DT still did not converge, setting CF to 0.99")
      if (model$size==1) {
        model <- C5.0(cForm, data = cDT,
                      control=C5.0Control(CF = 0.99))
      }
      
      ###predict each dependent variable, given all the parents
      vTrainOut[[kData]] <- as.vector(predict(model, newdata=cDT, type="prob")[,as.character(positiveOutcome)])
      dtTestOut[,kData] <- as.vector(predict(model, newdata=test, type="prob")[,as.character(positiveOutcome)])
      
    } else if (methodClassify=="gp") { #for gp
      ### train gp
      #rbfdot Radial Basis kernel function "Gaussian"
      #polydot Polynomial kernel function
      #vanilladot Linear kernel function
      #tanhdot Hyperbolic tangent kernel function
      #laplacedot Laplacian kernel function
      #besseldot Bessel kernel function
      #anovadot ANOVA RBF kernel function
      #splinedot Spline kernel
      #sigma inverse kernel width for "rbfdot", "laplacedot".
      #degree, scale, offset for "polydot"
      #scale, offset for "tanhdot"
      #sigma, order, degree for "besseldot".
      #sigma, degree for "anovadot".
      model <- gausspr(x=cForm, data=cDT, type='classification',
                       kernel=h.parameters[['kernel']],
                       kpar=h.parameters[['params']])
      
      ###predict each dependent variable, given all the parents
      vTrainOut[[kData]] <- as.vector(predict(model, newdata=cDT, type="probabilities")[,as.character(positiveOutcome)])
      dtTestOut[,kData] <- as.vector(predict(model, newdata=test, type="probabilities")[,as.character(positiveOutcome)])
      
    } else if (methodClassify=="rf") { #for rf
      ### train rf
      model <- ranger(cForm, data = cDT,
                      mtry=h.parameters[['vars.split']],
                      min.node.size = h.parameters[['minObsinNode']],
                      num.trees=h.parameters[['ntrees']],
                      importance="impurity", probability = T, num.threads=1)
      
      ###predict each dependent variable, given all the parents
      indPos <- which(c(positiveOutcome,negativeOutcome)==cDT[1,get(outcomeVar)])
      vTrainOut[[kData]] <- predict(model, data=cDT)$predictions[,indPos]
      indPos <- which(c(positiveOutcome,negativeOutcome)==test[1,get(outcomeVar)])
      dtTestOut[,kData] <- predict(model, data=test)$predictions[,indPos]
      
    } else if (methodClassify=="svm") { #for svm
      ### train svm
      model <- svm(cForm, data=cDT,
                   type='C-classification',
                   gamma=h.parameters[['gamma']],
                   cost=h.parameters[['cost']],
                   kernel=h.parameters[['svmkernel']],
                   probability=T)
      
      ###predict each dependent variable, given all the parents
      predTrain <- attr(predict(model, cDT, probability=T),'probabilities')
      indPos <- which(attr(predTrain,'dimnames')[[2]]==positiveOutcome)
      vTrainOut[[kData]] <- as.vector(predTrain[,indPos])
      predTest <- attr(predict(model, test, probability=T),'probabilities')
      indPos <- which(attr(predTest,'dimnames')[[2]]==positiveOutcome)
      dtTestOut[,kData] <- as.vector(predTest[,indPos])
      
    } else if (methodClassify=="knn") { #for knn
      ### train knn
      model <- knn3(cForm, data=cDT,
                    k=h.parameters[['clusters']])
      
      ###predict each dependent variable, given all the parents
      vTrainOut[[kData]] <- as.vector(predict(model, newdata=cDT, type="prob")[,as.character(positiveOutcome)])
      dtTestOut[,kData] <- as.vector(predict(model, newdata=test, type="prob")[,as.character(positiveOutcome)])
      
    } else if (methodClassify=="rsf") { #for knn
      ### train knn
      model <- ranger(cForm, data=cDT,
                      mtry=h.parameters[['vars.split']],
                      min.node.size = h.parameters[['minobsinnode']],
                      num.trees=h.parameters[['ntrees']],
                      probability=T,
                      importance="permutation")
      
      ###predict each dependent variable, given all the parents
      cPredictedTest <- predict(model, data=test)#
      cPredictedTrain <- predict(model, data=cDT)
      vTimesTest <- cPredictedTest$unique.death.times
      vTimesTrain <- cPredictedTrain$unique.death.times
      
      vTrainOut[[kData]] <- cPredictedTrain$chf[,which.max(vTimesTrain>timeThresh)]
      dtTestOut[,kData] <- cPredictedTest$chf[,which.max(vTimesTest>timeThresh)]
      
    } else if (methodClassify=="cox") { #for knn
      ### train knn
      model <- coxph(cForm, data=cDT,
                     method='breslow')
      
      ###predict each dependent variable, given all the parents
      cPredictedTest <- survfit(model, test)#
      cPredictedTrain <- survfit(model, cDT)
      vTimesTest <- cPredictedTest$time
      vTimesTrain <- cPredictedTrain$time
      vTrainOut[[kData]] <- 1 - as.vector(cPredictedTrain$surv[which.max(vTimesTrain>timeThresh),])
      dtTestOut[,kData] <- 1 - as.vector(cPredictedTest$surv[which.max(vTimesTest>timeThresh),])
      
    } else if (methodClassify=="sgbm") { #for gbm
      ### train survival gbm
      model <- gbm(cForm, data = cDT,
                   distribution='coxph',
                   n.trees=h.parameters[['ntrees']],
                   shrinkage=h.parameters[['shrinkage']],
                   interaction.depth=h.parameters[['depth']],
                   n.minobsinnode=h.parameters[['minobsinnode']],
                   bag.fraction=0.5,cv.folds=0,
                   keep.data=F,verbose=F,n.cores=1)
      
      ###predict each dependent variable, given all the parents
      ind.Best <- gbm.perf(model,method="OOB",plot.it=F)
      
      vTrainOut[[kData]] <- predict(model, cDT, ind.Best)
      dtTestOut[,kData] <- predict(model, test, ind.Best)#end of classification methods tree
      
      ### if there is a problem, print
      if (uniqueN(dtTestOut[,get(paste0("V",kData))])==1) {
        print(paste0("ERROR singular predictions in: ", kData, " dataset "))
      }
      
    } #end of dataset loop
    
    ### output the data
    return(list(predicted.train = vTrainOut,
                predicted.test = rowMeans(dtTestOut,na.rm=T),
                trueValues = test[,get(outcomeVar)]))
  }
  ############################# End of funcClassify ##############################

  
  
  funcSMOTE = function(dtIn,outcomeVar,method="both") {
    ################################# SMOTE data #################################
    # Wrapper for under/oversampling data
    #
    # Inputs:
    # dtIn: data.table of data
    # outcomeVar: outcome variable
    # method: can be
    #         "both" - both oversamples and undersamples
    #         "under" - only undersamples majority cases
    #         "over" - only oversample minority cases
    #
    # Outputs:
    # dtIn: processed data
    ################################################################################
    
    nObsTrain <- as.integer(nrow(dtIn)/2)
    perOver <- 100
    perUnder <- 100
    
    if (method == "both") {
      perOver <- nObsTrain/table(dtIn[,get(outcomeVar)])[2]*100
      perUnder <- table(dt.training[,get(outcomeVar)])[1]/nObsTrain*100
    } else if (method == "under") {
      perUnder <- table(dt.training[,get(outcomeVar)])[1]/nObsTrain*100
    } else if (method == "over") {
      perOver <- nObsTrain/table(dtIn[,get(outcomeVar)])[2]*100
    }
    
    dtIn <- data.table(SMOTE(form = cForm,
                             data = data.frame(train),
                             perc.over=perOver,
                             perc.under=perUnder))
    
    return(dtIn)
  } #end funcSMOTE
  
  
  
  funcMode <- function(v) {
    ################################### mode #######################################
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
  
  
  ############################### End of Functions ###############################
  