rm(list=ls(all=TRUE))
library(data.table)
library(matrixStats)

pathData <- "H:\\Work\\data\\etc\\Clinical_ICU_Mortality\\set-a\\" #set-b

vVarStatic <- c("RecordID","Age","Gender","Height","ICUType","Weight") 
vVarKeep <- c("RecordID","Age","Gender","Height","Weight","GCS",
            "HR","NIDiasABP","NIMAP","NISysABP","RespRate","Temp","Urine",
            "HCT","BUN","Creatinine","Glucose","HCO3","Mg","Platelets","K",
            "Na","WBC","pH","PaCO2","PaO2","DiasABP","FiO2","MAP","MechVent",
            "SysABP","SaO2","Albumin","ALP","ALT","AST","Bilirubin","Lactate",
            "Cholesterol","TroponinI","TroponinT","ICUType.1","ICUType.2",
            "ICUType.3","ICUType.4")
dtStatic <- data.table(RecordID=NA_integer_,
                       Age=NA_integer_,
                       Gender=NA_integer_,
                       Height=NA_integer_,
                       ICUType=NA_integer_,
                       Weight=NA_integer_)

vFiles <- list.files(path=pathData)
ct <- 1
for (kFile in vFiles[2001:4000]) {
  if (kFile %in% c("140501.txt","140936.txt","141264.txt")) next
  if (!exists("dt1")) {
    dt1 <- fread(paste0(pathData,kFile))
    dtTmp <- copy(dtStatic)
    for (kVar in vVarStatic) {
      set(dtTmp,1,kVar,dt1[Parameter==kVar,Value])
    }
    dt1 <- dt1[which(!(Parameter %in% vVarStatic))]
    dt1 <- reshape(dt1, idvar = "Time", timevar = "Parameter", direction = "wide")
    names(dt1) <- c("Time",sapply(strsplit(names(dt1)[2:ncol(dt1)],"Value."),`[[`,2))
    dtTmp <- dtTmp[rep(1,nrow(dt1))]
    dt1 <- cbind(dtTmp,dt1)
  } else {
    dt2 <- fread(paste0(pathData,kFile))
    dtTmp <- copy(dtStatic)
    for (kVar in vVarStatic) {
      set(dtTmp,1,kVar,dt2[Parameter==kVar,Value])
    }
    dt2 <- dt2[which(!(Parameter %in% vVarStatic))]
    dt2 <- reshape(dt2, idvar = "Time", timevar = "Parameter", direction = "wide")
    names(dt2) <- c("Time",sapply(strsplit(names(dt2)[2:ncol(dt2)],"Value."),`[[`,2))
    dtTmp <- dtTmp[rep(1,nrow(dt2))]
    dt2 <- cbind(dtTmp,dt2)
    
    dt1 <- rbind(dt1,dt2,fill=T)
  }
  rm(dt2,dtTmp)
  ct <- ct + 1
  if ((ct %% 10)==0) {
    print(ct)
  }
}; rm(ct)


set(dt1,which(dt1[,Height]==-1),"Height",NA)
set(dt1,which(dt1[,Weight]==-1),"Weight",NA)
set(dt1,which(dt1[,Gender]==-1),"Gender",NA)
set(dt1,which(is.na(dt1[,MechVent])),"MechVent",-1)
vTmp <- rep(-1,nrow(dt1))
vTmp[dt1[,ICUType]==1] <- 1
dt1[,ICUType.1:=vTmp]
vTmp <- rep(-1,nrow(dt1))
vTmp[dt1[,ICUType]==2] <- 1
dt1[,ICUType.2:=vTmp]
vTmp <- rep(-1,nrow(dt1))
vTmp[dt1[,ICUType]==3] <- 1
dt1[,ICUType.3:=vTmp]
vTmp <- rep(-1,nrow(dt1))
vTmp[dt1[,ICUType]==4] <- 1
dt1[,ICUType.4:=vTmp]
dt1[,ICUType:=NULL]
rm(vTmp)

fwrite(dt1,"H:\\Work\\data\\etc\\Clinical_ICU_Mortality\\train_raw.csv")

vVarCh <- sapply(dt1,class)=='character'
dt1[, (vVarNum):=lapply(.SD,funcNorm), .SDcols=vVarNum]


vID <- unique(dt1[,RecordID])
ct <- 1
for (kID in vID) {
  cDT <- dt1[RecordID==kID,c(vVarKeep,"time2"),with=F] 
  vTimeCt <- table(cDT[,time2])
  vMulti <- names(vTimeCt)[vTimeCt>1]
  
  for (k in vMulti) {
    vInd <- which(cDT[,time2]==k)
    vMeans <- colMeans(cDT[vInd],na.rm=T)
    for (kVar in names(which(!is.nan(vMeans)))) {
      set(cDT,vInd[1],(kVar),vMeans[kVar])
    }
    cDT <- cDT[-vInd[2:length(vInd)]]
    rm(kVar,vInd,vMeans)
  }
  cDT <- na.locf(cDT)
  if (!exists("dt2")) {
    dt2 <- copy(cDT)
  } else {
    dt2 <- rbind(dt2,cDT)
  }
  if ((ct %% 10) == 0) {
    print(ct)
  }
  ct <- ct + 1
  rm(cDT,vTimeCt,vMulti,k)
}; rm(kID)


dt3 <- dt2[,lapply(.SD,funcImp)]
vVarKeep2 <- c("Height","Weight","GCS","HR","NIDiasABP","NIMAP","NISysABP",
             "RespRate","Temp","Urine","HCT","BUN","Creatinine","Glucose",
             "HCO3","Mg","Platelets","K","Na","WBC","pH","PaCO2","PaO2",
             "DiasABP","FiO2","MAP","SysABP","SaO2","Albumin","ALP","ALT",
             "AST","Bilirubin","Lactate","Cholesterol","TroponinI","TroponinT")
dt3 <- dt3[,vVarKeep2,with=F]
names(dt3) <- paste0("miss.",names(dt3))
dt2 <- cbind(dt2,dt3)



#######################################################################




pathData <- "H:\\data\\public\\physionet_ICU\\set-a\\"
vFiles <- list.files(path=pathData)

vStatic <- c("Age","Gender","Height","ICUType","Weight")
vDyn <- c("GCS","HR","NIDiasABP","NIMAP","NISysABP","RespRate","Temp",
          "Urine","HCT","BUN","Creatinine","Glucose","HCO3","Mg","Platelets",
          "K","Na","WBC","pH","PaCO2","PaO2","DiasABP","FiO2","MAP","MechVent",
          "SysABP","SaO2","Albumin","ALP","ALT","AST","Bilirubin","Lactate",
          "Cholesterol","TroponinI","TroponinT")

ct <- 1
for (kFile in vFiles) {
  dtT <- fread(paste0(pathData,kFile))
  dtT[,RecordID:=dtT[Parameter=="RecordID",Value]]
  dtT <- dtT[Parameter!="RecordID"]
  dtT <- dtT[,c("RecordID","Time","Parameter","Value"),with=F]
  
  dtT1 <- dtT[Parameter %in% vDyn]
  dtT2 <- dtT[Parameter %in% vStatic,c("RecordID","Parameter","Value"),with=F]
  
  vInd <- which(dtT2[,Parameter]=="Weight")
  if (length(vInd)>0) {dt
    set(dtT2,which(dtT2[,Parameter]=="Weight")[1],"Value",median(dtT2[Parameter=="Weight",Value]))
  }
  if (length(vInd)>1) {
    vInd <- vInd[-1]
    dtT2 <- dtT2[-vInd]
  }
  dtT2 <- reshape(dtT2,idvar="RecordID",timevar="Parameter",direction="wide")
  
  if (ct==1) {
    dt1 <- copy(dtT1)
    dt2 <- copy(dtT2)
  } else {
    dt1 <- rbind(dt1,dtT1,fill=T)
    dt2 <- rbind(dt2,dtT2,fill=T)
  }
  if ((ct %% 100)==0) {
    print(c(ct,kFile))
  }
  ct <- ct + 1
  rm(dtT,dtT1,dtT2,vInd)
}; rm(ct,kFile)




set(dt2,which(dt2[,Height]==-1),"Height",NA)
set(dt2,which(dt2[,Gender]==-1),"Gender",NA)
set(dt2,which(dt2[,Weight] < 10),"Weight",NA)



######################### make static dataset #########################
pathData <- "H:\\Work\\data\\etc\\Clinical_ICU_Mortality\\set-a\\" #set-b

vVarStatic <- c("RecordID","Age","Gender","Height","ICUType","Weight") 
vVarKeep <- c("RecordID","Age","Gender","Height","Weight","GCS",
              "HR","NIDiasABP","NIMAP","NISysABP","RespRate","Temp","Urine",
              "HCT","BUN","Creatinine","Glucose","HCO3","Mg","Platelets","K",
              "Na","WBC","pH","PaCO2","PaO2","DiasABP","FiO2","MAP","MechVent",
              "SysABP","SaO2","Albumin","ALP","ALT","AST","Bilirubin","Lactate",
              "Cholesterol","TroponinI","TroponinT","ICUType.1","ICUType.2",
              "ICUType.3","ICUType.4")

# initialize datatable
vVars <- c("GCS","HR","NIDiasABP","NIMAP","NISysABP","RespRate","Temp",
           "Urine","HCT","BUN","Creatinine","Glucose","HCO3","Mg","Platelets",
           "K","Na","WBC","pH","PaCO2","PaO2","DiasABP","FiO2","MAP","MechVent",
           "SysABP","SaO2","Albumin","ALP","ALT","AST","Bilirubin","Lactate",
           "Cholesterol","TroponinI","TroponinT")
vVars <- c(paste0(vVars,".mean"),paste0(vVars,".min"),paste0(vVars,".max"),
           paste0(vVars,".Dmean"),paste0(vVars,".Dmin"),paste0(vVars,".Dmax"))
dt1 <- data.table(1)[,(vVars):=NA_real_][,V1:=NULL]
dt1<-dt1[rep(1,4000)]
dt1[,RecordID:=dtY[,RecordID]]


dtAgg <- fread("H:\\data\\public\\physionet_ICU\\dyn_aggregate.csv")
### loop through temporal data and create data
for (kID in dtY[,RecordID]) {
  cInd <- which(dt1[,RecordID]==kID)
  
  ### get data from aggregated data
  cDT <- dtAgg[RecordID==kID,c("Time","Parameter","Value"),with=F]
  # check if there are no records for patient
  if (nrow(cDT)<1) {
    next
  }
  
  ### flatten temporal data by getting mean, min, max in time
  cDT <- reshape(cDT,idvar="Time",timevar="Parameter",direction="wide")
  names(cDT) <- gsub("Value\\.","",names(cDT))
  cDT <- cDT[,-"Time"]
  v1 <- colMeans(cDT,na.rm=T)
  names(v1) <- paste0(names(v1),".mean")
  v2 <- sapply(cDT,function(x) {min(x,na.rm=T)})
  names(v2) <- paste0(names(v2),".min")
  v3 <- sapply(cDT,function(x) {max(x,na.rm=T)})
  names(v3) <- paste0(names(v3),".max")
  v4 <- sapply(cDT,function(x) {mean(diff(na.omit(x)))})
  names(v4) <- paste0(names(v4),".Dmean")
  v5 <- sapply(cDT,function(x) {min(diff(na.omit(x)))})
  names(v5) <- paste0(names(v5),".Dmin")
  v6 <- sapply(cDT,function(x) {max(diff(na.omit(x)))})
  names(v6) <- paste0(names(v6),".Dmax")
  vTmp <- c(v1,v2,v3,v4,v5,v6)
  
  ### fill up data table
  set(dt1,cInd,(names(vTmp)),vTmp)
  
  if ((cInd %% 100) == 0) {
    print(cInd)
  }
  rm(cInd,cDT,v1,v2,v3,v4,v5,v6,vTmp)
}; rm(kID)

### combine with static data
dtStatic <- fread("H:\\data\\public\\physionet_ICU\\static_aggregate.csv")
dt1 <- merge(dt1,dtStatic,by.x="RecordID",by.y="RecordID")
dt1 <- merge(dt1,dtY,by.x="RecordID",by.y="RecordID")
