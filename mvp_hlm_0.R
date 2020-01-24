library(lme4)
library(lmerTest)
library(Metrics)

df = read.csv('data/processed/mvp.csv')
df$Ownership.Type = relevel(df$Ownership.Type, 'Non-Profit')
head(df)

m0 <- lmer(neg_rec ~ Ownership.Type + 
           days_operation +
           Average_Daily_Census +
           H_001_01_OBSERVED +
           H_002_01_OBSERVED +
           H_003_01_OBSERVED +
           H_004_01_OBSERVED +
           H_005_01_OBSERVED +
           H_006_01_OBSERVED +
           H_007_01_OBSERVED +
           H_008_01_OBSERVED +
           H_009_01_OBSERVED +
           percRuralZipBen +
           aveAge +
           percMale +
           aveHomeHealth7dp +
           percSOSDhome +
           percSOSDassisLiv +
           percSOSlongtermcare +
           percSOSskilledNurse +
           percSOSinpatient +
           percSOSinpatientHospice +
           EMO_REL_BBV +
           EMO_REL_TBV +
           RESPECT_BBV +
           RESPECT_MBV +
           RESPECT_TBV +
           SYMPTOMS_BBV +
           SYMPTOMS_MBV +
           SYMPTOMS_TBV +
           TEAM_COMM_BBV +
           TEAM_COMM_MBV +
           TEAM_COMM_TBV +
           TIMELY_CARE_BBV +
           TIMELY_CARE_MBV +
           TIMELY_CARE_TBV +
           TRAINING_BBV +
           TRAINING_MBV +
           TRAINING_TBV +
           distinctBens +
           epStayCt +
           medStanPayPerBen + (1|State),
           data=df,
           REML=T)
summary(m0)

m1 <- lmer(neg_rec ~ percRuralZipBen +
             percSOSinpatientHospice +
             EMO_REL_BBV +
             RESPECT_BBV +
             RESPECT_TBV +
             SYMPTOMS_BBV +
             TEAM_COMM_BBV +
             TEAM_COMM_MBV +
             TEAM_COMM_TBV +
             TIMELY_CARE_BBV +
             TIMELY_CARE_TBV +
             TRAINING_BBV +
             distinctBens 
             + (1|State),
           data=df[sample(nrow(df), 1630),],
           REML=T)
summary(m1)

saveRDS(m1, 'data/interim/mvp_model.rds')

f_K_fold <- function(Nobs,K=5){
  rs <- runif(Nobs)
  id <- seq(Nobs)[order(rs)]
  k <- as.integer(Nobs*seq(1,K-1)/K)
  k <- matrix(c(0,rep(k,each=2),Nobs),ncol=2,byrow=TRUE)
  k[,1] <- k[,1]+1
  l <- lapply(seq.int(K),function(x,k,d) 
    list(train=d[!(seq(d) %in% seq(k[x,1],k[x,2]))],
         test=d[seq(k[x,1],k[x,2])]),k=k,d=id)
  return(l)
}

lmer_kfold = function(df, call, n_folds){
  rmse_test_out = c()
  mse_test_out = c()
  mae_test_out = c()
  rmse_train_out = c()
  mse_train_out = c()
  mae_train_out = c()
  folds = f_K_fold(dim(df)[1], n_folds)
  for(fold in folds){
    m = lmer(call,
             df[fold$train,],
             REML=T)
    preds = predict(m, newdata=df[fold$test,])
    train_preds = predict(m, newdata=df[fold$train,])
    rmse_test_out = append(rmse_test_out,
                      rmse(df$neg_rec[fold$test], preds))
    mse_test_out = append(mse_test_out,
                     mse(df$neg_rec[fold$test], preds))
    mae_test_out = append(mae_test_out,
                     mae(df$neg_rec[fold$test], preds))
    rmse_train_out = append(rmse_train_out,
                           rmse(df$neg_rec[fold$train], train_preds))
    mse_train_out = append(mse_train_out,
                          mse(df$neg_rec[fold$train], train_preds))
    mae_train_out = append(mae_train_out,
                          mae(df$neg_rec[fold$train], train_preds))
  }
  list(train=list(rmse=rmse_test_out, mse=mse_test_out, mae=mae_test_out),
       test=list(rmse=rmse_train_out, mse=mse_train_out, mae=mae_train_out))
}

kfold_results = lmer_kfold(df, summary(m1)$call, 10)

mean(kfold_results$train$rmse)
mean(kfold_results$test$rmse)

plot(m1)

