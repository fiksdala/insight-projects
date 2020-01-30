# libraries
library(lme4)
library(lmerTest)
library(tidyverse)
library(car)
library(Metrics)

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
             REML=F)
    preds = predict(m, newdata=df[fold$test,])
    train_preds = predict(m, newdata=df[fold$train,])
    rmse_test_out = append(rmse_test_out,
                           rmse(df$RECOMMEND_BBV[fold$test], preds))
    mse_test_out = append(mse_test_out,
                          mse(df$RECOMMEND_BBV[fold$test], preds))
    mae_test_out = append(mae_test_out,
                          mae(df$RECOMMEND_BBV[fold$test], preds))
    rmse_train_out = append(rmse_train_out,
                            rmse(df$RECOMMEND_BBV[fold$train], train_preds))
    mse_train_out = append(mse_train_out,
                           mse(df$RECOMMEND_BBV[fold$train], train_preds))
    mae_train_out = append(mae_train_out,
                           mae(df$RECOMMEND_BBV[fold$train], train_preds))
  }
  list(train=list(rmse=rmse_test_out, mse=mse_test_out, mae=mae_test_out),
       test=list(rmse=rmse_train_out, mse=mse_train_out, mae=mae_train_out))
}

# read in data
df_train = read.csv('data/processed/to_R_train.csv')
df_val = read.csv('data/processed/to_R_val.csv')
df_full_train = read.csv('data/processed/to_R.csv')
dim(df_train)
dim(df_val)

# val rmse comparison
val_rmse = c()
df_train$RECOMMEND_BBV
lm_perceptions = lm(RECOMMEND_BBV ~
                      H_002_01_OBSERVED +
                      EMO_REL_BTR +
                      RESPECT_BTR +
                      SYMPTOMS_BTR +
                      TEAM_COMM_BTR +
                      TIMELY_CARE_BTR +
                      TRAINING_BTR,
                    data=df_train)
summary(lm_perceptions)
val_rmse['lm_perceptions'] = rmse(df_val$RECOMMEND_BBV, 
                                  predict(lm_perceptions, df_val))
val_rmse

# Base model all predictors (no state)
lm_base = lm(RECOMMEND_BBV ~
               H_001_01_OBSERVED +
               H_002_01_OBSERVED +
               H_003_01_OBSERVED +
               H_004_01_OBSERVED +
               H_005_01_OBSERVED +
               H_006_01_OBSERVED +
               H_007_01_OBSERVED +
               H_008_01_OBSERVED +
               H_009_01_OBSERVED +
               EMO_REL_BTR +
               RESPECT_BTR +
               SYMPTOMS_BTR +
               TEAM_COMM_BTR +
               TIMELY_CARE_BTR +
               TRAINING_BTR +
               percWhite +
               percBlack +
               percHisp +
               percBen30orFewerDays +
               percBen180orFewerDays +
               aveSocialWork7dp +
               percSOSDhome +
               percSOSDassisLiv +
               percSOSlongtermcare +
               nurseVisitCtPB +
               socialWorkCtPB +
               homeHealthCtPB +
               physicianCtPB +
               totalMedStandPayPB +
               tot_med_PB +
               Care_Provided_Assisted_Living_Yes +
               Care_Provided_Home_Yes +
               Care_Provided_Inpatient_Hospice_Yes +
               Care_Provided_Inpatient_Hospital_Yes +
               Care_Provided_Nursing_Facility_Yes +
               Care_Provided_Skilled_Nursing_Yes +
               Care_Provided_other_locations_Yes +
               Provided_Home_Care_and_other_Yes +
               Provided_Home_Care_only_Yes,
             data=df_train)
summary(lm_base)
val_rmse['lm_all'] = rmse(df_val$RECOMMEND_BBV, 
                                predict(lm_base, df_val))

state_anova = aov(RECOMMEND_BBV ~ State, data=df_train)
summary(state_anova)

lmer_full = lmer(RECOMMEND_BBV ~
                   H_001_01_OBSERVED +
                   H_002_01_OBSERVED +
                   H_003_01_OBSERVED +
                   H_004_01_OBSERVED +
                   H_005_01_OBSERVED +
                   H_006_01_OBSERVED +
                   H_007_01_OBSERVED +
                   H_008_01_OBSERVED +
                   H_009_01_OBSERVED +
                   EMO_REL_BTR +
                   RESPECT_BTR +
                   SYMPTOMS_BTR +
                   TEAM_COMM_BTR +
                   TIMELY_CARE_BTR +
                   TRAINING_BTR +
                   percWhite +
                   percBlack +
                   percHisp +
                   percBen30orFewerDays +
                   percBen180orFewerDays +
                   aveSocialWork7dp +
                   percSOSDhome +
                   percSOSDassisLiv +
                   percSOSlongtermcare +
                   nurseVisitCtPB +
                   socialWorkCtPB +
                   homeHealthCtPB +
                   physicianCtPB +
                   totalMedStandPayPB +
                   tot_med_PB +
                   Care_Provided_Assisted_Living_Yes +
                   Care_Provided_Home_Yes +
                   Care_Provided_Inpatient_Hospice_Yes +
                   Care_Provided_Inpatient_Hospital_Yes +
                   Care_Provided_Nursing_Facility_Yes +
                   Care_Provided_Skilled_Nursing_Yes +
                   Care_Provided_other_locations_Yes +
                   Provided_Home_Care_and_other_Yes +
                   Provided_Home_Care_only_Yes +
                   (1|State),
                 data=df_train)
summary(lmer_full)
val_rmse['lmer_all'] = rmse(df_val$RECOMMEND_BBV, 
                            predict(lmer_full, df_val))

# Feature Type Subsets
# Patient Perceptions
lmer_perceptions = lmer(RECOMMEND_BBV ~
                   # H_001_01_OBSERVED +
                   # H_002_01_OBSERVED +
                   # H_003_01_OBSERVED +
                   # H_004_01_OBSERVED +
                   # H_005_01_OBSERVED +
                   # H_006_01_OBSERVED +
                   # H_007_01_OBSERVED +
                   # H_008_01_OBSERVED +
                   H_009_01_OBSERVED +
                   EMO_REL_BTR +
                   RESPECT_BTR +
                   SYMPTOMS_BTR +
                   TEAM_COMM_BTR +
                   TIMELY_CARE_BTR +
                   TRAINING_BTR +
                   # percWhite +
                   # percBlack +
                   # percHisp +
                   # percBen30orFewerDays +
                   # percBen180orFewerDays +
                   # aveSocialWork7dp +
                   # percSOSDhome +
                   # percSOSDassisLiv +
                   # percSOSlongtermcare +
                   # nurseVisitCtPB +
                   # socialWorkCtPB +
                   # homeHealthCtPB +
                   # physicianCtPB +
                   # totalMedStandPayPB +
                   # tot_med_PB +
                   # Care_Provided_Assisted_Living_Yes +
                   # Care_Provided_Home_Yes +
                   # Care_Provided_Inpatient_Hospice_Yes +
                   # Care_Provided_Inpatient_Hospital_Yes +
                   # Care_Provided_Nursing_Facility_Yes +
                   # Care_Provided_Skilled_Nursing_Yes +
                   # Care_Provided_other_locations_Yes +
                   # Provided_Home_Care_and_other_Yes +
                   # Provided_Home_Care_only_Yes +
                   (1|State),
                 data=df_train)
summary(lmer_perceptions)
val_rmse['lmer_perceptions'] = rmse(df_val$RECOMMEND_BBV, 
                            predict(lmer_perceptions, df_val))
val_rmse

# Patients Served
lmer_patients = lmer(RECOMMEND_BBV ~
                          # H_001_01_OBSERVED +
                          # H_002_01_OBSERVED +
                          # H_003_01_OBSERVED +
                          # H_004_01_OBSERVED +
                          # H_005_01_OBSERVED +
                          # H_006_01_OBSERVED +
                          # H_007_01_OBSERVED +
                          # H_008_01_OBSERVED +
                          # H_009_01_OBSERVED +
                          # EMO_REL_BTR +
                          # RESPECT_BTR +
                          # SYMPTOMS_BTR +
                          # TEAM_COMM_BTR +
                          # TIMELY_CARE_BTR +
                          # TRAINING_BTR +
                          # percWhite +
                          percBlack +
                          percHisp +
                          percBen30orFewerDays +
                          # percBen180orFewerDays +
                          # aveSocialWork7dp +
                          # percSOSDhome +
                          # percSOSDassisLiv +
                          # percSOSlongtermcare +
                          # nurseVisitCtPB +
                          # socialWorkCtPB +
                        # homeHealthCtPB +
                        # physicianCtPB +
                        # totalMedStandPayPB +
                        # tot_med_PB +
                        # Care_Provided_Assisted_Living_Yes +
                        # Care_Provided_Home_Yes +
                        # Care_Provided_Inpatient_Hospice_Yes +
                        # Care_Provided_Inpatient_Hospital_Yes +
                        # Care_Provided_Nursing_Facility_Yes +
                        # Care_Provided_Skilled_Nursing_Yes +
                        # Care_Provided_other_locations_Yes +
                        # Provided_Home_Care_and_other_Yes +
                        # Provided_Home_Care_only_Yes +
                        (1|State),
                        data=df_train)
summary(lmer_patients)
rmse(df_val$RECOMMEND_BBV, 
     predict(lmer_patients, df_val))
val_rmse['lmer_patients'] = rmse(df_val$RECOMMEND_BBV, 
                                    predict(lmer_patients, df_val))
val_rmse


# Services Offered
lmer_care_provided = lmer(RECOMMEND_BBV ~
                       # H_001_01_OBSERVED +
                       # H_002_01_OBSERVED +
                       # H_003_01_OBSERVED +
                       # H_004_01_OBSERVED +
                       # H_005_01_OBSERVED +
                       # H_006_01_OBSERVED +
                       # H_007_01_OBSERVED +
                       # H_008_01_OBSERVED +
                       # H_009_01_OBSERVED +
                       # EMO_REL_BTR +
                       # RESPECT_BTR +
                     # SYMPTOMS_BTR +
                     # TEAM_COMM_BTR +
                     # TIMELY_CARE_BTR +
                     # TRAINING_BTR +
                     # percWhite +
                     # percBlack +
                     #   percHisp +
                     #   percBen30orFewerDays +
                       # percBen180orFewerDays +
                       # aveSocialWork7dp +
                       # percSOSDhome +
                       # percSOSDassisLiv +
                       # percSOSlongtermcare +
                       # nurseVisitCtPB +
                       # socialWorkCtPB +
                       # homeHealthCtPB +
                       # physicianCtPB +
                       # totalMedStandPayPB +
                       # tot_med_PB +
                     Care_Provided_Assisted_Living_Yes +
                     Care_Provided_Home_Yes +
                     Care_Provided_Inpatient_Hospice_Yes +
                     # Care_Provided_Inpatient_Hospital_Yes +
                     # Care_Provided_Nursing_Facility_Yes +
                     Care_Provided_Skilled_Nursing_Yes +
                     # Care_Provided_other_locations_Yes +
                     # Provided_Home_Care_and_other_Yes +
                     # Provided_Home_Care_only_Yes +
                     (1|State),
                     data=df_train)
summary(lmer_care_provided)
rmse(df_val$RECOMMEND_BBV, 
     predict(lmer_care_provided, df_val))
val_rmse['lmer_care_provided'] = rmse(df_val$RECOMMEND_BBV, 
                                 predict(lmer_care_provided, df_val))
val_rmse


# Services Delivered
lmer_care_delivered = lmer(RECOMMEND_BBV ~
                            # H_001_01_OBSERVED +
                            # H_002_01_OBSERVED +
                            # H_003_01_OBSERVED +
                            # H_004_01_OBSERVED +
                            # H_005_01_OBSERVED +
                            # H_006_01_OBSERVED +
                            # H_007_01_OBSERVED +
                            # H_008_01_OBSERVED +
                            # H_009_01_OBSERVED +
                            # EMO_REL_BTR +
                            # RESPECT_BTR +
                          # SYMPTOMS_BTR +
                          # TEAM_COMM_BTR +
                          # TIMELY_CARE_BTR +
                          # TRAINING_BTR +
                          # percWhite +
                          # percBlack +
                          #   percHisp +
                          #   percBen30orFewerDays +
                          # percBen180orFewerDays +
                          # aveSocialWork7dp +
                          # percSOSDhome +
                          percSOSDassisLiv +
                          # percSOSlongtermcare +
                          nurseVisitCtPB +
                          socialWorkCtPB +
                          # homeHealthCtPB +
                          physicianCtPB +
                          totalMedStandPayPB +
                          # tot_med_PB +
                          # Care_Provided_Assisted_Living_Yes +
                          #   Care_Provided_Home_Yes +
                          #   Care_Provided_Inpatient_Hospice_Yes +
                            # Care_Provided_Inpatient_Hospital_Yes +
                            # Care_Provided_Nursing_Facility_Yes +
                            # Care_Provided_Skilled_Nursing_Yes +
                            # Care_Provided_other_locations_Yes +
                            # Provided_Home_Care_and_other_Yes +
                            # Provided_Home_Care_only_Yes +
                            (1|State),
                          data=df_train)

summary(lmer_care_delivered)
rmse(df_val$RECOMMEND_BBV, 
     predict(lmer_care_delivered, df_val))
val_rmse['lmer_care_delivered'] = rmse(df_val$RECOMMEND_BBV, 
                                      predict(lmer_care_delivered, df_val))
val_rmse

# Services Delivered
lmer_imp_features = lmer(RECOMMEND_BBV ~
                             H_002_01_OBSERVED +
                           EMO_REL_BTR +
                           RESPECT_BTR +
                           SYMPTOMS_BTR +
                           TEAM_COMM_BTR +
                           TIMELY_CARE_BTR +
                           TRAINING_BTR +
                           percBlack +
                             percHisp +
                             percBen30orFewerDays +
                           percSOSDassisLiv +
                             nurseVisitCtPB +
                             socialWorkCtPB +
                             physicianCtPB +
                             totalMedStandPayPB +
                             Care_Provided_Assisted_Living_Yes +
                             Care_Provided_Home_Yes +
                             Care_Provided_Inpatient_Hospice_Yes +
                             Care_Provided_Skilled_Nursing_Yes +
                             (1|State),
                           data=df_train)

summary(lmer_imp_features)
vif(lmer_imp_features)
rmse(df_val$RECOMMEND_BBV, 
     predict(lmer_imp_features, df_val))
val_rmse['lmer_care_delivered'] = rmse(df_val$RECOMMEND_BBV, 
                                       predict(lmer_imp_features, df_val))
val_rmse

# Get full train estimated RMSE on imp_features
kfold_results = lmer_kfold(df_full_train, summary(lmer_imp_features)$call, 5)
mean(kfold_results$test$rmse)

kfold_results_all = lmer_kfold(df_full_train, 
                               summary(lmer_full)$call, 5)
mean(kfold_results_all$test$rmse)

################################################################################
# Let's try some other stuff

rslopes = lmer(RECOMMEND_BBV ~
                 H_002_01_OBSERVED *
                 EMO_REL_BTR *
                 RESPECT_BTR *
                 SYMPTOMS_BTR *
                 TEAM_COMM_BTR *
                 TIMELY_CARE_BTR *
                 TRAINING_BTR *
                 percBlack *
                 percHisp *
                 percBen30orFewerDays *
                 percSOSDassisLiv *
                 nurseVisitCtPB *
                 socialWorkCtPB *
                 physicianCtPB *
                 totalMedStandPayPB *
                 Care_Provided_Assisted_Living_Yes *
                 Care_Provided_Home_Yes *
                 Care_Provided_Inpatient_Hospice_Yes *
                 Care_Provided_Skilled_Nursing_Yes +
                 (1|State),
               data=df_train)

summary(rslopes)
rmse(df_val$RECOMMEND_BBV, 
     predict(rslopes, df_val))
kfold_results_rslopes = lmer_kfold(df_full_train, 
                               summary(rslopes)$call, 5)
mean(kfold_results_rslopes$test$rmse)
