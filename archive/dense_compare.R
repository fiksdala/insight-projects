library(car)
library(Metrics)
library(MLmetrics)
library(betareg)

df = read.csv('data/processed/RATING_EST.csv')
colnames(df)

m_wr = lm(would_recommend ~ ., data=df)
summary(m_wr)
rmse(df$would_recommend, predict(m_wr, df))
mae(df$would_recommend, predict(m_wr, df))
vif(m_wr)

beta_df = df
beta_df$y = (df$RATING_EST*.01*(nrow(df)-1) + 0.5)/nrow(df)
beta_df$y = beta_df$RATING_EST
beta_df$RATING_EST = NULL

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

beta_kfold = function(df, n_folds){
  rmse_test_out = c()
  mse_test_out = c()
  mae_test_out = c()
  r2_test_out = c()
  rmse_train_out = c()
  mse_train_out = c()
  mae_train_out = c()
  r2_train_out = c()
  folds = f_K_fold(dim(df)[1], n_folds)
  for(fold in folds){
    m = betareg(y ~ .,
                df[fold$train,])
    preds = predict(m, newdata=df[fold$test,])
    train_preds = predict(m, newdata=df[fold$train,])
    rmse_test_out = append(rmse_test_out,
                           rmse(df$y[fold$test], preds))
    mse_test_out = append(mse_test_out,
                          mse(df$y[fold$test], preds))
    r2_test_out = append(r2_test_out,
                         R2_Score(preds, df$y[fold$test]))
    mae_test_out = append(mae_test_out,
                          mae(df$y[fold$test], preds))
    rmse_train_out = append(rmse_train_out,
                            rmse(df$y[fold$train], train_preds))
    mse_train_out = append(mse_train_out,
                           mse(df$y[fold$train], train_preds))
    mae_train_out = append(mae_train_out,
                           mae(df$y[fold$train], train_preds))
    r2_train_out = append(r2_train_out,
                          R2_Score(train_preds, df$y[fold$train]))
  }
  list(test=list(rmse=rmse_test_out, mse=mse_test_out, mae=mae_test_out,
                 r2=r2_test_out),
       train=list(rmse=rmse_train_out, mse=mse_train_out, mae=mae_train_out,
                  r2=r2_train_out))
}

eval = beta_kfold(beta_df, 5)
for(score in eval$test){
  print(mean(score)*100)
}
eval
