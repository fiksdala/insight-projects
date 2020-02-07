df = read.csv('data/processed/final_r.csv')
head(df)

m = lm(y ~ ., data=df)
summary(m)

vif(m)
df$percBen30orFewerDays
