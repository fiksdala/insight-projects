df = read.csv('data/processed/final_r.csv')
colnames(df)
df = read.csv('sanity_check.csv')
m = lm(RECOMMEND_BBV ~ EMO_REL_TBV +
         RESPECT_TBV + SYMPTOMS_TBV + TEAM_COMM_TBV +
         TIMELY_CARE_TBV, data=df)
summary(m)
vif(m)

m = lm(RECOMMEND_BBV ~ EMO_REL_BTR +
         RESPECT_BTR + SYMPTOMS_BTR + TEAM_COMM_BTR +
         TIMELY_CARE_BTR, data=df)
summary(m)
vif(m)


df$percBen30orFewerDays

colnames(df)
