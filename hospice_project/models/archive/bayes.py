#%%
import pandas as pd
import pymc3 as pm
from matplotlib import pyplot as plt
from hospice_project.data.transformers import MyScaler
from hospice_project.models.archive.bayes_vis import test_model
from sklearn.pipeline import Pipeline

#%%

var_dict = pd.read_pickle('data/interim/var_dict.pickle')


r_keep_vars = ["H_009_01_OBSERVED",
"EMO_REL_BTR",
"RESPECT_BTR",
"SYMPTOMS_BTR",
"TEAM_COMM_BTR",
"TIMELY_CARE_BTR",
"TRAINING_BTR",
"percBlack",
"percHisp",
"percBen30orFewerDays",
"percSOSDassisLiv",
"nurseVisitCtPB",
"socialWorkCtPB",
"physicianCtPB",
"totalMedStandPayPB",
"Care_Provided_Assisted_Living_Yes",
"Care_Provided_Home_Yes",
"Care_Provided_Inpatient_Hospice_Yes",
"Care_Provided_Skilled_Nursing_Yes"]

X_train = pd.read_pickle('data/processed/X_train.pickle')

#%%
# Loop DF: drop recommend vars and make recommend_bbv Y
bayes_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()]
bayes_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()]

# set up scale/impute pipeline
steps = [('scaler', MyScaler(dummy_vars=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(bayes_x)
X = pipe.transform(bayes_x)
X = pd.DataFrame(X, columns=bayes_x.columns)
X['RECOMMEND_BBV'] = bayes_y
#%%
# Formula

formula = 'RECOMMEND_BBV ~ ' + ' + '.join([str(i) for i in bayes_x.columns])

with pm.Model() as normal_model:
    # The prior for the model parameters will be a normal distribution
    family = pm.glm.families.Normal()

    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data=X)

    # Perform Markov Chain Monte Carlo sampling
    normal_trace = pm.sample(draws=2000, chains=2, tune=500)

#%%
pm.traceplot(normal_trace)
plt.show()

#%%
test_model(normal_trace,X.iloc[8,:], 'RECOMMEND_BBV')

#%%
pm.Beta()