#%%
import pyper as pr
import pandas as pd

#%%
r = pr.R(use_pandas = True)

#%%
r('library(lme4)')
r("tm = readRDS('data/interim/mvp_model.rds')")

huh = pd.DataFrame(r.get('data.frame(fixef(tm))'))

print(huh)
