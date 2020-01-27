import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%%
df = pickle.load(open('data/interim/full_prelim_df.pickle', 'rb'))

#%%
df.columns
dffull = pickle.load(open('data/interim/neg_rec.pickle', 'rb'))

#%%
dffull.columns[30:]

#%%
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

#%%
ax = sns.scatterplot(x="aveSocialWork7dp", y="RECOMMEND_BBV", data=dffull)
plt.show()

#%%
