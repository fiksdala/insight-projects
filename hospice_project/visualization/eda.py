import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
gdf = pd.read_pickle('data/interim/complete_df.pickle')

#%%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2.5)
ddf = gdf[~gdf['RECOMMEND_BBV'].isna()]
sns.distplot(ddf["RECOMMEND_BBV"], color="red",
             label="Would Not Recommend", bins=10)
sns.distplot(ddf["RECOMMEND_MBV"], color="orange",
             label="Would Probably Recommend", bins=10)
sns.distplot(ddf["RECOMMEND_TBV"], color="green",
             label="Would Definitely Recommend", bins=10)
plt.xlabel('Percent')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2.5)
ddf = gdf[~gdf['RATING_BBV'].isna()]
sns.distplot(ddf["RATING_BBV"], color="red",
             label="Rating 1-6", bins=10)
sns.distplot(ddf["RATING_MBV"], color="orange",
             label="Rating 7-8", bins=10)
sns.distplot(ddf["RATING_TBV"], color="green",
             label="Rating 9-10", bins=10)
plt.xlabel('Percent')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set(font_scale=2)
ddf = gdf[~gdf['SYMPTOMS_BBV'].isna()]
sns.distplot(ddf["SYMPTOMS_BBV"], color="red",
             label="Sometimes or Never Managed Symptoms", bins=10)
sns.distplot(ddf["SYMPTOMS_MBV"], color="orange",
             label="Usually Managed Symptoms", bins=10)
sns.distplot(ddf["SYMPTOMS_TBV"], color="green",
             label="Always Managed Symptoms", bins=10)
plt.xlabel('Percent')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set(font_scale=2)
ddf = gdf[~gdf['MANAGED_SYMPTOMS'].isna()]
sns.distplot(ddf["MANAGED_SYMPTOMS"], bins=12)
plt.xlabel('Percent')
plt.ylabel('Density')
plt.title('Usually or Always Managed Symptoms')
plt.show()

#%%
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.set(font_scale=2.5)
ddf = gdf[~gdf['would_recommend'].isna()]
sns.distplot(ddf["would_recommend"], bins=20)
plt.xlabel('Percent')
plt.ylabel('Density')
plt.title('Would Probably or Definitely Recommend')
plt.show()
