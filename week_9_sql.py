
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

import pandas as pd


countries=['US','AU','CA','DE','FR','GB','IE','JP']

C=["https://raw.githubusercontent.com/hiring-lab/job_postings_tracker/master/{0}/aggregate_job_postings_{0}.csv".format(country,country) for country in countries]

dfs=[pd.read_csv(c,parse_dates=['date']) for c in C]

df_world=pd.DataFrame()
df_world['date']=dfs[0]['date']
df_world['variable']=dfs[0]['variable']
for i in range(len(countries)):
    df_world[countries[i]]=dfs[i]['indeed_job_postings_index_SA']

df_world.reset_index(inplace=True,drop=True)


import sqlite3
import sqlalchemy as sa

engine = sa.create_engine('sqlite:///job.db')

df_world.to_sql('job', con=engine, if_exists='replace', index=False)

df_world.to_csv("job.csv",index=False)

conn = sqlite3.connect('job.db') 
SQL_df = pd.read_sql("SELECT * FROM job WHERE variable='new postings'", conn, parse_dates=['date'])
SQL_df

SQL_df.info()


import duckdb

con = duckdb.connect(database='job.db', read_only=True)

data_query = "FROM 'job' LIMIT 5"
con.execute(data_query).df()

query="""
SELECT 
     DISTINCT variable
From job        
ORDER BY variable       
"""

con.execute(query).df()


query_1="""
SELECT 
        date,US,AU,CA,DE,FR,GB,IE,JP
From job        
WHERE variable =  'new postings'          
"""

con.execute(query_1).df()



import streamlit as st
import altair as alt

# query for all data
query="""
   SELECT * 
   FROM Job
"""
#date=list(con.execute(query).df().columns)[0]
#kinds=list(con.execute(query).df().columns)[1]
Countries=list(con.execute(query).df().columns)[2:]

con.execute(query).df()



st.subheader('Filters')

col1, col2 = st.columns(2)

with col1:
    query="""
            SELECT 
                 DISTINCT variable
            From job        
            ORDER BY variable       
          """

    kinds=con.execute(query).df()
    kind = st.selectbox('Kind of Statistics',kinds)
with col2: 
    country = st.selectbox('Country',Countries)


#  !pip install duckdb-engine sqlalchemy

engine = sa.create_engine('duckdb:///Job.duckdb')

# Write the dataframe to a new SQL table named 'job'
df_world.to_sql('job', con=engine, if_exists='replace', index=False)

con = duckdb.connect(database='Job.duckdb')
#preview_data_query = "SELECT COUNT(*) FROM 'itineraries_snappy.parquet'"
#con.execute(preview_data_query).df()
data_df = pd.read_sql("SELECT * FROM job where variable='total postings'", con,parse_dates=['date'])
data_df



from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np
from tqdm import tqdm

df_world.index=df_world['date']

columns = ['US','AU','CA','DE','FR','GB','IE','JP'] #注意columns需要被定義

df_weekly = df_world.resample('W').mean()

df_ = df_weekly[columns]

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

df1=pd.DataFrame()
df1[['US','CA']]=df_[['US','CA']]
df1.index=range(len(df1))
path = dtw.warping_path(df1['US'], df1['CA'])
dtwvis.plot_warping(df1['US'], df1['CA'], path, filename="warp1.png")

df1

from PIL import Image

img = Image.open('warp1.png')

width = 800
height = int(width * img.size[1] / img.size[0]) # 保持比例
img = img.resize((width, height))

img.show()

#Image("warp1.png",width=800)  無效使用

dist, paths = dtw.warping_paths(df_['US'], df_['JP'], window=25, psi=2)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(df_['US'], df_['JP'], paths, best_path);

# Select the columns of interest
columns = ['US','AU','CA','DE','FR','GB','IE','JP']
df = df_[columns]

# Compute the DTW distance between each pair of columns
dtw_distances = np.zeros((len(columns), len(columns)))
for i, col1 in tqdm(enumerate(columns), total=len(columns)):
    for j, col2 in enumerate(columns):
        if i == j:
            dtw_distances[i,j] = 0
        else:
            dist, _ = dtw.warping_paths(df[col1], df[col2])
            #dist, _ = dtw_distance(df[col1], df[col2])
            dtw_distances[i,j] = dist

# Print the DTW distances
print('DTW distances:')
print(pd.DataFrame(dtw_distances, columns=columns, index=columns))

from scipy.stats import pearsonr
# Compute the Pearson correlation coefficient between each pair of columns
pearson_corrs = np.zeros((len(columns), len(columns)))
for i, col1 in tqdm(enumerate(columns), total=len(columns)):
    for j, col2 in enumerate(columns):
        if i == j:
            pearson_corrs[i,j] = 1
        else:
            corr, _ = pearsonr(df[col1], df[col2])
            pearson_corrs[i,j] = corr

# Print the Pearson correlation coefficients
print('Pearson correlation coefficients:')
print(pd.DataFrame(pearson_corrs, columns=columns, index=columns))

# Create a heatmap of the Pearson correlation coefficients
plt.figure(figsize=(8, 8))
sns.heatmap(pearson_corrs, annot=True, cmap='coolwarm', xticklabels=columns, yticklabels=columns)
plt.title('Pearson correlation coefficients')

plt.show()

# Create a heatmap of the Pearson correlation coefficients

# Compute the Pearson correlation coefficient between each pair of columns
pearson_corrs = np.zeros((len(columns), len(columns)))
for i, col1 in tqdm(enumerate(columns), total=len(columns)):
    for j, col2 in enumerate(columns):
        if i <= j:
            corr, _ = pearsonr(df_weekly[col1], df_weekly[col2])
            pearson_corrs[i,j] = corr

# Create a heatmap of the Pearson correlation coefficients
plt.figure(figsize=[8,8])
mask = np.zeros_like(pearson_corrs, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(pearson_corrs, mask=~mask, cmap='coolwarm', annot=True, fmt='.2f', square=True, xticklabels=columns, yticklabels=columns)

plt.title('Pearson correlation coefficients',size=16);

