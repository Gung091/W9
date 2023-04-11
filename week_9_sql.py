
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

df_world


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

"""Self-Practicing
---
Query all the data pairs (date, US,JP) with variable="total postings", 自行練習將滿足條件 variable="total postings" 資料 (date, US,JP) 取出

Streamlit Artifact
---
"""

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

"""Notice
---
Streamlit artifact is availed for web-service, 主要是用來發展雲端程式; this means it would only work while the server is up and worked without error, 換句話說必須具備伺服器系統程式正常啟動才可以運作. If you have install a completed python environment, (for instance, Anaconda Jupyter, VS code + Python", 如果你已經安裝完整的 Python 環境，例如 Anaconda Jupyter 或者 VS code + Python, you can activate streamlit artifact, app.py, in console, 可以在文字介面中利用下列方式啟動雲端程式測試:
```shell
streamlit run app.py
```
and test by the IP address: 
```
localhost:8501
```

Code Explanation
---
Two single-select options, one for "variable", and the other for "country" (multiple-selection), 雲端程式包含兩個選項，一個選擇資料型態，一個選擇國家（複選).
"""

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

"""Full Code, app.py
---
```
import pandas as pd
import streamlit as st

import altair as alt
import duckdb

con = duckdb.connect(database='Job.db', read_only=True) 

# Countries
query='''
   SELECT * 
   FROM job
'''
Countries=list(con.execute(query).df().columns)[2:]


st.subheader('Investingation')

col1, col2 = st.columns(2)

with col1:
    query='''
            SELECT 
                 DISTINCT variable
            From job        
            ORDER BY variable       
          '''

    kinds=con.execute(query).df()
    kind = st.selectbox('Kind of Statistics',kinds)
with col2: 
    country = st.selectbox('Country',Countries)
    

result_df = con.execute('''
    SELECT 
        *
    FROM Job 
    WHERE variable=?
    ''', [kind]).df()

chart = alt.Chart(result_df).mark_circle().encode(
    x = 'date',
    y = country,
    #color = 'carrier'
).interactive()
st.altair_chart(chart, theme="streamlit", use_container_width=True)
    
```

Finally
---
1. create a folder, named "streamlit-SQL", 新增一個目錄，命名為 "streamlit-SQL"，
2. copy `app.py, Job.db`, into this folder, 將檔案  `app.py, Job.db` 拷貝到目錄中
3. also create a file, requirements.txt, which includes packages required in artifact, 並且新增一個檔案，requirements.txt，裡面包含雲端程式所需要的函式庫，如下:
```
dockdb
altair
```
4. create new github and push files into repo, 新增給他資源庫，並將上述檔案上傳;
5. sign in streamlit official to create artifact，到 streamlit 官方網站設定遠端程式. 

[Streamlit Artifact](https://cchuang2009-streamlit-sql-app-68vvn2.streamlit.app/)
"""

#  !pip install duckdb-engine sqlalchemy

engine = sa.create_engine('duckdb:///Job.duckdb')

# Write the dataframe to a new SQL table named 'job'
df_world.to_sql('job', con=engine, if_exists='replace', index=False)

con = duckdb.connect(database='Job.duckdb')
#preview_data_query = "SELECT COUNT(*) FROM 'itineraries_snappy.parquet'"
#con.execute(preview_data_query).df()
data_df = pd.read_sql("SELECT * FROM job where variable='total postings'", con,parse_dates=['date'])
data_df

"""Measurement of Similarity of Time-series data
---
In the csv data, https://raw.githubusercontent.com/cchuang2009/streamlit-SQL/main/job.csv
how to measure the similarity of time series among ['US','AU','CA','DE','FR','GB','IE','JP'] columns



1. Dynamic Time Warping (DTW): DTW is a popular technique for measuring the similarity between two time series, even if they have different lengths and speed. It works by finding the optimal alignment between two time series that minimizes the difference between them. DTW can be used to measure the similarity between each pair of columns in the csv data.

2. Pearson Correlation Coefficient: Pearson correlation measures the linear correlation between two time series. It ranges from -1 to 1, where -1 means perfect negative correlation, 0 means no correlation, and 1 means perfect positive correlation. You can compute the Pearson correlation coefficient between each pair of columns in the csv data.
"""


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
