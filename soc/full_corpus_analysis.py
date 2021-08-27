

#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'soc.db')

con = sqlite3.connect(db_path)
cursor = con.cursor()

#%%

# cursor.execute("SELECT id from raw_text ORDER BY RANDOM() LIMIT 100")
cursor.execute("SELECT id FROM raw_text LIMIT 10000")
results = cursor.fetchall()

ids = [t[0] for t in results]
#%%

from nlp_utils.fileio import load_df_semantic

df = load_df_semantic(con, ids)

#%%

df['outCitations'].apply(len).hist(bins=100)
plt.yscale('log')
plt.xscale('log')


#%%


df.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()



#%%
df['incits_year'] = df['inCitations'].apply(len)/df['years_ago']

df['incits_year'].hist(bins=1000)
plt.yscale('log')
plt.xscale('log')
# df

#%%

# I think making a citation map skews the results to a higher average citations per year. 
# TODO: check this. 

df_keep = df.where(df['incits_year'] > 1).dropna(how='all')
df_keep

# %%

# Drop any articles from the citation map
# This ended up droping a tiny fraction 

### df_temp.index.to_series().to_csv('data/cit_map_ids.csv') # line used to output this file

# cit_map_ids = pd.read_csv(r'data/cit_map_ids.csv')['id']
# # idx_keep = [id for id in df.index if id not in cit_map_ids]
# # len(idx_k)

# df = df.drop(index=cit_map_ids, errors='ignore')

#%%

from nlp_utils.text_process import TextNormalizer

from nlp_utils.text_analysis import top_words

df_tm = df

# The text we will analyze is words in both the title and abstract concatenated. 
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

text_normalizer = TextNormalizer()
texts_out = list(text_normalizer.transform(texts))


# %%
texts_out

tw = top_words(texts_out,-1)

tw_word = [t[0] for t in tw]
tw_count = [t[1] for t in tw]

tw = pd.Series(tw_count, tw_word)

tw.to_csv('data/gen_literature_top_words.csv')
# %%
