#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'db_s2orc.db')
con = sqlite3.connect(db_path)

#TODO: I wonder if it might be faster doing first pass to see if there were any of the search terms, then search for each term in that subset.
regexes = [
    'energy storage',
    'electricity storage',
    'lithium ion',
    # 'lead acid',
    # 'solid oxide fuel cell',
    # 'compressed air',
    # 'pumped thermal',
    # 'thermomechanical',
    # 'thermal energy storage', #just thermal would probably be a lot of articles...
    # 'flywheel',
    # 'superconducting magnetic',
    # 'supercapacitor',
]

regexes = ['%' + r + '%' for r in regexes]

all_ids = []
for regex in regexes:

    print('Searching for regex: ' + regex)
    ids = nu.fileio.gen_ids_searchterm(con, regex, idx_name='paper_id', search_fields=['abstract', 'title'], search_limit=int(1e5))
    all_ids.append(ids)


df = pd.DataFrame(all_ids).transpose()
df.columns = regexes

# df.to_sql('indexed_searches', con, if_exists='replace')

con.close()

df