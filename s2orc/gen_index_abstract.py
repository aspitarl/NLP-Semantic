#%%

import sqlite3
import pandas as pd

db_path = r'C:\Users\aspit\Git\MLEF-Energy-Storage\s2orc\data\full\db_s2orc.db'

con = sqlite3.connect(db_path)

table_info = con.execute(f'PRAGMA table_info(raw_text);').fetchall()
columns = [t[1] for t in table_info]

#TODO: I wonder if it might be faster doing first pass to see if there were any of the search terms, then search for each term in that subset.

SEARCH_LIMIT = int(1e7)

cursor = con.cursor()

regexes = [
    'energy storage',
    'electricity storage',
    'lithium ion',
    'lead acid',
    'solid oxide fuel cell',
    'compressed air',
    'pumped thermal',
    'thermomechanical',
    'thermal energy storage', #just thermal would probably be a lot of articles...
    'flywheel',
    'superconducting magnetic',
    'supercapacitor',
]

regexes = ['%' + r + '%' for r in regexes]

all_ids = []

for regex in regexes:

    print('Searching for regex: ' + regex)

    # put -- to comment out line
    cursor.execute("""
    SELECT paper_id FROM
    --raw_text
    (SELECT * FROM raw_text LIMIT {})
    WHERE abstract LIKE '{}'
    OR title like '{}'
    """.format(SEARCH_LIMIT,regex,regex)
        )
    ## Just get count, seems to take as long. 
    # cursor.execute("SELECT COUNT(*) FROM raw_text WHERE abstract LIKE '%flywheel energy storage%'")

    results = cursor.fetchall()

    print("Num Results: " + str(len(results)))

    ids = [r[0] for r in results]
    all_ids.append(ids)

df = pd.DataFrame(all_ids).transpose()
df.columns = regexes

df.to_sql('indexed_searches', con, if_exists='replace')

con.close()