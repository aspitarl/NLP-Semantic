#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from nlp_utils.fileio import load_df_semantic

def get_citation(con, paper_id, field='inbound_citations'):

    cursor = con.cursor()

    cursor.execute("SELECT {} from raw_text where paper_id = {}".format(field, paper_id))
    results = cursor.fetchone()

    if results is not None:
        citing_ids = results[0][1:-1].replace('\'', '').split(', ')
        citing_ids = [c for c in citing_ids if len(c)] #Empty string getting returned rarely, TODO: investigate
    else:
        citing_ids = None

    return citing_ids


def get_citation_list(con, paper_ids, field='inbound_citations'):

    citing_ids = []
    for id in paper_ids:
        cits = get_citation(con, id, field=field)
        if cits != None:
            citing_ids.append(cits)

    citing_ids = [j for i in citing_ids for j in i]

    return citing_ids


def gen_citation_tree(con, paper_ids, steps):
    results = {}

    citing_ids = paper_ids

    for i, step in enumerate(steps):

        citing_ids = get_citation_list(con, citing_ids, step)
        results[i] = citing_ids

    print('Num cittions for each step: ')
    print([len(results[key]) for key in results.keys()])

    all_results = []

    for key in results.keys():
        all_results.extend(results[key])

    return all_results

def get_ids_regex(con, regex):

    ids = con.execute("select \"%{}%\" from indexed_searches".format(regex)).fetchall()
    ids = [i[0] for i in ids if i[0] != None]

    return ids

db_path = r'C:\Users\aspit\Git\MLEF-Energy-Storage\s2orc\data\full\db_s2orc.db'

con = sqlite3.connect(db_path)

ids = get_ids_regex(con, 'energy storage') 

df = load_df_semantic(con, ids, dataset='s2orc')


df.info()

#%%

df['n_inbound_citations'] = df['inbound_citations'].apply(len)
df['inbound_cits_per_year'] = df['n_inbound_citations']/df['years_ago']
# %%
df['inbound_cits_per_year'].hist(bins=100)
plt.yscale('log')
# %%
df_sel = df.where(df['inbound_cits_per_year'] > 25).dropna(how='all')

df_sel

#%%

downsel_ids = list(df_sel.index.values)
inbound = get_citation_list(con, downsel_ids, 'inbound_citations')
outbound = get_citation_list(con, downsel_ids, 'outbound_citations')

all_citations = downsel_ids + inbound + outbound
# %%

results = load_df_semantic(con, all_citations, dataset='s2orc')
# %%

results.to_csv('data/citation_tree_results.csv')
# %%

cursor = con.cursor()

paper_ids = [
    '32577851',
    '113639889',
    '52998904',
]

steps = [
    'inbound_citations',
    'outbound_citations',
    'inbound_citations',
]

results = gen_citation_tree(con, paper_ids, steps)

print("Starting Papers")

cursor.execute("SELECT title from raw_text where paper_id in ({})".format(", ".join(paper_ids)))
cursor.fetchall()


print("sample resutls: ")

cursor.execute("SELECT title from raw_text where paper_id in ({})".format(
    ", ".join(results)
    ))

print(cursor.fetchmany(10))


con.close()