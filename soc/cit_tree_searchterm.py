#%%

import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.io import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph_fraction, get_frac_connected, trim_graph_size

# db_folder = r'/media/lee/Shared Storage'
db_folder = r'E:\\'

# %%

con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

regex = '%energy storage%'
ids = nu.io.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e6))
#%%
df = load_df_semantic(con, ids)
df['inCitations'].apply(len).value_counts().sort_index()
# df = df.where(df['inCitations'].apply(len) >100).dropna(how='all')

G = nx.Graph()
G.add_nodes_from(df.index.values, round=0)


## Initial internal citation trim
G = gen_citation_tree(G, df, cit_field='inCitations', only_existing=True)
# G = gen_citation_tree(G, df, cit_field='outCitations', only_existing=True)
print("size of citation tree: {}".format(len(G.nodes)))

G = trim_graph_size(G, 100)
df = load_df_semantic(con, G.nodes)


#%%

for n_max in [1000, 10000, 100000]:

    df = load_df_semantic(con, G.nodes)
    G = gen_citation_tree(G, df, cit_field='inCitations', only_existing=False)
    G = gen_citation_tree(G, df, cit_field='outCitations', only_existing=False)
    print("size of citation tree: {}".format(len(G.nodes)))
    G = trim_graph_size(G, n_max)


#%%
#%%

df_tm = load_df_semantic(con, G.nodes)
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values

corex_anchors = []
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

X = pipeline.fit_transform(texts)
feature_names = pipeline['vectorizer'].get_feature_names()

topic_model = ct.Corex(n_hidden=30, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)
# %%

from importlib import reload
reload(nu.plot)

highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=30, highlight_topics=highlight_topics)


#%%

