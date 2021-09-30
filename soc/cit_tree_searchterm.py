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

regex = '%carbon nanotube%'
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

for n_max in [1000, 10000]:

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

import _pickle as cPickle
#Save model
model_name =  'mod_cit_tree.pkl' 
output_folder = r'C:\Users\aspit\Git\NLP-Semantic\soc\output'
cPickle.dump(topic_model, open(os.path.join(output_folder, model_name), 'wb'))

#%%
# from nlp_utils.gensim_utils import basic_gensim_lda
# from nlp_utils import gensim_utils

# pipeline = Pipeline([
#     ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
#     ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
#     # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# ])

# texts_bigram = pipeline.fit_transform(texts)

# n_topics = 50
# alpha = 1/n_topics

# lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
# id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)


# lda_model.texts_bigram = texts_bigram
# lda_model.id2word = id2word
# lda_model.data_words = data_words
# lda_model.idx = df_tm.index.values
# lda_model.save(r'C:\Users\aspit\Git\NLP-Semantic\soc\output\ldamod_cit_tree.lda')

