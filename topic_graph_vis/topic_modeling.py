import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import sqlite3
import networkx as nx
from nlp_utils.io import load_df_semantic
import nlp_utils as nu

#%%

G = nx.read_gexf('data/G_cit_tree.gexf')
db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

#%%
num_papers_gen_lit_sw = 10000
print('finding top terms in first {} papers for general literature stopwords'.format(num_papers_gen_lit_sw))

cursor.execute("SELECT id FROM raw_text LIMIT {}".format(num_papers_gen_lit_sw))
results = cursor.fetchall()
ids = [t[0] for t in results]

df = load_df_semantic(con, ids)

from nlp_utils.text_process import TextNormalizer
from nlp_utils.text_analysis import top_words

# The text we will analyze is words in both the title and abstract concatenated. 
docs = df['title'] + ' ' + df['paperAbstract']
texts = docs.apply(str.split)

text_normalizer = TextNormalizer()
texts_out = list(text_normalizer.transform(texts))
tw = top_words(texts_out,-1)

tw_word = [t[0] for t in tw]
tw_count = [t[1] for t in tw]

gen_lit_tw = pd.Series(tw_count, tw_word)
gen_lit_remove = gen_lit_tw[0:130].index.values
#
#%%

print("Corex Topic Modeling")
df_tm = load_df_semantic(con, G.nodes)

docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

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

topic_model = ct.Corex(n_hidden=50, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

import _pickle as cPickle
#Save model
if not os.path.exists('models'): os.mkdir('models')
cPickle.dump(topic_model, open(os.path.join('models', 'mod_cit_tree.pkl'), 'wb'))

#%%

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)


from importlib import reload
reload(nu.plot)

highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, highlight_topics=highlight_topics)

plt.savefig('output/top_slopes_plot.png')


nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, highlight_topics=highlight_topics,ascending=True)

plt.savefig('output/neg_slopes_plot.png')



#%%

# print("LDA Topic Modeling")
# from nlp_utils.gensim_utils import basic_gensim_lda
# from nlp_utils import gensim_utils

# pipeline = Pipeline([
#     ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
#     ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
#     # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
# ])

# texts_bigram = pipeline.fit_transform(texts)

# n_topics = 30
# alpha = 1/n_topics

# lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
# id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)


# lda_model.texts_bigram = texts_bigram
# lda_model.id2word = id2word
# lda_model.data_words = data_words
# lda_model.idx = df_tm.index.values
# lda_model.save(r'C:\Users\aspit\Git\NLP-Semantic\soc\output\ldamod_cit_tree.lda')

