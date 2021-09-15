#%%

#%% 
import pandas as pd
import os
import numpy as np
import sys
import gensim
import matplotlib.pyplot as plt
import sqlite3

from nlp_utils.io import load_df_semantic, get_column_as_list

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'db_s2orc.db')
con = sqlite3.connect(db_path)

regex = 'flywheel'
ids = get_column_as_list(con, '%flywheel%', 'indexed_searches')

# ids =pd.read_csv('data/citation_tree_results.csv')['paper_id'].astype(str)

df= load_df_semantic(con, ids, dataset='s2orc')

#%%
from nlp_utils.text_process import text_processing_pipeline

# The text we will analyze is words in both the title and abstract concatenated. 
docs = df['title'] + ' ' + df['abstract']

texts_out = text_processing_pipeline(docs, debug=False)


bigram = gensim.models.Phrases(texts_out, threshold=20, min_count=10)
bigram_mod = gensim.models.phrases.Phraser(bigram)

texts_bigram = [bigram_mod[text] for text in texts_out]

docs_bigram = [" ".join(t) for t in texts_bigram]

from sklearn.feature_extraction.text import CountVectorizer

# maxx_features = 2**12
vectorizer = CountVectorizer(max_features=None, min_df=2, max_df = 0.9)
X = vectorizer.fit_transform(docs_bigram)

feature_names = vectorizer.get_feature_names()
# feature_names


# %%
from corextopic import corextopic as ct
from nlp_utils import corex_utils

topic_model = ct.Corex(n_hidden=30)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=None, anchors=None, anchor_strength=5)

topic_names = ['topic_' + str(i) for i in range(topic_model.n_hidden)]

s_topic_words = corex_utils.get_s_topic_words(topic_model, topic_names)
s_topic_words


# corex_utils.print_top_docs(topic_model, df['title'])


#%%

doc_topic_probs = topic_model.p_y_given_x

df_doc_topic_probs = pd.DataFrame(doc_topic_probs, index=df.index , columns=topic_names)
#%%
import nlp_utils.common as nu_common 

s_year = pd.Series(df['year'], index=df.index)

df_topicsyear = nu_common.calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=False)
# df_topicsyear 

#%%

df_topicsyear.rolling(5).mean().plot()
#%%


year_range = slice(1990, 2020)
# year_range = None

df_topicsyear.index = df_topicsyear.index.astype(int) #TODO: move inside fn

fits = nu_common.fit_topic_year(df_topicsyear, year_range)


#%%
from nlp_utils.plot import top_slopes_plot

topic_strs = s_topic_words

year_range_fit = slice(2000,2020)
year_range_plot = slice(1990,2020)

top_slopes_plot(df_topicsyear.loc[year_range_plot], topic_strs, year_range_fit, n_plots=10)

# %%

top_slopes_plot(df_topicsyear.loc[year_range_plot], topic_strs, year_range_fit, ascending=True, n_plots=10)


# %%
# Train successive layers
tm_layer2 = ct.Corex(n_hidden=3)
tm_layer2.fit(topic_model.labels)

tm_layer2.get_topics()
#%%


for i, topic in enumerate(tm_layer2.get_topics()):
    print("---topic group " + str(i) + "---")
    for num, prob, sign in topic:
        print(topic_strs['topic_' + str(num)])

    print('\n')

# tm_layer3 = ct.Corex(n_hidden=1)
# tm_layer3.fit(tm_layer2.labels)
#
# %%
