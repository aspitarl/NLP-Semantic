#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

from nlp_utils.fileio import load_df_semantic

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

#%%

all_dois = pd.read_csv('data/tech_doi.csv')
input_dois = all_dois['general'].dropna()

df = load_df_semantic(con, input_dois, cust_idx_name='doi')

for doi in input_dois:
    if doi not in df['doi'].values:
        print("Could not find doi: " + str(doi))

df

#%%

from nlp_utils.citation import build_citation_community
df_com = build_citation_community(df, con, n_iter=3, frac_keep_factor=0.5, n_initial_trim=200)

#%%
docs = df_com['title'] + ' ' + df_com['paperAbstract']
texts = docs.apply(str.split)

gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values


#%%

from sklearn.pipeline import Pipeline
from nlp_utils.text_process import TextNormalizer
from nlp_utils.gensim_utils import Gensim_Bigram_Transformer
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from nlp_utils import corex_utils

bigram_kwargs = {'threshold':20, 'min_count':10}

corex_anchors = [['fuel_cel', 'electrolyz'], ['pump_thermal', 'heat_pump'], 'li_ion']

fixed_bigrams = []

for word in corex_anchors:
    if type(word) == list:
            fixed_bigrams.extend(word)
    else:
        fixed_bigrams.append(word)

fixed_bigrams = [w for w in fixed_bigrams if '_' in w]
fixed_bigrams

#%%

import xarray as xr
import nlp_utils.common as nu_common 
n_topics = [20,30,40,50]
# n_topics = [5,6]

topic_models= []

for n_topic in n_topics:


    pipeline = Pipeline([
        ('text_norm', TextNormalizer(post_stopwords=gen_lit_remove)),
        ('bigram', Gensim_Bigram_Transformer(bigram_kwargs=bigram_kwargs, fixed_bigrams=fixed_bigrams)),
        ('vectorizer', CountVectorizer(max_features=None, min_df=2, max_df = 0.9, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
        # ('corex', ct.Corex(n_hidden=n_topic, anchors = corex_anchors, anchor_strength=1000))
    ])

    X = pipeline.fit_transform(texts)

    feature_names = pipeline['vectorizer'].get_feature_names()

    topic_model = ct.Corex(n_hidden=n_topic)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

    topic_models.append(topic_model)


#%%

dss = []

for topic_model in topic_models:

    n_topic = topic_model.n_hidden

    topic_names = ['topic_' + str(i) for i in range(topic_model.n_hidden)]
    s_topic_words = corex_utils.get_s_topic_words(topic_model, topic_names, n_words=20)

    print(n_topic)
    print(s_topic_words)

    doc_topic_probs = topic_model.p_y_given_x
    df_doc_topic_probs = pd.DataFrame(doc_topic_probs, index=df_com.index , columns=topic_names)

    s_year = pd.Series(df_com['year'], index=df_com.index)

    df_topicsyear = nu_common.calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=False)
    df_topicsyear.index = df_topicsyear.index.astype(int)
    df_topicsyear = df_topicsyear.sort_index()

    year_range = slice(2015,2020)

    avg_prob = df_topicsyear.loc[year_range].mean()
    da_avg_prob = xr.DataArray(avg_prob, name='avg_prob')

    df_fit = nu_common.fit_topic_year(df_topicsyear, year_range=year_range)
    df_fit = df_fit.astype(float)
    da_slopes = xr.DataArray(df_fit['slope'], name='slope')

    da_topic_words = xr.DataArray(s_topic_words, name='topic_words')

    ds = xr.merge([da_slopes, da_topic_words, da_avg_prob]).assign_coords(n_topics=n_topic)
    dss.append(ds)


ds = xr.concat(dss,'n_topics')
ds = ds.rename(dim_0='topic')
ds

# %%
overall_avg_prob = ds['avg_prob'].groupby('n_topics').mean('topic')
ds['avg_prob_norm'] = ds['avg_prob']/overall_avg_prob
overall_avg_slope = abs(ds['slope']).groupby('n_topics').mean('topic')
ds['avg_slope_norm'] = ds['slope']/overall_avg_slope

# %%

anchor_topic_names = ['topic_0','topic_1', 'topic_2']

for i, name in enumerate(anchor_topic_names):
    print("{}: {}".format(name, corex_anchors[i]))

da_metric = ds[['avg_prob_norm','avg_slope_norm']].to_array('var')

da_metric.sel(topic=anchor_topic_names).plot(col='var', hue='topic')

#%%
# ds['slope']


