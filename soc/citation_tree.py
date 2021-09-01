#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct

import nlp_utils as nu

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

#Words common in all literature
gen_lit_tw = pd.read_csv('data/gen_literature_top_words.csv',index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values

#%%

all_dois = pd.read_csv('data/tech_doi.csv')
input_dois = all_dois['general'].dropna()

df = nu.io.load_df_semantic(con, input_dois, cust_idx_name='doi')

for doi in input_dois:
    if doi not in df['doi'].values:
        print("Could not find doi: " + str(doi))

#%%

df_com = nu.citation.build_citation_community(df, con, n_iter=1, frac_keep_factor=0.5, n_initial_trim=10000)

#%%
docs = df_com['title'] + ' ' + df_com['paperAbstract']
texts = docs.apply(str.split)

# corex_anchors = [['fuel_cel', 'electrolyz'], ['pump_thermal', 'heat_pump'], 'li_ion']
corex_anchors = []

fixed_bigrams = []
for word in corex_anchors:
    if type(word) == list:
        fixed_bigrams.extend(word)
    else:
        fixed_bigrams.append(word)

fixed_bigrams = [w for w in fixed_bigrams if '_' in w]

#%%
pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    ('vectorizer', CountVectorizer(max_features=None, min_df=2, max_df = 0.9, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

X = pipeline.fit_transform(texts)
feature_names = pipeline['vectorizer'].get_feature_names()

topic_model = ct.Corex(n_hidden=30, seed=42)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_com.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_com['year'], norm_each_topic=False)
#%%
year_range_fit = slice(2015,2020)
year_range_plot = slice(2000,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=20)
#%%
nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, ascending=True)
#%%

topic_num = 24
topic_str = 'topic_' + str(topic_num)
top_docs = topic_model.get_top_docs()[topic_num]

print(topic_str)
print(s_topic_words[topic_str])
print("-"+"\n-".join(df_com['title'].loc[[t[0] for t in top_docs]]))

#%%

df_subset = df_com.loc[[t[0] for t in topic_model.get_top_docs(topic=topic_num)]]

df_com_2 = nu.citation.build_citation_community(df_subset, con, n_iter=5, frac_keep_factor=0.5, n_initial_trim=10000)
